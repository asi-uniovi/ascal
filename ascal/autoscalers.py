"""
Implement a base class for autoscalers
"""
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from numpy import percentile
from fcma import Fcma, SolvingPars, Allocation, App, RequestsPerTime
from ascal.timedops import TimedOps
from ascal.transition import Command
from ascal.helper import mncf_allocation


class AutoscalerTypes(Enum):
    H_REACTIVE = 1     # Horizontal reactive
    HV_REACTIVE = 2    # Horizontal/vertival reactive
    H_PREDICTIVE = 3   # Horizontal predictive   (not implemented yet)
    HV_PREDICTIVE = 4  # Horizontal/vertical predictive
    H_REACTIVE_HV_REACTIVE   = 5  # Mixed reactive horizontal and horizontal/vertical autoscaling
    H_REACTIVE_HV_PREDICTIVE = 6  # Mixed reactive horizontal and horizontal/vertical autoscaling

class AllocationSolver(Enum):
    FCMA1 = 1         # FCMA algorithm with speed level 1
    FCMA2 = 2         # FCMA algorithm with speed level 2
    FCMA3 = 3         # FCMA algorithm with speed level 3
    FCMA4 = 4         # FCMA algorithm with speed level 4
    FCMA =  FCMA1     # FCMA algorithm with speed level 1
    MNCF = 5          # Minimum Node Cost Fit allocation


@dataclass(frozen=True)
class AutoscalerStatistics:
    """
    Statistics returned by the run() method of autoscalers.
    """
    perf_changed: bool # True if the performance has changed
    billing_changed: bool # True if the allocation has changed
    transition_time: float # Time to calculate the transition
    total_time: float # Time to perform all the autoscaling calculations (includes transition)
    node_recycling_level: float = 0.0 # Node recycling level in [0, 1]
    container_recycling_level: float = 0.0 # Container recycling level in [0,1]

class Autoscaler(ABC):
    """
    Base autoscaler class
    """

    # Constant used to deal with numerical approximations
    _DELTA = 0.000001

    # Very small load
    _DELTA_LOAD = RequestsPerTime(f"{_DELTA} req/s")

    # Invalid recycling value
    INVALID_RECYCLING = -1

    """
    Abstract class for autoscalers.
    """
    def __init__(self, timing_args: TimedOps.TimingArgs | None = None):
        """
        Constructor for the abstract autoscaler. It sets properties common to all the autoscalers.
        :param timing_args: Timings for creation/removal of nodes and containers.
        """
        self.system = None                # Application performances of containers on instances class families
        self.time = 0                    # Current time in seconds. Times start at zero
        self.apps = None                  # Applications
        self.allocation = None            # Current allocation
        if timing_args is None:
            self.timing_args = TimedOps.TimingArgs(0, 0, 0, 0, 0) # All the creation/removal times are zero
        else:
            self.timing_args = timing_args
        self._timedops = TimedOps(self.timing_args) # Event based timing machine
        self._log_path = None # Log path
        self._log_f = None # Log file
        self.prediction_percentile = None
        self.prediction_window = None

    @property
    def log_path(self):
        return self._log_path

    @log_path.setter
    def log_path(self, new_value):
        self._log_path = new_value
        if self._log_path is None:
            self._log_f = None
        else:
            self._log_f = open(self.log_path, "w")
        self._timedops.log = self.log

    def _solve_allocation(self, workloads, solver: AllocationSolver=AllocationSolver.FCMA1) -> Allocation:
        """
        Solve allocation problem for the given application workloads.
        :param workloads: Workloads to calculate the allocation.
        :param solver: Solver used to find an allocation.
        :return: An allocation as a list of nodes.
        """
        if solver in (AllocationSolver.FCMA, AllocationSolver.FCMA1, AllocationSolver.FCMA2, AllocationSolver.FCMA3):
            fcma_speed_level = solver.value
            problem = Fcma(self.system, workloads=workloads)
            solution = problem.solve(SolvingPars(speed_level=fcma_speed_level))
            return [node for _, nodes in solution.allocation.items() for node in nodes]
        elif solver == AllocationSolver.MNCF:
            return mncf_allocation(self.system, workloads)
        else:
            raise ValueError("Invalid allocation solver")

    def log_allocation_summary(self):
        """
        Log a summary with the current allocation.
        """
        self.log(f'Current allocation with {tuple(str(node) for node in self.allocation)}')
        for node in self.allocation:
            for cg in node.cgs:
                self.log(f'  - Allocated {cg.replicas} replicas {cg.cc} on node {str(node)}')

    @staticmethod
    def _set_delta_loads_if_zero(workloads: dict[App, RequestsPerTime]):
        """
        Set workloads to a delta workload if they are zero.
        :param workloads: Dictionary with application's workload.
        """
        for app, workload in workloads.items():
            if workload.magnitude == 0:
                workloads[app] = Autoscaler._DELTA_LOAD

    @staticmethod
    def workload_predictions(predictive_autoscaler: 'Autoscaler',
                             app_workloads: dict[App, [RequestsPerTime]]):
        """
        Calculate the load predictions at times multiple of the configured prediction window.
        :param predictive_autoscaler: A predictive autoscaler
        :param app_workloads: Application workloads.
        """
        predictive_autoscaler.predicted_workloads = {}
        n_seconds = len(list(app_workloads.values())[0])
        time = 0
        while time < n_seconds:
            predictive_autoscaler.predicted_workloads[time] = {}
            for app, workload_values in app_workloads.items():
                workload = \
                    percentile(workload_values[time: min(time + predictive_autoscaler.prediction_window, n_seconds)],
                                      predictive_autoscaler.prediction_percentile)
                predictive_autoscaler.predicted_workloads[time][app] = RequestsPerTime(f'{workload} req/s')
            time += predictive_autoscaler.prediction_window

    @staticmethod
    def _handle_node_removals(commands1: list[Command], commands2: list[Command]):
        """
        A node removal operation in the first command list is deleted if the corresponding node is used
        to allocate containers in the second command list. If this is not the case and the node removal
        operation remains in the first command list, it is then removed from the second command list.
        :param commands1: First list of commands.
        :param commands2: Second list of commands.
        """

        # Nodes used in the second command list
        nodes_used2 = [node for command2 in commands2 for node, _, _ in command2.allocate_containers]

        # Nodes removed in the second command list
        nodes_removed_commands2 = {
            node: command2
            for command2 in commands2
            if len(command2.remove_nodes) > 0
            for node in command2.remove_nodes
        }

        for command1 in commands1[:]:
            if len(command1.remove_nodes) > 0:
                for removed_node1 in command1.remove_nodes:
                    # Check if the node removed in the first cmomand list
                    # was used in allocations in the second command list
                    if removed_node1 in nodes_used2:
                        command1.remove_nodes.remove(removed_node1)
                    elif removed_node1 in nodes_removed_commands2:
                        command2 = nodes_removed_commands2[removed_node1]
                        command2.remove_nodes.remove(removed_node1)
                        if command2.is_null():
                            commands2.remove(command2)
                if command1.is_null():
                    commands1.remove(command1)

    @abstractmethod
    def run(self, workloads: dict[App, RequestsPerTime]) -> AutoscalerStatistics:
        """
        Run autoscaling for 1 second.
        :param workloads: Workload for the applications at the last second.
        :return: The statistics after running autoscaling for 1 second.
        """
        pass

    def log(self, message: str):
        """
        Print the message in the log file.
        :param message: Message to print.
        """
        if self._log_f is not None:
            try:
                self._log_f.write(f'{self.time}:  {message}\n')
            except Exception as e:
                print(f"[LOG ERROR] {e}")
        #print(f'{self.time}: {message}', flush=True)

    def __del__(self):
        """
        Close the log file at the exit.
        """
        if self._log_f is not None:
            self._log_f.close()

    def _transition_execute_sync(self, commands: list[Command], start_time: int=-1,
                                 timedops: TimedOps=None):
        """
        Transition between two allocations executing a list of synchronous commands. The execution consists
        of adding new events to the event list.
        :param commands: Synchronous commands that implement the transition.
        :param start_time: The first command is executed at this time.
        :param timedops: Object with the event list for creating/removing containers and nodes. Defaults to
        the autoscaler timedops object.
        """
        # Execute the transition
        if start_time < 0:
            start_time = self.time
        if timedops is None:
            timedops = self._timedops
        curr_time = start_time
        for command in commands:
            if command.sync_on_nodes_upgrade:
                curr_time = max(curr_time, start_time + self.timing_args.hot_node_scale_up_time)
            if command.sync_on_nodes_creation:
                curr_time = max(curr_time, start_time + self.timing_args.node_creation_time)
            if len(command.upgrade_nodes) > 0:
                for initial_node, final_ic in command.upgrade_nodes:
                    timedops.upgrade_node(curr_time, initial_node, final_ic)
            if len(command.create_nodes) > 0:
                for node in command.create_nodes:
                    node.clear()
                    timedops.create_node(curr_time, node)
                    self.allocation.append(node)
            if len(command.remove_containers) > 0:
                for node, cc, replicas in command.remove_containers:
                    timedops.remove_container_replicas(curr_time, cc, replicas, node)
                curr_time += self.timing_args.container_removal_time
            if len(command.remove_nodes) > 0:
                for node in command.remove_nodes:
                    timedops.remove_node(curr_time, node)
            if len(command.allocate_containers) > 0:
                for node, cc, replicas in command.allocate_containers:
                    timedops.allocate_container_replicas(curr_time, cc, replicas, node)
                curr_time += self.timing_args.container_creation_time
