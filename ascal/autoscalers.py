"""
Implement several autoscalers
"""

from time import time as current_time
from enum import Enum
from math import ceil
from abc import ABC, abstractmethod

from numpy import percentile as percentile
from fcma import (
    Fcma,
    App,
    SolvingPars,
    Allocation,
    Vm,
    ContainerClass,
    InstanceClassFamily,
    ContainerGroup,
    RequestsPerTime,
)
from ascal.timedops import TimedOps
from ascal.nodestates import NodeStates
from ascal.transition import Transition, Command
from ascal.helper import get_min_max_load


class AutoscalerTypes(Enum):
    H_REACTIVE = 1     # Horizontal reactive
    HV_REACTIVE = 2    # Horizontal/vertival reactive
    H_PREDICTIVE = 3   # Horizontal predictive   (not implemented yet)
    HV_PREDICTIVE = 4  # Horizontal/vertical predictive
    H_REACTIVE_HV_REACTIVE   = 5  # Mixed reactive horizontal and horizontal/vertical autoscaling
    H_REACTIVE_HV_PREDICTIVE = 6  # Mixed reactive horizontal and horizontal/vertical autoscaling
    FCMA =  10         # FCMA algorithm with speed level 1
    FCMA1 = 11         # FCMA algorithm with speed level 1
    FCMA2 = 12         # FCMA algorithm with speed level 2
    FCMA3 = 13         # FCMA algorithm with speed level 3

class Autoscaler(ABC):
    """
    Base autoscaler class
    """

    # Constant used to deal with numerical approximations
    _DELTA = 0.000001

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

    def log_allocation_summary(self):
        """
        Log a summary with the current allocation.
        """
        self.log(f'Current allocation with {tuple(str(node) for node in self.allocation)}')
        for node in self.allocation:
            for cg in node.cgs:
                self.log(f'  - Allocated {cg.replicas} replicas {cg.cc} on node {str(node)}')

    @staticmethod
    def workload_predictions(predictive_autoscaler: 'HVPredictiveAutoscaler',
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
                    # Check if the node removed in the first comand list
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
    def run(self, workloads: dict[App, RequestsPerTime]) -> tuple[bool, bool, float]:
        """
        Run autoscaling for 1 second.
        :param workloads: Workload for the applications at the last second.
        """
        pass

    def log(self, message: str):
        """
        Print the message in the log file.
        :param message: Message to print.
        """
        if self._log_f is not None:
            self._log_f.write(f'{self.time}:  {message}\n')
        print(f'{self.time}: {message}', flush=True)

    def __del__(self):
        """
        Close the log file at the exit.
        """
        if self._log_f is not None:
            self._log_f.close()

    def _get_app_ccs(self, app_aggs: dict[App, list[int]]) -> dict[App, list[int]]:
        """
        Get a list of container classs for each application in the system. Container classes are sorted
        by decreasing aggregation level.
        :param app_aggs: Application aggregations. Use None to consider all the application
        aggregations in system.
        :return: A dictionary with a list of containers for each application.
        """
        # Container classes for applications sorted by decreasing aggregation
        app_ccs = {}
        icf = list(self.system.keys())[0][1]
        if app_aggs is None:
            aggs = {app_icf[0]: perf.aggs for app_icf, perf in self.system.items()}
        else:
            aggs = app_aggs
        for app in aggs:
            app_ccs[app] = []
            for agg in aggs[app]:
                cc = ContainerClass(
                    app=app,
                    ic=None,
                    fm=icf,
                    cores=self.system[(app, icf)].cores * agg,
                    mem=self.system[(app, icf)].mem[0],
                    perf=self.system[(app, icf)].perf * agg,
                    agg_level=agg,
                    aggs=tuple(aggs[app])
                )
                app_ccs[app].append(cc)
            app_ccs[app].sort(key=lambda c: c.agg_level, reverse=True)
        return app_ccs

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
            if command.sync_on_nodes_creation:
                curr_time = max(curr_time, start_time + self.timing_args.node_creation_time)
            if len(command.create_nodes) > 0:
                for node in command.create_nodes:
                    node.free_cores = node.ic.cores
                    node.free_mem = node.ic.mem
                    node.cgs.clear()
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

class HReactiveAutoscaler(Autoscaler):
    """
    Horizontal and reactive autoscaler for containers and nodes.
    """
    def __init__(self, time_period:int = 60, desired_cpu_utilization: float = 0.6,
                 node_utilization_threshold:float = 0.5, aggs: dict[App, int] = None,
                 timing_args: TimedOps.TimingArgs | None = None):
        """
        Constructor for the horizontal and reactive autoscaler.
        :param time_period: Time period to evaluate a new autoscaling.
        :param desired_cpu_utilization: Desired CPU utilization for the application containers.
        :param node_utilization_threshold: Below this threshold, a node is tried to be removed.
        :param aggs: The aggregation level for each application. A None value allows the use of the
        application aggregation levels.
        :param timing_args: Timings for creation/removal of nodes and containers.
        """
        super().__init__(timing_args)
        self.time_period = time_period
        self.desired_cpu_utilization = desired_cpu_utilization
        self.node_utilization_threshold = node_utilization_threshold
        self._app_loads: dict[App, list[RequestsPerTime]] = {} # Application workloads in a time period
        self._icf: InstanceClassFamily = None # Instance class family
        self._app_ccs: dict[App: list[ContainerClass]] = {} # Application container classes
        self._desired_app_replicas: dict[App, int] = {} # Desired application replicas
        self._enable_node_creation = True # Set to enable node creation
        self._enable_node_removal = True # Set to enable node removal
        self._enable_container_allocation = True # Set to enable container allocation
        self._enable_container_removal = True # Set to enable container removal
        self._new_nodes_required = False # True if new nodes are required to allocate more recplicas
        self._aggs: dict[App, list[int]] = aggs

    def _initial_allocation(self, workloads: dict[App, RequestsPerTime]) -> Allocation:
        """
        Initial allocation for all the applications, based on their first workload.
        Creation/removal times for nodes and containers are assumed to be zero in the initial alocation.
        :param workloads: First workload sample for each application.
        :return: The initial allocation.
        """
        cgs = []
        for app in workloads:
            workload = workloads[app] / self.desired_cpu_utilization
            for cc in self._app_ccs[app]:
                if cc == self._app_ccs[app][-1]:
                    replicas = int(ceil((workload / cc.perf).magnitude))
                else:
                    replicas = int((workload // cc.perf).magnitude)
                if replicas > 0:
                    cgs.append(ContainerGroup(cc, replicas))
                    workload -= replicas * cc.perf
                    if workload.magnitude == 0:
                        break

        required_nodes = self._get_required_nodes(cgs, allocate=True)
        for node in required_nodes:
            NodeStates.set_state(node, NodeStates.READY)
        self.log(f'Initial allocation with {tuple(str(node) for node in required_nodes)}')
        for node in required_nodes:
            for cg in node.cgs:
                self.log(f'  - Allocated {cg.replicas} replicas {cg.cc} on node {str(node)}')

        return required_nodes

    def _get_replicas(self, app: App, node: Vm = None) -> int:
        """
        Get the number of equivalent agg=1 replicas of an application in the current allocation.
        Replicas that are in the process of being removed are ignored.
        :param app: Application.
        :param node: Restrict to this node.
        :return: Number of replicas.
        """
        if node is None:
            nodes = [n for n in self.allocation if NodeStates.get_state(n) == NodeStates.READY]
        elif NodeStates.get_state(node) != NodeStates.READY:
            return 0
        else:
            nodes = [node]
        return sum(
            cg.replicas * cg.cc.agg_level
            for node in nodes
            for cg in node.cgs
            if cg.cc.app == app
        )

    def _get_required_nodes(self, cgs: list[ContainerGroup], allocate:bool = False) -> Allocation:
        """
        Get the required nodes to allocate the containers.
        :param cgs: Container groups, defined by container classes and number of replicas.
        :param allocate: In addition to get the required nodes, allocate containers on the nodes.
        :return: A list with the required nodes.
        """
        required_nodes = []

        # Sort available instance classes by decreasing number of resources
        ics = [ic for ic in self._icf.ics]
        ics_value = {ic: ic.cores.magnitude * ic.mem.magnitude for ic in self._icf.ics}
        ics.sort(key=lambda ic: ics_value[ic], reverse=True)

        # Simulate allocation using the minimum number of the biggest instance class
        new_node = Vm(ics[0])
        required_nodes.append(new_node)
        for cg in cgs:
            replicas = cg.replicas
            while replicas > 0:
                # Allocate as many replicas as possible
                allocatable_replicas_cpu = int((new_node.free_cores / cg.cc.cores).magnitude + Autoscaler._DELTA)
                allocatable_replicas_mem = int((new_node.free_mem / cg.cc.mem[0]).magnitude + Autoscaler._DELTA)
                allocated_replicas = min(replicas, allocatable_replicas_cpu, allocatable_replicas_mem)
                if allocated_replicas > 0:
                    new_node.free_cores -= allocated_replicas * cg.cc.cores
                    new_node.free_mem -= allocated_replicas * cg.cc.mem[0]
                    new_node.cgs.append(ContainerGroup(cg.cc, allocated_replicas))
                    replicas -= allocated_replicas
                else:
                    new_node = Vm(ics[0])
                    required_nodes.append(new_node)
        # Try to reduce the cost of the latest added virtual machine
        lowest_cost_ic = new_node.ic
        cpu_usage = new_node.ic.cores - new_node.free_cores
        mem_usage = new_node.ic.mem - new_node.free_mem
        for ic in ics:
            if ic.cores >= cpu_usage and ic.mem >= mem_usage and ic.price < lowest_cost_ic.price:
                lowest_cost_ic = ic
        # If a cheaper instance class is found with enough capacity
        if lowest_cost_ic != new_node.ic:
            last_node = Vm(lowest_cost_ic)
            last_node.cgs = new_node.cgs
            last_node.free_cores = last_node.ic.cores - cpu_usage
            last_node.free_mem = last_node.ic.mem - mem_usage
            required_nodes[-1] = last_node
        # Remove all the containers of the required nodes when allocation is not required
        if not allocate:
            for node in required_nodes:
                node.clear()

        return required_nodes

    def _remove_excess_of_replicas(self):
        """
        Reduce the surplus number of application's replicas.
        """
        # Number of equivalent 1x replicas, i.e., with aggregation level 1
        replicas_to_remove = {
            app: self._get_replicas(app) - self._desired_app_replicas[app]
            for app in self.apps
            if self._get_replicas(app) - self._desired_app_replicas[app] > 0
        }
        if len(replicas_to_remove) <= 0:
            return

        # Firstly, sort the nodes by increasing size, so replicas are tried to be removed from the
        # smallest nodes to reduce cluster fragmentation
        nodes = [node for node in self.allocation if NodeStates.get_state(node) == NodeStates.READY]
        nodes_size = {node: node.ic.cores.magnitude * node.ic.mem.magnitude for node in nodes}
        nodes.sort(key=lambda node: nodes_size[node])

        for app, replicas in replicas_to_remove.items():
            for node in nodes:
                # Select container groups allocating the application containers, sorted by decreasing
                # aggregation levels
                cgs = [cg for cg in node.cgs if cg.cc.app == app]
                cgs.sort(key=lambda cg: cg.cc.agg_level, reverse=True)
                while len(cgs) > 0:
                    replicas_to_remove = replicas // cgs[0].cc.agg_level
                    if replicas_to_remove > 0:
                        removed_replicas = \
                            self._timedops.remove_container_replicas(self.time, cgs[0].cc, replicas_to_remove, node)
                        replicas -= removed_replicas * cgs[0].cc.agg_level
                        if replicas == 0:
                            break
                    cgs.pop(0)

    def _allocate_deficit_replicas(self) -> list[ContainerGroup]:
        """
        Try allocating replicas for those applications with a deficit.
        :return: The replicas that can not be allocated.
        """
        # Number of equivalent agg=1 replicas
        replicas_to_add = {
            app: self._desired_app_replicas[app] - self._get_replicas(app)
            for app in self.apps
            if self._desired_app_replicas[app] - self._get_replicas(app) > 0
        }
        if len(replicas_to_add) == 0:
            return []

        if not self._enable_container_allocation:
            return replicas_to_add

        cgs = [] # List of container groups that can not be allocated with the current nodes
        nodes = [node for node in self.allocation if NodeStates.get_state(node) == NodeStates.READY]
        for app, replicas in replicas_to_add.items():
            for cc in self._app_ccs[app]:
                for node in nodes:
                    replicas_to_allocate = replicas // cc.agg_level
                    allocated_replicas = self._timedops.allocate_container_replicas(self.time, cc,
                                                                                    replicas_to_allocate, node)
                    replicas -= allocated_replicas * cc.agg_level
                    if replicas == 0:
                        break
                if replicas == 0:
                    break
            if replicas > 0:
                cgs.append(ContainerGroup(self._app_ccs[app][-1], replicas))
        return cgs

    def _create_required_nodes(self, cgs: list[ContainerGroup]) -> bool:
        """
        Create new nodes to allocate the remaining container replicas.
        :param cgs: Container groups with the replicas to allocate.
        :return: True if new nodes need to be created.
        """
        # Ignore those containers that can be allocated on nodes that are not ready yet
        cgs_to_allocate = [cg for cg in cgs]
        no_ready_nodes = [
            node
            for node  in self.allocation
            if NodeStates.get_state(node) in [NodeStates.BOOTING, NodeStates.BILLED]
        ]
        cgs_allocated = []
        for cg in cgs:
            cc = cg.cc
            replicas = cg.replicas
            for node in no_ready_nodes:
                cpu_allocatable_replicas = int(node.free_cores.magnitude / cc.cores.magnitude + Autoscaler._DELTA)
                mem_allocatable_replicas = int(node.free_mem.magnitude / cc.mem[0].magnitude + Autoscaler._DELTA)
                allocatable_replicas = min(cpu_allocatable_replicas, mem_allocatable_replicas, replicas)
                replicas -= allocatable_replicas
                cg.replicas -= allocatable_replicas
                node.free_cores -= allocatable_replicas * cc.cores
                node.free_mem -= allocatable_replicas * cc.mem[0]
                cgs_allocated.append((node, ContainerGroup(cc, allocatable_replicas)))
                if cg.replicas == 0:
                    cgs_to_allocate.remove(cg)
                    break

        # Recover the allocation state of no ready nodes
        for node in no_ready_nodes:
            node.cgs = []
            node.free_cores = node.ic.cores
            node.free_mem = node.ic.mem

        # Create the nodes
        if len(cgs_to_allocate) > 0 and self._enable_node_creation:
            new_nodes = self._get_required_nodes(cgs)
            for new_node in new_nodes:
                self._timedops.create_node(self.time, new_node)
            self.allocation.extend(new_nodes)
            # Try to allocate replicas, since new ready nodes may be available inmediately when
            # the creation time is zero.
            self._allocate_deficit_replicas()

        return len(cgs_to_allocate) > 0

    def _remove_low_utilization_nodes(self) -> None:
        """
        Try to remove nodes with CPU and memory utilization below the utilization threshold.
        """

        # Nodes can not be removed while creating new nodes
        for node in self.allocation:
            node_state = NodeStates.get_state(node)
            if node_state == NodeStates.BOOTING or node_state == NodeStates.BILLED:
                return 0

        # Firstly, try to remove the smallest nodes to reduce cluster fragmentation
        # Only nodes in the ready state are elegible
        nodes = [node for node in self.allocation if NodeStates.get_state(node) == NodeStates.READY]
        nodes_size = {node: node.ic.cores.magnitude * node.ic.mem.magnitude for node in nodes}
        nodes.sort(key=lambda n: nodes_size[n])

        for node in nodes:
            # Check the threshold utilization condition
            if node.free_cores / node.ic.cores > self.node_utilization_threshold and \
                    node.free_mem / node.ic.mem > self.node_utilization_threshold:
                # If the node is empty
                if len(node.cgs) == 0:
                    self._timedops.remove_node(self.time, node)
                    continue
                # If the node is not empty, try to allocate its containers into other nodes
                # The number of replicas to allocate are the number of equivalent agg=1 replicas
                replicas_to_allocate = {app: self._get_replicas(app, node) for app in self.apps}
                # Free capacity in other nodes and replicas to allocate
                other_nodes_free_capacity = {
                    other_node: [other_node.free_cores, other_node.free_mem]
                    for other_node in nodes if other_node != node
                }

                # Simulate the allocation of the replicas in other nodes

                # Get equivalent agg=1 application replicas, including those that are starting and ignoring those
                # that are being removed
                apps = {app for app in replicas_to_allocate.keys()}
                for app in apps:
                    for cc in self._app_ccs[app]:
                        if replicas_to_allocate[app] < cc.agg_level:
                            continue
                        for other_node in other_nodes_free_capacity:
                            allocatable_replicas_cpu = int(
                                (other_nodes_free_capacity[other_node][0] / cc.cores).magnitude +  Autoscaler._DELTA
                            )
                            allocatable_replicas_mem = int(
                                (other_nodes_free_capacity[other_node][1] / cc.mem[0]).magnitude + Autoscaler._DELTA
                            )
                            allocatable_replicas_agg = (replicas_to_allocate[app] // cc.agg_level) * cc.agg_level
                            allocatable_replicas = min(allocatable_replicas_cpu, allocatable_replicas_mem,
                                                       allocatable_replicas_agg)
                            if allocatable_replicas > 0:
                                other_nodes_free_capacity[other_node][0] -= allocatable_replicas * cc.cores
                                other_nodes_free_capacity[other_node][1] -= allocatable_replicas * cc.mem[0]
                                replicas_to_allocate[app] -= allocatable_replicas * cc.agg_level
                                if replicas_to_allocate[app] == 0:
                                    break
                        if replicas_to_allocate[app] == 0:
                            del replicas_to_allocate[app]
                            break

                # If all the node replicas can be allocated in other nodes, then allocate the replicas
                # and remove the node
                if len(replicas_to_allocate) == 0 and self._enable_container_allocation:
                    # Prevent node from being used
                    NodeStates.set_state(node, NodeStates.REMOVING)
                    self._timedops.timed_log(self.time, f'Moving containers from node {node} to other nodes')
                    # Get equivalent agg=1 application replicas, including those that are starting and ignoring those
                    # that are being removed
                    replicas_to_allocate = {app: self._get_replicas(app, node) for app in self.apps}
                    apps = {app for app in replicas_to_allocate.keys()}
                    for app in apps:
                        for cc in self._app_ccs[app]:
                            if replicas_to_allocate[app] < cc.agg_level:
                                continue
                            for other_node in other_nodes_free_capacity:
                                allocatable_replicas_cpu = int(
                                    (other_nodes_free_capacity[other_node][0] / cc.cores).magnitude + Autoscaler._DELTA
                                )
                                allocatable_replicas_mem = int(
                                    (other_nodes_free_capacity[other_node][1] / cc.mem[0]).magnitude + Autoscaler._DELTA
                                )
                                allocatable_replicas_agg = (replicas_to_allocate[app] // cc.agg_level) * cc.agg_level
                                allocatable_replicas = min(allocatable_replicas_cpu, allocatable_replicas_mem,
                                                           allocatable_replicas_agg)
                                allocated_replicas = \
                                    self._timedops.allocate_container_replicas(self.time, cc,
                                                                               allocatable_replicas, other_node)
                                assert allocatable_replicas == allocated_replicas, "Can not allocate replicas"
                                replicas_to_allocate[app] -= allocated_replicas * cc.agg_level
                                if replicas_to_allocate[app] == 0:
                                    break
                            if replicas_to_allocate[app] == 0:
                                del replicas_to_allocate[app]
                                break
                    # Remove containers in the node
                    cgs = [cg for cg in node.cgs]
                    for cg in cgs:
                        # If the application is not being removed at this time
                        if cg.cc.app is not None:
                            new_time = self.time + self.timing_args.container_creation_time
                            self._timedops.remove_container_replicas(new_time, cg.cc, cg.replicas, node)
                    # Start the node removal after the allocation of the moved containers
                    new_time = self.time + self.timing_args.container_creation_time + \
                                self.timing_args.container_removal_time
                    self._timedops.remove_node(new_time, node)

    def _clear_removed_nodes(self):
        """
        Clear the removed nodes.
        """
        nodes_to_clear = [node for node in self.allocation if NodeStates.get_state(node) == NodeStates.REMOVED]
        for node in nodes_to_clear:
            self.allocation.remove(node)

    def _set_desired_replicas(self):
        """
        Set the desired number of equivalent agg=1 replicas for each application.
        """
        for app, icf in self.system:
            replica_perf = self.system[(app, icf)].perf
            current_replicas = self._get_replicas(app)
            average_load = sum(self._app_loads[app][-self.time_period:]) / self.time_period
            average_cpu_utilization = (average_load / (replica_perf * current_replicas)).magnitude
            self._desired_app_replicas[app] = \
                ceil(current_replicas * average_cpu_utilization / self.desired_cpu_utilization)
            self._timedops.timed_log(self.time,
                                     f'Load of {app}: {average_load.to("req/s").magnitude:.2f} req/s')
            self._timedops.timed_log(self.time, f'Current replicas 1x of {app} {current_replicas}, '
                                                 f'desired {self._desired_app_replicas[app]}')

    def run(self, app_workloads: dict[App, RequestsPerTime]) -> tuple[bool, bool, float]:
        """
        Simulate horizontal reactive autoscaling of containers and nodes in the one second.
        :param app_workloads: Workload for all the applications at the current time.
        :return: A tuple with boolean values for performance changes, billing changes and processing time.
        """
        initial_time = current_time() # Reference to calculate the processing time

        # If it is the first execution
        if self.time == 0:
            # Prepare data required in the next times
            self._app_loads = {app: [workload] for app, workload in app_workloads.items()}
            self._icf = list(self.system.keys())[0][1] # The instance class family used
            self._app_ccs = self._get_app_ccs(self._aggs)
            # Node creation time and container allocation time are assumed to be zero for the initial allocation
            self.allocation = self._initial_allocation(app_workloads)
            self.time += 1
            return True, True, current_time() - initial_time
        else:
            # Update the application loads
            for app in app_workloads:
                self._app_loads[app].append(app_workloads[app])
            # Dispatch events until the current time
            self._timedops.dispatch_events(self.time)
            # Clear from the allocation nodes in the removed state
            self._clear_removed_nodes()
            # At the beginning of each time period
            if self.time % self.time_period == 0:
                self._set_desired_replicas()
                if self._enable_container_removal:
                    self._remove_excess_of_replicas()
                unallocatable_replicas = self._allocate_deficit_replicas()
                self._new_nodes_required = self._create_required_nodes(unallocatable_replicas)
                if self._enable_node_removal:
                    self._remove_low_utilization_nodes()
                # Reset loads
                for app in self._app_loads:
                    self._app_loads[app].clear()

            # At any other time try to allocate replicas of applications with deficit if
            # new nodes are available or the allocation has changed
            if self._timedops.new_nodes_ready or self._timedops.perf_changed:
                self._allocate_deficit_replicas()

            self.time += 1
            return (self._timedops.perf_changed, self._timedops.node_billing_changed,
                    current_time() - initial_time)


class HVReactiveAutoscaler(Autoscaler):
    """
    Horizontal/vertical and reactive autoscaler for containers and nodes.
    """

    def __init__(self, time_period: int = 60, desired_cpu_utilization: float = 0.6,
                 timing_args: TimedOps.TimingArgs = None, algorithm: AutoscalerTypes = AutoscalerTypes.FCMA,
                 transition_time_budget: int = 0):
        """
        Constructor for the horizontal/vertical reactive autoscaler.
        :param time_period: Time period to evaluate a new autoscaling.
        :param desired_cpu_utilization: Desired CPU utilization for the application containers.
        :param timing_args: Timings for creation/removal of nodes and containers.
        :param algorithm: Allocation algorithm.
        :param transition_time_budget: Approximate transition time budget. The actual transition time can be higher.
        """
        super().__init__(timing_args)
        self.time_period = time_period
        self.desired_cpu_utilization = desired_cpu_utilization
        self._app_loads = {}  # Application workloads in a time period
        self._fcma_speed_level = 1
        if algorithm == AutoscalerTypes.FCMA2:
            self._fcma_speed_level = 2
        elif algorithm == AutoscalerTypes.FCMA3:
            self._fcma_speed_level = 3
        self.transition = None
        self.transition_time_budget = transition_time_budget
        self._new_allocation = None
        self._timedops = TimedOps(self.timing_args)

    def run(self, app_workloads: dict[App, RequestsPerTime]) -> tuple[bool, bool, float]:
        """
        Simulate for 1 second the horizontal/vertical and reactive autoscaling of containers and nodes.
        :param app_workloads: Workload for all the applications at the current time.
        :return: A tuple with billing changes, performance changes and processing time.
        """

        initial_time = current_time() # Reference to calculate the processing time

        if self.time == 0:
            # Start with average loads equal to the first loads. Loads are incremented to obtain
            # the desired utilization
            incremented_workloads = {app: app_workloads[app] / self.desired_cpu_utilization for app in app_workloads}
            # Initialize the transition
            self.transition = Transition(self.timing_args, self.system, time_limit=self.transition_time_budget)
            # Calculate the first allocation
            fcma_problem = Fcma(self.system, workloads=incremented_workloads)
            solving_pars = SolvingPars(speed_level=self._fcma_speed_level)
            fcma_allocation = fcma_problem.solve(solving_pars).allocation
            self._new_allocation = [node for family, nodes in fcma_allocation.items() for node in nodes]
            self._app_loads = {}  # Application workloads in a time period
            self.allocation = self._new_allocation
            self.log_allocation_summary()
            for node in self.allocation + self._new_allocation:
                NodeStates.set_state(node, NodeStates.READY)
            self._app_loads = {app: [workload] for app, workload in app_workloads.items()}
            self.time += 1
            return True, True, current_time() - initial_time

        # Update the application loads
        for app in app_workloads:
            self._app_loads[app].append(app_workloads[app])

        # A new allocation is calculated every time period if there are no pending transitions
        if self.time % self.time_period == 0:
            # Average workloads are artificially incremented to obtain the desired CPU utilization
            incremented_workloads = {}
            for app in self._app_loads:
                incremented_workloads[app] = \
                    sum(self._app_loads[app][-self.time_period:]) / self.time_period / self.desired_cpu_utilization
            # If any transition is completed
            if self._timedops.is_event_list_empty():
                self.allocation = self._new_allocation
                # Use FCMA algorithm to calculate the new allocation
                fcma_problem = Fcma(self.system, workloads=incremented_workloads)
                solving_pars = SolvingPars(speed_level=self._fcma_speed_level)
                fcma_allocation = fcma_problem.solve(solving_pars).allocation
                self._new_allocation = [node for family, nodes in fcma_allocation.items() for node in nodes]
                for node in self.allocation + self._new_allocation:
                    NodeStates.set_state(node, NodeStates.READY)
                # Calculate the transition between the previous allocation and the new one
                commands, transition_time = self.transition.calculate_sync(self.allocation, self._new_allocation)
                self.log(f"Transition: {transition_time} seconds")
                self.log(f"- From {[str(node) for node in self.allocation]}")
                self.log(f"- To   {[str(node) for node in self._new_allocation]}")
                self.log(f"- Temporal nodes {[str(node) for node in commands[0].create_nodes if node.id < 0]}")
                # Generate transition events from the current time
                self._transition_execute_sync(commands)

            # Reset loads
            for app in self._app_loads:
                self._app_loads[app].clear()

        # Dispatch events until the current time
        self._timedops.dispatch_events(self.time)

        # Complete the removal of nodes
        for node in self.allocation:
            if NodeStates.get_state(node) == NodeStates.REMOVED:
                self.allocation.remove(node)

        self.time += 1
        return self._timedops.perf_changed, self._timedops.node_billing_changed, current_time() - initial_time


class HVPredictiveAutoscaler(Autoscaler):
    """
    Horizontal/vertival and predictive autoscaler for containers and nodes.
    """

    def __init__(self, prediction_window: int = 3600, prediction_percentile: int = 95,
                 timing_args: TimedOps.TimingArgs = None,
                 algorithm: AutoscalerTypes = AutoscalerTypes.FCMA,
                 transition_time_budget: int = 0):
        """
        Constructor for the horizontal/vertical reactive autoscaler.
        :param prediction_percentile: Load prediction percentile.
        :param, prediction_window: Prediction window in seconds.
        :param timing_args: Timings for creation/removal of nodes and containers.
        """
        super().__init__(timing_args)
        self.prediction_percentile = prediction_percentile
        self.prediction_window = prediction_window
        self._fcma_speed_level = 1
        if algorithm == AutoscalerTypes.FCMA2:
            self._fcma_speed_level = 2
        elif algorithm == AutoscalerTypes.FCMA3:
            self._fcma_speed_level = 3
        self.predicted_workloads = None
        self.transition = None
        self.transition_time_budget = transition_time_budget
        self.new_allocation = None
        self._timedops = TimedOps(self.timing_args)
        self._app_load = None
        self._waiting_for_transition_completion = False

    def run(self, dummy) -> tuple[bool, bool, float]:
        """
        Simulate for 1 second the horizontal/vertical and predictive autoscaling of containers and nodes.
        :param dummy: This parameter is ignored.
        :return: A tuple with billing changes, performance changes and processing time.
        """

        initial_time = current_time() # Reference to calculate the processing time

        if self.time == 0:
            self._app_load = self.predicted_workloads[self.time]
            self.transition = Transition(self.timing_args, self.system, time_limit=self.transition_time_budget//2)
            # The first allocation uses the load for the first prediction window
            fcma_problem = Fcma(self.system, workloads=self._app_load)
            solving_pars = SolvingPars(speed_level=self._fcma_speed_level)
            fcma_allocation = fcma_problem.solve(solving_pars).allocation
            self.new_allocation = [node for family, nodes in fcma_allocation.items() for node in nodes]
            self.allocation = self.new_allocation
            self.log_allocation_summary()
            for node in self.allocation + self.new_allocation:
                NodeStates.set_state(node, NodeStates.READY)
            self._timedops.perf_changed = True
            self._timedops.node_billing_changed = True
            self._waiting_for_transition_completion = True
            self.time += 1
            return True, True, current_time() - initial_time

        # An allocation for the next prediction window is calculated when the transition for the current window ends
        if self.time % self.prediction_window == 0:
            self._waiting_for_transition_completion = True
        next_prediction_window_time = (self.time // self.prediction_window + 1) * self.prediction_window
        if self._waiting_for_transition_completion and self._timedops.is_event_list_empty() and \
                next_prediction_window_time in self.predicted_workloads:
            self._waiting_for_transition_completion = False
            self.allocation = self.new_allocation
            new_app_load = self.predicted_workloads[next_prediction_window_time]

            # Use FCMA algorithm to calculate an intermediate allocation for the next prediction window.
            # This allocation works with the maximum loads evaluated between the current and next prediciton windows
            _, max_app_load = get_min_max_load(self._app_load, new_app_load)
            fcma_problem = Fcma(self.system, workloads=max_app_load)
            solving_pars = SolvingPars(speed_level=self._fcma_speed_level)
            fcma_allocation1 = fcma_problem.solve(solving_pars).allocation
            intermediate_allocation = [node for family, nodes in fcma_allocation1.items() for node in nodes]

            # Use FCMA to calculate the new allocation
            fcma_problem = Fcma(self.system, workloads=new_app_load)
            fcma_allocation2 = fcma_problem.solve(solving_pars).allocation
            self.new_allocation = [node for family, nodes in fcma_allocation2.items() for node in nodes]

            # Prepare application's load for the next prediction window
            self._app_load = new_app_load

            # Calculate the transition from the current allocation to the intermediate allocation
            for node in self.allocation + intermediate_allocation:
                NodeStates.set_state(node, NodeStates.READY)
            commands1, _ = self.transition.calculate_sync(self.allocation, intermediate_allocation)

            # Calculate the transition from the intermediate allocation to the new allocation.
            # The second transition uses all the nodes, even those removed in the first transition
            removed_nodes = [node for comand in commands1 for node in comand.remove_nodes]
            removed_nodes_backup = {
                node: (node.free_cores, node.free_mem, node.cgs, node.history)
                for node in removed_nodes
            }
            intermediate_allocation.extend([node.clear() for node in removed_nodes])
            commands2, _ = self.transition.calculate_sync(intermediate_allocation, self.new_allocation)
            for node in removed_nodes:
                node.free_cores, node.free_mem, node.cgs, node.history = removed_nodes_backup[node]

            # Common node removals in both transitions must be handled
            Autoscaler._handle_node_removals(commands1, commands2)

            # Calculate the times for the two transitions
            transition1_time = Transition.get_transition_time(commands1, self.timing_args)
            transition2_time = Transition.get_transition_time(commands2, self.timing_args)

            self.log(f"Transition at {next_prediction_window_time - transition1_time}:"
                     f"{transition1_time + transition2_time} seconds")
            self.log(f"Predicted loads for {self.prediction_percentile:.1f} % percentile:")
            for app, load in new_app_load.items():
                self.log(f"- {app.name} -> {load.to('req/s').magnitude:.2f} req/s")
            self.log(f"- From {[str(node) for node in self.allocation]}")
            self.log(f"- To   {[str(node) for node in self.new_allocation]}")
            temporal_nodes = []
            for command1 in commands1:
                for node in command1.create_nodes:
                    if node not in self.new_allocation:
                        temporal_nodes.append(str(node))
                for command2 in commands2:
                    for node in command2.create_nodes:
                        if node not in self.new_allocation:
                            temporal_nodes.append(str(node))
            self.log(f"- Temporal nodes {temporal_nodes}")

            # Generate transition events
            self._transition_execute_sync(commands1, next_prediction_window_time - transition1_time)
            self._transition_execute_sync(commands2, next_prediction_window_time)

        # Dispatch events until the current time
        self._timedops.dispatch_events(self.time)

        # Complete the removal of nodes
        for node in self.allocation:
            if NodeStates.get_state(node) == NodeStates.REMOVED:
                self.allocation.remove(node)

        self.time += 1
        return self._timedops.perf_changed, self._timedops.node_billing_changed, current_time() - initial_time


class HReactiveHVReactiveAutoscaler(HReactiveAutoscaler):
    """
    Mixed Horizontal and horizontal/vertical reactive autoscaler for containers and nodes.
    """

    def __init__(self, h_time_period: int = 60, desired_cpu_utilization: float = 0.6,
                 h_node_utilization_threshold: float = 0.5, timing_args: TimedOps.TimingArgs = None,
                 hv_algorithm: AutoscalerTypes = AutoscalerTypes.FCMA,
                 hv_time_period: int = 300,
                 hv_transition_time_budget: int = 0
                 ):
        """
        Constructor for the mixed reactive horizontal and reactive horizontal/vertical autoscaler.
        :param h_time_period: Time period to evaluate a new autoscaling.
        :param desired_cpu_utilization: Desired CPU utilization for the application containers.
        :param h_node_utilization_threshold: Below this threshold, a node is tried to be removed.
        :param timing_args: Timings for creation/removal of nodes and containers.
        :param hv_algorithm: Allocation algorithm.
        :param hv_time_period: Time period for H/V autoscaling.
        :param hv_transition_time_budget: Approximate transition time budget. The actual transition time can be higher.
        """
        super().__init__(h_time_period, desired_cpu_utilization, h_node_utilization_threshold, None, timing_args)
        self.h_time_period = h_time_period
        self.hv_time_period = hv_time_period
        self.desired_cpu_utilization = desired_cpu_utilization
        self._hv_app_loads = {} # Application workloads in a time period for the HV autoscaler
        self._aggs = None # H autoscaler works with all the aggregation levels
        self._fcma_speed_level = 1
        if hv_algorithm == AutoscalerTypes.FCMA2:
            self._fcma_speed_level = 2
        elif hv_algorithm == AutoscalerTypes.FCMA3:
            self._fcma_speed_level = 3
        self.transition = None
        self.transition_time_budget = hv_transition_time_budget
        self._new_allocation = None
        self._hv_timedops = TimedOps(self.timing_args)
        self._next_hv_autoscaling_time = self.hv_time_period

    def enable_disable_near_h_operations(self):
        """
        Disable horizontal node/container creation/removal operations when they are too close
        to the next Horizontal/vertical period.
        """
        if self.time + self.timing_args.node_creation_time >= self._next_hv_autoscaling_time:
            self._enable_node_creation = False
        else:
            self._enable_node_creation = True
        if self.time + self.timing_args.node_removal_time >= self._next_hv_autoscaling_time:
            self._enable_node_removal = False
        else:
            self._enable_node_removal = True
        if self.time + self.timing_args.container_creation_time >= self._next_hv_autoscaling_time:
            self._enable_container_allocation = False
        else:
            self._enable_container_allocation = True
        if self.time + self.timing_args.container_removal_time >= self._next_hv_autoscaling_time:
            self._enable_container_removal = False
        else:
            self._enable_container_removal = True

    def run(self, app_workloads: dict[App, RequestsPerTime]) -> tuple[bool, bool, float]:
        """
        Simulate for 1 second the horizontal/vertical and reactive autoscaling of containers and nodes.
        :param app_workloads: Workload for all the applications at the current time.
        :return: A tuple with billing changes, performance changes and processing time.
        """
        initial_time = current_time() # Reference to calculate the processing time

        # If it is the first execution
        if self.time == 0:
            # Initialize the HV application load in the last period
            self._hv_app_loads = {app: [] for app in app_workloads}
            # Initialize the transition
            self.transition = Transition(self.timing_args, self.system, time_limit=self.transition_time_budget)
            super().run(app_workloads)
            return True, True, current_time() - initial_time

        # Update the application loads
        for app in app_workloads:
            self._hv_app_loads[app].append(app_workloads[app])

        # If HV autoscaling is not in progress
        if self.time < self._next_hv_autoscaling_time and self._hv_timedops.is_event_list_empty():
            self.enable_disable_near_h_operations()
            # Perform H autoscaling. Note that calling run() method increments the value of self.time
            node_billing_changed, perf_changed, _ = super().run(app_workloads)
            # If the H autoscaler requires a node creation too close to the next HV autoscaling transition
            node_creation_end = self.time + self.timing_args.node_creation_time
            if self._new_nodes_required and node_creation_end >= self._next_hv_autoscaling_time:
                self._next_hv_autoscaling_time = self.time
            return node_billing_changed, perf_changed, current_time() - initial_time

        # Update application loads for the horizontal autoscaler. Note that run() method does not execute
        # in the code that follows
        for app in app_workloads:
            self._app_loads[app].append(app_workloads[app])

        # After this point, we want to perform a new HV autoscaling, but we need to process pending
        # events of the H autoscaler before performing an HV autoscaling
        if not self._timedops.is_event_list_empty():
            self.time += 1
            self._timedops.dispatch_events(self.time)
            return self._timedops.node_billing_changed, self._timedops.perf_changed, current_time() - initial_time

        elif self._hv_timedops.is_event_list_empty():
            # Perform an HV autoscaling
            # The load is calculated averaging the last time_period samples
            incremented_workloads = {
                app: sum(self._hv_app_loads[app][-self.time_period:]) / self.time_period / self.desired_cpu_utilization
                for app in app_workloads
            }
            for app in self._hv_app_loads:
                self._hv_app_loads[app].clear()
            # Use FCMA algorithm to calculate the new allocation
            fcma_problem = Fcma(self.system, workloads=incremented_workloads)
            solving_pars = SolvingPars(speed_level=self._fcma_speed_level)
            fcma_allocation = fcma_problem.solve(solving_pars).allocation
            new_allocation = [node for family, nodes in fcma_allocation.items() for node in nodes]
            # Calculate the transition between the previous allocation and the new one
            for node in self.allocation + new_allocation:
                NodeStates.set_state(node, NodeStates.READY)
            commands, transition_time = self.transition.calculate_sync(self.allocation, new_allocation)
            self.log(f"Transition: {transition_time} seconds")
            self.log(f"- From {[str(node) for node in self.allocation]}")
            self.log(f"- To   {[str(node) for node in new_allocation]}")
            self.log(f"- Temporal nodes {[str(node) for node in commands[0].create_nodes if node.id < 0]}")
            # Generate transition events from the current time and move the events to the HV event list
            self._transition_execute_sync(commands, timedops=self._hv_timedops)
            # Calculate the next HV autoscaling time
            self._next_hv_autoscaling_time = int(ceil(self.time / self.hv_time_period + 1)) * self.hv_time_period

        # Dispatch events in the HV event list
        self._hv_timedops.dispatch_events(self.time)

        # Complete the removal of nodes
        for node in self.allocation:
            if NodeStates.get_state(node) == NodeStates.REMOVED:
                self.allocation.remove(node)
        self.time += 1


        return self._hv_timedops.perf_changed, self._hv_timedops.node_billing_changed, current_time() - initial_time


class HReactiveHVPredictiveAutoscaler(HReactiveAutoscaler):
    """
    Mixed horizontal reactive and horizontal/vertical predictive autoscaler for containers and nodes.
    """

    def __init__(self, h_time_period: int = 60, h_desired_cpu_utilization: float = 0.6,
                 h_node_utilization_threshold: float = 0.5, timing_args: TimedOps.TimingArgs = None,
                 hv_algorithm: AutoscalerTypes = AutoscalerTypes.FCMA,
                 hv_prediction_window: int = 600, hv_prediction_percentile: float = 0.95,
                 hv_transition_time_budget: int = 0
                 ):
        """
        Constructor for the mixed reactive horizontal and predictive horizontal/vertical autoscaler.
        :param h_time_period: Time period to evaluate a new autoscaling.
        :param h_desired_cpu_utilization: Desired CPU utilization for the application containers.
        :param h_node_utilization_threshold: Below this threshold, a node is tried to be removed.
        :param timing_args: Timings for creation/removal of nodes and containers.
        :param hv_algorithm: Allocation algorithm.
        :param hv_prediction_window: Prediction window for the H/V autoscaler.
        :àram hv_prediction_percentile: Prediction percentile for the H/V autoscaler.
        :param hv_transition_time_budget: Approximate transition time budget. The actual transition time can be higher.
        """
        super().__init__(h_time_period, h_desired_cpu_utilization,
                         h_node_utilization_threshold, None, timing_args)
        self.time_period = h_time_period
        self.prediction_window = hv_prediction_window
        self.prediction_percentile = hv_prediction_percentile
        self.predicted_workloads = None
        self._fcma_speed_level = 1
        if hv_algorithm == AutoscalerTypes.FCMA2:
            self._fcma_speed_level = 2
        elif hv_algorithm == AutoscalerTypes.FCMA3:
            self._fcma_speed_level = 3
        self.transition = None
        self.transition_time_budget = hv_transition_time_budget
        self._hv_timedops = TimedOps(self.timing_args)
        self._next_prediction_window_time = hv_prediction_window
        self._hv_app_loads = {} # Application workloads in a time period for the HV autoscaler

    def enable_disable_close_h_operations(self):
        """
        Disable horizontal node/container creation/removal operations when they are too close
        to the next horizontal/vertical prediction window.
        """
        transition_time = self.transition.get_worst_case_transition_time()
        if self.time + self.timing_args.node_creation_time >= self._next_prediction_window_time - transition_time:
            self._enable_node_creation = False
        else:
            self._enable_node_creation = True
        if self.time + self.timing_args.node_removal_time >= self._next_prediction_window_time - transition_time:
            self._enable_node_removal = False
        else:
            self._enable_node_removal = True
        if self.time + self.timing_args.container_creation_time >= self._next_prediction_window_time - transition_time:
            self._enable_container_allocation = False
        else:
            self._enable_container_allocation = True
        if self.time + self.timing_args.container_removal_time >= self._next_prediction_window_time - transition_time:
            self._enable_container_removal = False
        else:
            self._enable_container_removal = True

    def run(self, app_workloads: dict[App, RequestsPerTime]) -> tuple[bool, bool, float]:
        """
        Simulate for 1 second the mixed reactive horizontal and predictive horizontal/vertical autoscaler.
        :param app_workloads: Workload for all the applications at the current time.
        :return: A tuple with billing changes, performance changes and processing time.
        """

        initial_time = current_time() # Reference to calculate the processing time

        # If it is the first execution
        if self.time == 0:
            # Initialize the HV application load in the last period
            self._hv_app_loads = {app: [] for app in app_workloads}
            # Initialize the transition
            self.transition = Transition(self.timing_args, self.system, time_limit=self.transition_time_budget)
            super().run(app_workloads)
            return True, True, current_time() - initial_time

        # Update the HV application loads
        for app in app_workloads:
            self._hv_app_loads[app].append(app_workloads[app])

        # If HV autoscaling is not in progress, and there is enough time until the first transition
        latest_h_time = self._next_prediction_window_time - self.transition.get_worst_case_transition_time()
        if self.time <  latest_h_time and self._hv_timedops.is_event_list_empty():
            self.enable_disable_close_h_operations()
            # Perform H autoscaling. Note that calling run() method increments the value of self.time
            node_billing_changed, perf_changed, _ = super().run(app_workloads)
            return node_billing_changed, perf_changed, current_time() - initial_time

        # Horizontal autoscaler does not run in the code that follows, so application loads for the
        # horizontal autoscaler need to be updated
        for app in app_workloads:
            self._app_loads[app].append(app_workloads[app])

        # Perform an HV transition after completing any event of the horizontal autoscaler
        if self._hv_timedops.is_event_list_empty() and self._next_prediction_window_time in self.predicted_workloads:
            # Use FCMA algorithm to calculate an intermediate allocation for the next prediction window.
            # This allocation works with the maximum loads evaluated between the current and next prediciton windows
            hv_app_load = {
                app: sum(self._hv_app_loads[app][-self.time_period:]) / self.time_period
                for app in app_workloads
            }
            for app in self._hv_app_loads:
                self._hv_app_loads[app].clear()
            hv_new_app_load = self.predicted_workloads[self._next_prediction_window_time]
            _, max_app_load = get_min_max_load(hv_app_load, hv_new_app_load)
            fcma_problem = Fcma(self.system, workloads=max_app_load)
            solving_pars = SolvingPars(speed_level=self._fcma_speed_level)
            fcma_allocation1 = fcma_problem.solve(solving_pars).allocation
            intermediate_allocation = [node for family, nodes in fcma_allocation1.items() for node in nodes]

            # Use FCMA to calculate the new allocation
            fcma_problem = Fcma(self.system, workloads=hv_new_app_load)
            fcma_allocation2 = fcma_problem.solve(solving_pars).allocation
            new_allocation = [node for family, nodes in fcma_allocation2.items() for node in nodes]

            # Calculate the transition from the current allocation to the intermediate allocation
            for node in self.allocation + intermediate_allocation:
                NodeStates.set_state(node, NodeStates.READY)
            commands1, _ = self.transition.calculate_sync(self.allocation, intermediate_allocation)

            # Calculate the transition from the intermediate allocation to the new allocation.
            # The second transition uses all the nodes, even those removed in the first transition
            removed_nodes = [node for comand in commands1 for node in comand.remove_nodes]
            removed_nodes_backup = {
                node: (node.free_cores, node.free_mem, node.cgs, node.history)
                for node in removed_nodes
            }
            intermediate_allocation.extend([node.clear() for node in removed_nodes])
            commands2, _ = self.transition.calculate_sync(intermediate_allocation, new_allocation)
            for node in removed_nodes:
                node.free_cores, node.free_mem, node.cgs, node.history = removed_nodes_backup[node]

            # Common node removals in both transitions must be handled
            Autoscaler._handle_node_removals(commands1, commands2)

            # Calculate the times for the two transitions
            transition1_time = Transition.get_transition_time(commands1, self.timing_args)
            transition2_time = Transition.get_transition_time(commands2, self.timing_args)

            self.log(f"Transition at {self._next_prediction_window_time - transition1_time}:"
                     f"{transition1_time + transition2_time} seconds")
            self.log(f"Predicted loads for {self.prediction_percentile:.1f} % percentile:")
            for app, load in hv_new_app_load.items():
                self.log(f"- {app.name} -> {load.to('req/s').magnitude:.2f} req/s")
            self.log(f"- From {[str(node) for node in self.allocation]}")
            self.log(f"- To   {[str(node) for node in new_allocation]}")
            temporal_nodes = []
            for command1 in commands1:
                for node in command1.create_nodes:
                    if node not in new_allocation:
                        temporal_nodes.append(str(node))
                for command2 in commands2:
                    for node in command2.create_nodes:
                        if node not in new_allocation:
                            temporal_nodes.append(str(node))
            self.log(f"- Temporal nodes {temporal_nodes}")

            # Generate transition events
            self._transition_execute_sync(commands1, self._next_prediction_window_time - transition1_time,
                                          timedops=self._hv_timedops)
            self._transition_execute_sync(commands2, self._next_prediction_window_time, timedops=self._hv_timedops)

        # Dispatch events in the HV event list
        self._hv_timedops.dispatch_events(self.time)
        if self._hv_timedops.is_event_list_empty():
            # Calculate the next HV autoscaling time
            self._next_prediction_window_time += self.prediction_window

        # Complete the removal of nodes
        for node in self.allocation:
            if NodeStates.get_state(node) == NodeStates.REMOVED:
                self.allocation.remove(node)

        self.time += 1

        return self._hv_timedops.perf_changed, self._hv_timedops.node_billing_changed, current_time() - initial_time

