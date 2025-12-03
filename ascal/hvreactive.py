"""
Implement the horizontal/vertical reactive autoscaler
"""

from time import time as current_time
from fcma import App, RequestsPerTime
from ascal.timedops import TimedOps
from ascal.nodestates import NodeStates
from ascal.autoscalers import AllocationSolver, Autoscaler, AutoscalerStatistics
from ascal.transition import Transition


class HVReactiveAutoscaler(Autoscaler):
    """
    Horizontal/vertical and reactive autoscaler for containers and nodes.
    """

    def __init__(self, time_period: int = 60, desired_cpu_utilization: float = 0.6,
                 timing_args: TimedOps.TimingArgs = None, algorithm: AllocationSolver = AllocationSolver.FCMA,
                 transition_time_budget: int = 0, hot_node_scale_up: bool = False,
                 hot_replicas_scale: bool = False ):
        """
        Constructor for the horizontal/vertical reactive autoscaler.
        :param time_period: Time period to evaluate a new autoscaling.
        :param desired_cpu_utilization: Desired CPU utilization for the application containers.
        :param timing_args: Timings for creation/removal of nodes and containers.
        :param algorithm: Allocation algorithm.
        :param transition_time_budget: Approximate transition time budget. The actual transition time can be higher.
        :param hot_node_scale_up: Set to enable hot node scaling-up.
        :param hot_replicas_scale: Set to enable hot scaling of replicas' computational parameters.
        """
        super().__init__(timing_args)
        self.time_period = time_period
        self.desired_cpu_utilization = desired_cpu_utilization
        self._app_loads = {}  # Application workloads in a time period
        self._allocation_solver = algorithm
        self.transition = None
        self.transition_time_budget = transition_time_budget
        self.hot_node_scale_up = hot_node_scale_up
        self.hot_replicas_scale = hot_replicas_scale
        self._new_allocation = None
        self._timedops = TimedOps(self.timing_args)

    def run(self, app_workloads: dict[App, RequestsPerTime]) -> AutoscalerStatistics:
        """
        Simulate one second of horizontal/vertical reactive autoscaling of containers and nodes.
        :param app_workloads: Workload for all the applications at the current time.
        :return: Simulation statistics.
        """

        initial_time = current_time() # Reference to calculate the processing time

        node_recycling_level = Autoscaler.INVALID_RECYCLING
        container_recycling_level = Autoscaler.INVALID_RECYCLING

        if self.time == 0:
            # Start with average loads equal to the first loads. Loads are incremented to obtain
            # the desired utilization
            incremented_workloads = {app: app_workloads[app] / self.desired_cpu_utilization for app in app_workloads}
            # At least one application replica
            Autoscaler._set_delta_loads_if_zero(incremented_workloads)
            # Initialize the transition
            self.transition = Transition(self.timing_args, self.system, time_limit=self.transition_time_budget,
                                         hot_node_scale_up=self.hot_node_scale_up,
                                         hot_replicas_scale=self.hot_replicas_scale)
            # Calculate the first allocation
            self._new_allocation = self._solve_allocation(incremented_workloads, self._allocation_solver)
            self._app_loads = {}  # Application workloads in a time period
            self.allocation = self._new_allocation
            self.log_allocation_summary()
            for node in set(self.allocation + self._new_allocation):
                NodeStates.set_state(node, NodeStates.READY)
            self._app_loads = {app: [workload] for app, workload in app_workloads.items()}
            self.time += 1
            statistics = AutoscalerStatistics(True, True, 0, current_time() - initial_time, -1, -1)
            return statistics

        # Update the application loads
        for app in app_workloads:
            self._app_loads[app].append(app_workloads[app])

        # Time required to perform the transition
        transition_time = 0

        # A new allocation is calculated every time period if there are no pending transitions
        if self.time % self.time_period == 0:
            # Average workloads are artificially incremented to obtain the desired CPU utilization
            incremented_workloads = {}
            for app in self._app_loads:
                incremented_workloads[app] = \
                    sum(self._app_loads[app][-self.time_period:]) / self.time_period / self.desired_cpu_utilization
            # At least one application replica
            Autoscaler._set_delta_loads_if_zero(incremented_workloads)
            # If any transition is completed
            if self._timedops.is_event_list_empty():
                self.allocation = self._new_allocation
                # Use FCMA algorithm to calculate the new allocation
                self._new_allocation = self._solve_allocation(incremented_workloads, self._allocation_solver)
                # Calculate the transition between the previous allocation and the new one
                transition_time_start = current_time()
                for node in self.allocation + self._new_allocation:
                    NodeStates.set_state(node, NodeStates.READY)
                commands, transition_time = self.transition.calculate_sync(self.allocation, self._new_allocation)
                transition_time = current_time() - transition_time_start
                self.log(f"Transition: {transition_time} seconds")
                self.log(f"- From {[str(node) for node in self.allocation]}")
                self.log(f"- To   {[str(node) for node in self._new_allocation]}")
                if len(commands) > 0:
                    self.log(f"- Temporal nodes {[str(node) for node in commands[0].create_nodes if node.id < 0]}")
                    # Generate transition events from the current time
                    self._transition_execute_sync(commands)
                # Get recycling levels
                node_recycling_level, container_recycling_level = self.transition.get_recycling_levels()

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

        statistics = AutoscalerStatistics(self._timedops.perf_changed, self._timedops.node_billing_changed,
                                          transition_time, current_time() - initial_time, node_recycling_level,
                                          container_recycling_level)
        return statistics

