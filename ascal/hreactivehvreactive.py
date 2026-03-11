"""
Implement a mixed horizontal and horizontal/vertical reactive autoscaler
"""

from math import ceil
from time import time as current_time
from fcma import App, RequestsPerTime
from ascal.timedops import TimedOps
from ascal.nodestates import NodeStates
from ascal.autoscalers import AllocationSolver, TransitionAlgorithm, Autoscaler, AutoscalerStatistics
from ascal.hreactive import HReactiveAutoscaler
from ascal.recycling import Recycling
from ascal.transition import TransitionBaseline, TransitionRAC


class HReactiveHVReactiveAutoscaler(HReactiveAutoscaler):
    """
    Mixed horizontal and horizontal/vertical reactive autoscaler for containers and nodes.
    """

    def __init__(self, h_time_period: int = 60, desired_cpu_utilization: float = 0.6,
                 h_node_utilization_threshold: float = 0.5, 
                 h_replica_scale_down_stabilization_time: int = 300,
                 h_node_scale_down_stabilization_time: int = 600, 
                 timing_args: TimedOps.TimingArgs = None,
                 hv_algorithm: AllocationSolver = AllocationSolver.FCMA,
                 hv_time_period: int = 300,
                 hv_transition_time_budget: int = 0, hot_node_scale_up: bool = False):
        """
        Constructor for the mixed reactive horizontal and reactive horizontal/vertical autoscaler.
        :param h_time_period: Time period to evaluate a new autoscaling.
        :param desired_cpu_utilization: Desired CPU utilization for the application containers.
        :param h_node_utilization_threshold: Below this threshold, a node is tried to be removed.
        :param h_replica_scale_down_stabilization_time: Minimum time from a previous replica scale-up 
        to a replica scale-down.
        :param h_node_scale_down_stabilization_time: Minimum time from a previous node scale-up
        to a node scale-down.
        :param timing_args: Timings for creation/removal of nodes and containers.
        :param hv_algorithm: Allocation algorithm.
        :param hv_time_period: Time period for H/V autoscaling.
        :param hv_transition_time_budget: Approximate transition time budget. The actual transition time can be higher.
        """
        super().__init__(h_time_period, desired_cpu_utilization, h_node_utilization_threshold, 
                         h_replica_scale_down_stabilization_time,
                         h_node_scale_down_stabilization_time, None, timing_args)
        self.h_time_period = h_time_period
        self.hv_time_period = hv_time_period
        self.desired_cpu_utilization = desired_cpu_utilization
        self._hv_app_loads = {} # Application workloads in a time period for the HV autoscaler
        self._aggs = None # H autoscaler works with all the aggregation levels
        self._allocation_solver, self._transition_algorithm = hv_algorithm
        self.transition = None
        self.transition_time_budget = hv_transition_time_budget
        self.hot_node_scale_up = hot_node_scale_up
        self._new_allocation = None
        self._hv_timedops = TimedOps(self.timing_args)
        self._next_hv_autoscaling_time = self.hv_time_period

    def enable_disable_near_h_operations(self):
        """
        Disable horizontal node/container creation/removal operations when they are too close
        to the next Horizontal/vertical period.
        """
        operations = [
            ('node_creation', self.timing_args.node_creation_time),
            ('node_removal', self.timing_args.node_removal_time),
            ('container_allocation', self.timing_args.container_creation_time),
            ('container_removal', self.timing_args.container_removal_time),
        ]
        for name, offset in operations:
            enabled = self.time + offset < self._next_hv_autoscaling_time
            setattr(self, f"_enable_{name}", enabled)

    def run(self, app_workloads: dict[App, RequestsPerTime]) -> AutoscalerStatistics:
        """
        Simulate for 1 second the mixed autoscaling strategy:
            - Fast horizontal reactions are applied unless a HV scaling point is near.
            - HV autoscaling is triggered periodically and overrides horizontal decisions.
        :param app_workloads: Workload for all the applications at the current time.
        :return: Simulation statistics.
        """
        initial_time = current_time() # Reference to calculate the processing time

        node_recycling_level = Recycling.INVALID_RECYCLING
        container_recycling_level = Recycling.INVALID_RECYCLING

        # Time required to calculate the transition
        transition_calc_time = 0

        # If it is the first execution
        if self.time == 0:
            # Initialize the HV application load in the last period
            self._hv_app_loads = {app: [] for app in app_workloads}
            # Initialize the transition
            if self._transition_algorithm == TransitionAlgorithm.BASELINE:
                self.transition = TransitionBaseline(self.timing_args, self.system)
            else:
                self.transition = TransitionRAC(self.timing_args, self.system, time_limit=self.transition_time_budget,
                                                hot_node_scale_up=self.hot_node_scale_up)
            super().run(app_workloads)
            statistics = AutoscalerStatistics(True, True, 0, current_time() - initial_time,
                                              Recycling.INVALID_RECYCLING, Recycling.INVALID_RECYCLING)
            return statistics

        # Update the application loads
        for app in app_workloads:
            self._hv_app_loads[app].append(app_workloads[app])

        # If HV autoscaling is not in progress
        if self.time < self._next_hv_autoscaling_time and self._hv_timedops.is_event_list_empty():
            self.enable_disable_near_h_operations()
            # Perform H autoscaling. Note that calling run() method increments the value of self.time
            statistics = super().run(app_workloads)
            # If the H autoscaler requires a node creation too close to the next HV autoscaling transition
            node_creation_end = self.time + self.timing_args.node_creation_time
            if self._new_nodes_required and node_creation_end >= self._next_hv_autoscaling_time:
                self._next_hv_autoscaling_time = self.time
            return statistics

        # Update application loads for the horizontal autoscaler. Note that run() method does not execute
        # in the code that follows
        for app in app_workloads:
            self._app_loads[app].append(app_workloads[app])

        # After this point, we want to perform a new HV autoscaling, but we need to process pending
        # events of the H autoscaler before performing an HV autoscaling
        if not self._timedops.is_event_list_empty():
            self.time += 1
            self._timedops.dispatch_events(self.time)
            statistics = AutoscalerStatistics(self._timedops.node_billing_changed, self._timedops.perf_changed,
                                              0, current_time() - initial_time, Recycling.INVALID_RECYCLING,
                                              Recycling.INVALID_RECYCLING)
            return statistics

        elif self._hv_timedops.is_event_list_empty():
            # Perform an HV autoscaling
            # The load is calculated averaging the last time_period samples
            incremented_workloads = {
                app: sum(self._hv_app_loads[app][-self.time_period:]) / self.time_period / self.desired_cpu_utilization
                for app in app_workloads
            }
            # At least one application replica
            Autoscaler._set_delta_loads_if_zero(incremented_workloads)

            for app in self._hv_app_loads:
                self._hv_app_loads[app].clear()
            # Use FCMA algorithm to calculate the new allocation
            new_allocation = self._solve_allocation(incremented_workloads, self._allocation_solver)

            # Calculate the transition between the previous allocation and the new one
            transition_time_start = current_time()
            for node in self.allocation + new_allocation:
                NodeStates.set_state(node, NodeStates.READY)
            commands, transition_time = self.transition.calculate_sync(self.allocation, new_allocation)
            transition_calc_time = current_time() - transition_time_start

            self.log(f"Transition: {transition_time} seconds")
            self.log(f"- From {[str(node) for node in self.allocation]}")
            self.log(f"- To   {[str(node) for node in new_allocation]}")
            if len(commands) > 0:
                self.log(f"- Temporal nodes {[str(node) for node in commands[0].create_nodes if node.id < 0]}")
                # Generate transition events from the current time and move the events to the HV event list
                self._transition_execute_sync(commands, timedops=self._hv_timedops)
            # Calculate the next HV autoscaling time
            self._next_hv_autoscaling_time = int(ceil(self.time / self.hv_time_period + 1)) * self.hv_time_period
            # Get recycling levels
            node_recycling_level, container_recycling_level = self.transition.get_recycling_levels()

        # Dispatch events in the HV event list
        self._hv_timedops.dispatch_events(self.time)

        # Complete the removal of nodes
        for node in list(self.allocation):
            if NodeStates.get_state(node) == NodeStates.REMOVED:
                self.allocation.remove(node)
        self.time += 1

        statistics = AutoscalerStatistics(self._hv_timedops.perf_changed, self._hv_timedops.node_billing_changed,
                                          transition_calc_time, current_time() - initial_time, node_recycling_level,
                                          container_recycling_level)
        return statistics

