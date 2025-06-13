"""
Implement a mixed horizontal and horizontal/vertical reactive autoscaler
"""

from math import ceil
from time import time as current_time
from fcma import App, Fcma, SolvingPars, RequestsPerTime
from ascal.timedops import TimedOps
from ascal.nodestates import NodeStates
from ascal.autoscalers import AutoscalerTypes, Autoscaler, AutoscalerStatistics
from ascal.hreactive import HReactiveAutoscaler
from ascal.transition import Transition


class HReactiveHVReactiveAutoscaler(HReactiveAutoscaler):
    """
    Mixed horizontal and horizontal/vertical reactive autoscaler for containers and nodes.
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
        :return: Simulation statistics.
        """
        initial_time = current_time() # Reference to calculate the processing time

        node_recycling_level = Autoscaler.INVALID_RECYCLING
        container_recycling_level = Autoscaler.INVALID_RECYCLING

        # If it is the first execution
        if self.time == 0:
            # Initialize the HV application load in the last period
            self._hv_app_loads = {app: [] for app in app_workloads}
            # Initialize the transition
            self.transition = Transition(self.timing_args, self.system, time_limit=self.transition_time_budget)
            super().run(app_workloads)
            statistics = AutoscalerStatistics(True, True, current_time() - initial_time,
                                              Autoscaler.INVALID_RECYCLING, Autoscaler.INVALID_RECYCLING)
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
                                              current_time() - initial_time, Autoscaler.INVALID_RECYCLING,
                                              Autoscaler.INVALID_RECYCLING)
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
        for node in self.allocation:
            if NodeStates.get_state(node) == NodeStates.REMOVED:
                self.allocation.remove(node)
        self.time += 1

        statistics = AutoscalerStatistics(self._hv_timedops.perf_changed, self._hv_timedops.node_billing_changed,
                                          current_time() - initial_time, node_recycling_level,
                                          container_recycling_level)
        return statistics

