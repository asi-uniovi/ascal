"""
Implement a mixed horizontal and horizontal/vertical predictive autoscaler
"""

from time import time as current_time
from fcma import App, RequestsPerTime
from ascal.timedops import TimedOps
from ascal.nodestates import NodeStates
from ascal.autoscalers import AllocationSolver, Autoscaler, AutoscalerStatistics
from ascal.hreactive import HReactiveAutoscaler
from ascal.recycling import Recycling
from ascal.transition import TransitionAlgorithm, TransitionRBT
from ascal.helper import get_min_max_load


class HReactiveHVPredictiveAutoscaler(HReactiveAutoscaler):
    """
    Mixed horizontal reactive and horizontal/vertical predictive autoscaler for containers and nodes.
    """

    def __init__(self, h_time_period: int = 60, h_desired_cpu_utilization: float = 0.6,
                 h_node_utilization_threshold: float = 0.5, 
                 h_replica_scale_down_stabilization_time: int = 300,
                 h_node_scale_down_stabilization_time: int = 600, 
                 timing_args: TimedOps.TimingArgs = None,
                 hv_algorithm: tuple[AllocationSolver, TransitionAlgorithm] = \
                    (AllocationSolver.FCMA, TransitionAlgorithm.RBT),
                 hv_prediction_window: int = 600, hv_prediction_percentile: float = 0.95,
                 hot_node_scale_up: bool = False,
                 hot_container_scale: bool = False):
        """
        Constructor for the mixed reactive horizontal and predictive horizontal/vertical autoscaler.
        :param h_time_period: Time period to evaluate a new autoscaling.
        :param h_desired_cpu_utilization: Desired CPU utilization for the application containers.
        :param h_node_utilization_threshold: Below this threshold, a node is tried to be removed.
        :param h_replica_scale_down_stabilization_time: Minimum time from a previous replica scale-up 
        to a replica scale-down.
        :param h_node_scale_down_stabilization_time: Minimum time from a previous node scale-up
        to a node scale-down.
        :param timing_args: Timings for creation/removal of nodes and containers.
        :param hv_algorithm: Allocation/transition algorithm.
        :param hv_prediction_window: Prediction window for the H/V autoscaler.
        :àram hv_prediction_percentile: Prediction percentile for the H/V autoscaler.
        :param hot_node_scale_up: If True, hot vertical scale-up of nodes is used.
        :param hot_container_scale: If True, hot vertical scaling of containers is used.
        """
        super().__init__(h_time_period, h_desired_cpu_utilization, h_node_utilization_threshold, 
                         h_replica_scale_down_stabilization_time,
                         h_node_scale_down_stabilization_time, None, timing_args)
        self.time_period = h_time_period
        self.prediction_window = hv_prediction_window
        self.prediction_percentile = hv_prediction_percentile
        self.predicted_workloads = None
        self._allocation_solver, self._transition_algorithm = hv_algorithm
        self.transition = None
        self.hot_node_scale_up = hot_node_scale_up
        self.hot_container_scale = hot_container_scale
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

    def run(self, app_workloads: dict[App, RequestsPerTime]) -> AutoscalerStatistics:
        """
        Simulate for 1 second the mixed reactive horizontal and predictive horizontal/vertical autoscaler.
        :param app_workloads: Workload for all the applications at the current time.
        :return: A tuple with billing changes, performance changes and processing time.
        """

        initial_time = current_time() # Reference to calculate the processing time

        node_recycling_level1 = Recycling.INVALID_RECYCLING
        node_recycling_level2 = Recycling.INVALID_RECYCLING
        container_recycling_level1 = Recycling.INVALID_RECYCLING
        container_recycling_level2 = Recycling.INVALID_RECYCLING

        # If it is the first execution
        if self.time == 0:
            # Initialize the HV application load in the last period
            self._hv_app_loads = {app: [] for app in app_workloads}
            # Initialize the transition
            if self._transition_algorithm == TransitionAlgorithm.BASELINE:
                ValueError("'baseline' transition is not valid for the horizontal/vertical predictive autoscaler")
            else:
                self.transition = TransitionRBT(self.timing_args, self.system, 
                                                transition_algorithm=self._transition_algorithm,
                                                hot_node_scale_up=self.hot_node_scale_up)
                
            super().run(app_workloads)
            statistics = AutoscalerStatistics(True, True, 0, current_time() - initial_time,
                                              Recycling.INVALID_RECYCLING, Recycling.INVALID_RECYCLING)
            return statistics

        # Update the HV application loads
        for app in app_workloads:
            self._hv_app_loads[app].append(app_workloads[app])

        # If HV autoscaling is not in progress, and there is enough time until the first transition
        latest_h_time = self._next_prediction_window_time - self.transition.get_worst_case_transition_time()
        if self.time <  latest_h_time and self._hv_timedops.is_event_list_empty():
            self.enable_disable_close_h_operations()
            # Perform H autoscaling. Note that calling run() method increments the value of self.time
            statistics = super().run(app_workloads)
            return statistics

        # Horizontal autoscaler does not run in the code that follows, so application loads for the
        # horizontal autoscaler need to be updated
        for app in app_workloads:
            self._app_loads[app].append(app_workloads[app])

        # Time required to perform the transition
        transition_time = 0

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
            # At least one application replica
            Autoscaler._set_delta_loads_if_zero(hv_app_load)
            Autoscaler._set_delta_loads_if_zero(hv_new_app_load)
            _, max_app_load = get_min_max_load(hv_app_load, hv_new_app_load)
            intermediate_allocation = self._solve_allocation(max_app_load, self._allocation_solver)
            new_allocation = self._solve_allocation(hv_new_app_load, self._allocation_solver)

            # We need to measure the time required to perform the transition
            transition_time_start = current_time()

            # Calculate the transition from the current allocation to the intermediate allocation
            for node in self.allocation + intermediate_allocation:
                NodeStates.set_state(node, NodeStates.READY)
            commands1, _ = self.transition.calculate_sync(self.allocation, intermediate_allocation)

            # Recycling levels coming from the first transition
            node_recycling_level1, container_recycling_level1 = self.transition.get_recycling_levels()

            # Get a dictionary with the initial node corresponding to each recycled node
            recycled_node_pairs1 = self.transition.get_recycled_node_pairs()
            inverse_recycled_node_pairs1 = {
                final_node: initial_node
                for initial_node, final_node in recycled_node_pairs1.items()
            }
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

            # Recycling levels coming from the second transition
            node_recycling_level2, container_recycling_level2 = self.transition.get_recycling_levels()

            # Commands of the second transition work with the intermediate nodes, but need to work
            # with the same nodes as the first transition
            commands2 = [command2.replace_nodes(inverse_recycled_node_pairs1) for command2 in commands2]

            # Common node removals in both transitions must be handled
            Autoscaler._handle_node_removals(commands1, commands2, new_allocation)

            # Time required to perform the transition
            transition_time = current_time() - transition_time_start

            # Calculate the times for the two transitions
            transition1_time = self.transition.get_transition_time(commands1, self.timing_args)
            transition2_time = self.transition.get_transition_time(commands2, self.timing_args)

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

        self._hv_timedops.dispatch_events(self.time)
        if self._hv_timedops.is_event_list_empty():
            # Calculate the next HV autoscaling time
            self._next_prediction_window_time += self.prediction_window

        # Complete the removal of nodes
        for node in self.allocation:
            if NodeStates.get_state(node) == NodeStates.REMOVED:
                self.allocation.remove(node)

        self.time += 1

        statistics = AutoscalerStatistics(self._hv_timedops.perf_changed, self._hv_timedops.node_billing_changed,
                                          transition_time, current_time() - initial_time,
                                          min(node_recycling_level1, node_recycling_level2),
                                          min(container_recycling_level1, container_recycling_level2))
        return statistics
