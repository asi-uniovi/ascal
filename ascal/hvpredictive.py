"""
Implement the horizontal/vertical predictive autoscaler
"""

from time import time as current_time
from copy import deepcopy
from ascal.timedops import TimedOps
from ascal.nodestates import NodeStates
from ascal.autoscalers import AllocationSolver, Autoscaler, AutoscalerStatistics
from ascal.recycling import Recycling
from ascal.transition import TransitionAlgorithm, TransitionBaseline, TransitionRBT
from ascal.helper import get_min_max_load, get_allocation_cost


class HVPredictiveAutoscaler(Autoscaler):
    """
    Horizontal/vertical and predictive autoscaler for containers and nodes.
    """

    def __init__(self, prediction_window: int = 3600, prediction_percentile: int = 95,
                 timing_args: TimedOps.TimingArgs = None,
                 algorithm: tuple[AllocationSolver, TransitionAlgorithm] = \
                    (AllocationSolver.FCMA, TransitionAlgorithm.RBT),
                 tolerance_cost: float = 0.0, 
                 hot_node_scale_up: bool = False, hot_container_scale: bool = False):
        """
        Constructor for the horizontal/vertical reactive autoscaler.
        :param prediction_percentile: Load prediction percentile.
        :param prediction_window: Prediction window in seconds.
        :param timing_args: Timings for creation/removal of nodes and containers.
        :param algorithm: Allocation/transition algorithm.
        :param tolerance_cost: Minimum relative cost increment to enable the second transition to a new deployment.
        :param hot_node_scale_up: Set to enable hot node scaling-up.
        :param hot_container_scale: Set to enable hot container scaling-up and scaling-down.
        """
        super().__init__(timing_args)
        self._prediction_percentile = prediction_percentile
        self._prediction_window = prediction_window
        self._allocation_solver, self._transition_algorithm = algorithm
        self.predicted_workloads = None
        self._transition = None
        self._tolerance_cost = tolerance_cost
        self._hot_node_scale_up = hot_node_scale_up
        self._hot_container_scale = hot_container_scale
        self._new_allocation = None
        self._app_load = None
        self._waiting_to_start_transition_calculation = False

    def _initialize_allocation(self, initial_time: float) -> AutoscalerStatistics:
        """
        Perform the initial allocation based on the first prediction.
        """
        if self.time not in self.predicted_workloads:
            raise ValueError("Missing predicted workload for time 0")

        self._app_load = self.predicted_workloads[self.time]

        # A minimum load is required for each application
        Autoscaler._set_delta_loads_if_zero(self._app_load)

        # Initialize the transition
        if self._transition_algorithm == TransitionAlgorithm.BASELINE:
            self._transition = TransitionBaseline(self.timing_args, self.system)
        else:
            self._transition = TransitionRBT(self.timing_args, self.system, 
                                            transition_algorithm=self._transition_algorithm,
                                            hot_node_scale_up=self._hot_node_scale_up,
                                            hot_container_scale=self._hot_container_scale)
        # Calculate a new allocation
        self._new_allocation = self._solve_allocation(self._app_load, self._allocation_solver)
        self.allocation = self._new_allocation

        self.log_allocation_summary()

        # Set all the nodes to the ready state
        for node in set(self.allocation + self._new_allocation):
            NodeStates.set_state(node, NodeStates.READY)

        self._timedops.perf_changed = True
        self._timedops.node_billing_changed = True
        self._waiting_to_start_transition_calculation = True
        self.time += 1

        return AutoscalerStatistics(True, True, 0, current_time() - initial_time, 0, 0)

    def run(self, dummy) -> tuple[bool, bool, float]:
        """
        Simulate for 1 second the horizontal/vertical and predictive autoscaling of containers and nodes.
        :param dummy: This parameter is ignored.
        :return: Simulation statistics.
        """
        if self.time == 0:
            return self._initialize_allocation(current_time())
        
        if self._transition_algorithm == TransitionAlgorithm.BASELINE:
            return self.run_baseline()
        else:
            return self.run_rbt()

    def run_baseline(self) -> tuple[bool, bool, float]:
        """
        Simulate for 1 second the horizontal/vertical and predictive autoscaling of containers and nodes
        using the baseline transition algorithm.
        :return: Simulation statistics.
        """

        initial_time = current_time() # Reference to calculate the processing time

        node_recycling_level = Recycling.INVALID_RECYCLING
        container_recycling_level = Recycling.INVALID_RECYCLING

        # Time required to perform the transition
        transition_calc_time = 0

        # An allocation for the next prediction window is calculated when the transition for the current window ends
        if self.time % self._prediction_window == 0:
            self._waiting_to_start_transition_calculation = True
        next_prediction_window_time = (self.time // self._prediction_window + 1) * self._prediction_window
        if self._waiting_to_start_transition_calculation and self._timedops.is_event_list_empty() and \
                next_prediction_window_time in self.predicted_workloads:
            # A new transition calculation is started
            self._waiting_to_start_transition_calculation = False
            self.allocation = self._new_allocation
            new_app_load = self.predicted_workloads[next_prediction_window_time]

            # At least one application replica
            Autoscaler._set_delta_loads_if_zero(new_app_load)

            # Use FCMA algorithm to calculate the allocation for the next prediction window
            self._new_allocation = self._solve_allocation(new_app_load, self._allocation_solver)

            # We need to measure the time required to calculate the transition
            transition_calc_time_start = current_time()

            # Calculate the transition from the current allocation to the new allocation
            for node in self.allocation:
                NodeStates.set_state(node, NodeStates.READY)
            commands, transition_time = \
                self._transition.calculate_transition_plan_sync(self.allocation, self._new_allocation)

            transition_calc_time = current_time() - transition_calc_time_start
            if len(commands) > 0:
                self.log(f"- Temporal nodes {[str(node) for node in commands[0].create_nodes if node.id < 0]}")
                # Generate transition events
                node_container_creation_time = self._transition.get_creation_in_transition_time()
                transition_start_time = next_prediction_window_time - node_container_creation_time
                self._transition_execute_sync(commands, transition_start_time)
                self.log(f"Transition at {transition_start_time}. Duration: {transition_time} seconds")
                self.log(f"- From {[str(node) for node in self.allocation]}")
                self.log(f"- To   {[str(node) for node in self._new_allocation]}")
            # Get recycling levels
            node_recycling_level, container_recycling_level = self._transition.get_recycling_levels()

        # Dispatch events until the current time
        self._timedops.dispatch_events(self.time)

        # Complete the removal of nodes
        for node in self.allocation:
            if NodeStates.get_state(node) == NodeStates.REMOVED:
                self.allocation.remove(node)

        self.time += 1

        statistics = AutoscalerStatistics(self._timedops.perf_changed, self._timedops.node_billing_changed,
                                          transition_calc_time, current_time() - initial_time, node_recycling_level,
                                          container_recycling_level)
        return statistics

    def run_rbt(self) -> tuple[bool, bool, float]:
        """
        Simulate for 1 second the horizontal/vertical and predictive autoscaling of containers and nodes
        using RBT transitions.
        :return: Simulation statistics.
        """

        initial_time = current_time() # Reference to calculate the processing time

        node_recycling_level1 = Recycling.INVALID_RECYCLING
        node_recycling_level2 = Recycling.INVALID_RECYCLING
        container_recycling_level1 = Recycling.INVALID_RECYCLING
        container_recycling_level2 = Recycling.INVALID_RECYCLING

        # Time required to perform the transition
        transition_calc_time = 0

        # An allocation for the next prediction window is calculated when the transition for the current window ends
        if self.time % self._prediction_window == 0:
            self._waiting_to_start_transition_calculation = True
        next_prediction_window_time = (self.time // self._prediction_window + 1) * self._prediction_window

        if self._waiting_to_start_transition_calculation and self._timedops.is_event_list_empty() and \
                next_prediction_window_time in self.predicted_workloads:
            # A new transition calculation is started
            self._waiting_to_start_transition_calculation = False
            self.allocation = self._new_allocation
            new_app_load = self.predicted_workloads[next_prediction_window_time]

            # At least one application replica
            Autoscaler._set_delta_loads_if_zero(self._app_load)
            Autoscaler._set_delta_loads_if_zero(new_app_load)

            # Use FCMA algorithm to calculate an intermediate allocation for the next prediction window.
            # This allocation works with the maximum loads evaluated between the current and next prediction windows
            _, max_app_load = get_min_max_load(self._app_load, new_app_load)
            intermediate_allocation = self._solve_allocation(max_app_load, self._allocation_solver)
            intermediate_cost = get_allocation_cost(intermediate_allocation).to("usd/hour").magnitude

            # Use FCMA to calculate the new allocation
            self._new_allocation = self._solve_allocation(new_app_load, self._allocation_solver)
            new_cost = get_allocation_cost(self._new_allocation).to("usd/hour").magnitude

            # There is not a second transition when the cost savings are small. First transition is mandatory
            if (intermediate_cost - new_cost) / intermediate_cost <= self._tolerance_cost:
                self._new_allocation = deepcopy(intermediate_allocation)

            # Prepare application's load for the next prediction window
            self._app_load = new_app_load

            # Calculate the transition from the current allocation to the intermediate allocation
            for node in self.allocation:
                NodeStates.set_state(node, NodeStates.READY)
            commands1, _ = self._transition.calculate_transition_plan_sync(self.allocation, intermediate_allocation)

            # Recycling levels coming from the first transition
            node_recycling_level1, container_recycling_level1 = self._transition.get_recycling_levels()

            # Get a dictionary with the initial node corresponding to each recycled node.
            recycled_node_pairs1 = self._transition.get_recycled_node_pairs()
            inverse_recycled_node_pairs1 = {
                final_node: initial_node
                for initial_node, final_node in recycled_node_pairs1.items()
            }

            # Calculate the transition from the intermediate allocation to the new allocation.
            # The second transition uses all the nodes, even those removed in the first transition
            removed_nodes = [node for command in commands1 for node in command.remove_nodes]
            removed_nodes_backup = {
                node: (node.free_cores, node.free_mem, node.cgs, node.history)
                for node in removed_nodes
            }
            # Remove all the containers in the removed nodes of the first transition and add the nodes
            # to the intermediate allocation. These nodes may be useful in the second transition
            intermediate_allocation.extend([node.clear() for node in removed_nodes])
            commands2, _ = self._transition.calculate_transition_plan_sync(intermediate_allocation, self._new_allocation)
            for node in removed_nodes:
                node.free_cores, node.free_mem, node.cgs, node.history = removed_nodes_backup[node]

            # Commands of the second transition work with the intermediate nodes, but need to work
            # with the same nodes as the first transition
            commands2 = [command2.replace_nodes(inverse_recycled_node_pairs1) for command2 in commands2]

            # Recycling levels coming from the second transition
            node_recycling_level2, container_recycling_level2 = self._transition.get_recycling_levels()

            # Common node removals in both transitions must be handled
            Autoscaler._handle_node_removals(commands1, commands2, self._new_allocation)

            # Calculate the times for the two transitions
            transition1_time = self._transition.get_transition_time(commands1, self.timing_args)
            transition2_time = self._transition.get_transition_time(commands2, self.timing_args)

            # Log transition info
            self.log(f"Transition at {next_prediction_window_time - transition1_time}. "
                     f"Duration: {transition1_time + transition2_time} seconds")
            self.log(f"Predicted loads for {self._prediction_percentile:.1f} % percentile:")
            for app, load in new_app_load.items():
                self.log(f"- {app.name} -> {load.to('req/s').magnitude:.2f} req/s")
            self.log(f"- From {[str(node) for node in self.allocation]}")
            self.log(f"- To   {[str(node) for node in self._new_allocation]}")
            temporal_nodes = []
            for command1 in commands1:
                for node in command1.create_nodes:
                    if node not in self._new_allocation:
                        temporal_nodes.append(str(node))
                for command2 in commands2:
                    for node in command2.create_nodes:
                        if node not in self._new_allocation:
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

        # Generate statistics
        statistics = AutoscalerStatistics(self._timedops.perf_changed, self._timedops.node_billing_changed,
                                          transition_calc_time, current_time() - initial_time,
                                          min(node_recycling_level1, node_recycling_level2),
                                          min(container_recycling_level1, container_recycling_level2))
        return statistics



