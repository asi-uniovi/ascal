"""
Implement the horizontal/vertical predictive autoscaler
"""

from time import time as current_time
from fcma import Fcma, SolvingPars
from ascal.timedops import TimedOps
from ascal.nodestates import NodeStates
from ascal.autoscalers import AutoscalerTypes, Autoscaler, AutoscalerStatistics
from ascal.transition import Transition
from ascal.helper import get_min_max_load


class HVPredictiveAutoscaler(Autoscaler):
    """
    Horizontal/vertical and predictive autoscaler for containers and nodes.
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
        :return: Simulation statistics.
        """

        initial_time = current_time() # Reference to calculate the processing time

        node_recycling_level1 = Autoscaler.INVALID_RECYCLING
        node_recycling_level2 = Autoscaler.INVALID_RECYCLING
        container_recycling_level1 = Autoscaler.INVALID_RECYCLING
        container_recycling_level2 = Autoscaler.INVALID_RECYCLING

        if self.time == 0:
            self._app_load = self.predicted_workloads[self.time]
            # At least one application replica
            Autoscaler._set_delta_loads_if_zero(self._app_load)
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
            statistics = AutoscalerStatistics(True, True, current_time() - initial_time, 0, 0)
            return statistics

        # An allocation for the next prediction window is calculated when the transition for the current window ends
        if self.time % self.prediction_window == 0:
            self._waiting_for_transition_completion = True
        next_prediction_window_time = (self.time // self.prediction_window + 1) * self.prediction_window
        if self._waiting_for_transition_completion and self._timedops.is_event_list_empty() and \
                next_prediction_window_time in self.predicted_workloads:
            self._waiting_for_transition_completion = False
            self.allocation = self.new_allocation
            new_app_load = self.predicted_workloads[next_prediction_window_time]

            # At least one application replica
            Autoscaler._set_delta_loads_if_zero(self._app_load)
            Autoscaler._set_delta_loads_if_zero(new_app_load)

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
            for node in self.allocation:
                NodeStates.set_state(node, NodeStates.READY)
            commands1, _ = self.transition.calculate_sync(self.allocation, intermediate_allocation)

            # Recycling levels coming from the first transition
            node_recycling_level1, container_recycling_level1 = self.transition.get_recycling_levels()

            # Get a dictionary with the initial node corresponding to each recycled node.
            recycled_node_pairs1 = self.transition.get_recycled_node_pairs()
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
            intermediate_allocation.extend([node.clear() for node in removed_nodes])
            commands2, _ = self.transition.calculate_sync(intermediate_allocation, self.new_allocation)
            for node in removed_nodes:
                node.free_cores, node.free_mem, node.cgs, node.history = removed_nodes_backup[node]

            # Commands of the second transition work with the intermediate nodes, but need to work
            # with the same nodes as the first transition
            commands2 = [command2.replace_nodes(inverse_recycled_node_pairs1) for command2 in commands2]

            # Recycling levels coming from the second transition
            node_recycling_level2, container_recycling_level2 = self.transition.get_recycling_levels()

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
        statistics = AutoscalerStatistics(self._timedops.perf_changed, self._timedops.node_billing_changed,
                                          current_time() - initial_time,
                                          min(node_recycling_level1, node_recycling_level2),
                                          min(container_recycling_level1, container_recycling_level2))
        return statistics



