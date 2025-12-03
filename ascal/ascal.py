"""
Main module of the ascal package. It defines class Ascal to calculate the sequence of allocations for
a given autoscaler and applications
"""

from math import ceil, floor
from copy import deepcopy
from yaml import dump as yaml_dump
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import csv
from fcma import RequestsPerTime, Allocation
from ascal.autoscalers import Autoscaler
from ascal.hvpredictive import HVPredictiveAutoscaler
from ascal.hreactivehvpredictive import HReactiveHVPredictiveAutoscaler
from ascal.nodestates import NodeStates
from ascal.ascalconfig import AscalConfig

class Ascal:
    """
    This class provides methods to simulate the autoscaling of a system under a given load trace.
    """

    def __init__(self, ascal_config: AscalConfig, log=None):
        """
        Ascal constructor.
        :param ascal_config: Configuration for the Ascal problem. It is not currently checked.
        """
        self._workload_vectors = ascal_config.workload_vectors
        self._autoscaler = ascal_config.autoscaler
        self._autoscaler.system = ascal_config.system
        self._autoscaler.apps = ascal_config.apps
        self._autoscaler.log_path = log # A string with the path of the log file
        self.time = -1 # Current simulation time
        self.last_time = len(next(iter(self._workload_vectors.values()))) - 1 # Last simulation time
        self.performance_changes: list[(int, Allocation)] = [] # Pairs time and allocation
        self.billing_changes: list[(int, Allocation)] = [] # Pairs time and allocation
        self.calc_times: dict[str, list[float]] = {"transition_times": [], "total_times": []} # Calculation times
        self.node_recycling_levels: list[float] = [] # List of node recycling levels
        self.container_recycling_levels: list[float] = [] # List of container recycling levels

    def run(self, break_point: int | None = None):
        """
        Continue simulating autoscaling until the given breakpoint.
        :param break_point: Simulate until this time. When it is None it simulates until the end.
        """
        if break_point is None:
            # The breakpoint is placed at the last time in simulation
            break_point = self.last_time
        while self.time < break_point:
            self.time += 1
            if self.time == 0 and (isinstance(self._autoscaler, HVPredictiveAutoscaler) or
                                   isinstance(self._autoscaler, HReactiveHVPredictiveAutoscaler)):
                Autoscaler.workload_predictions(self._autoscaler, self._workload_vectors)
            if self.time % 100 == 0:
                print(f'Time: {self.time} s')
            workloads = {}
            for app in self._workload_vectors:
                workload = RequestsPerTime(f"{self._workload_vectors[app][self.time] * 3600}  req/hour")
                workloads[app] = workload
            statistics = self._autoscaler.run(workloads)
            if statistics.perf_changed or statistics.billing_changed or self.time == break_point:
                allocation_copy = (self.time, deepcopy(self._autoscaler.allocation))
                if statistics.perf_changed or self.time == break_point:
                    self.performance_changes.append(allocation_copy)
                if statistics.billing_changed or self.time == break_point:
                    self.billing_changes.append(allocation_copy)
            self.calc_times["transition_times"].append(statistics.transition_time)
            self.calc_times["total_times"].append(statistics.total_time)
            self.node_recycling_levels.append(statistics.node_recycling_level)
            self.container_recycling_levels.append(statistics.container_recycling_level)
        self._autoscaler.log_allocation_summary()

    def get_workloads(self) -> dict[str, list[int]]:
        """
        Get application workloads.
        :return: For each application the workloads in req/s at every second, starting from 0 seconds.
        """
        return {str(key): value for key, value in self._workload_vectors.items()}

    def get_recycling_levels(self) -> tuple[list[float], list[float]]:
        """
        Get node and container recycling levels.
        :return: The recyclings at every second, starting from 0 seconds.
        """
        return self.node_recycling_levels, self.container_recycling_levels

    def get_performances(self) -> dict[str, list[int]]:
        """
        Get application performances.
        :return: For each application the performances in req/s at every second, starting at 0 seconds.
        """
        app_perfs = {str(app): [] for app in self._workload_vectors}

        previous_time = -1
        for current_time, current_nodes in self.performance_changes:
            # Repeat the previous allocation performances
            if current_time - previous_time > 1:
                for app_name, perf in app_perfs.items():
                    app_perfs[app_name].extend([perf[-1]] * (current_time - previous_time - 1))
            # Get application performances for the current allocation
            current_perfs = {str(app): 0 for app in self._workload_vectors}
            for node in current_nodes:
                for cg in node.cgs:
                    app = cg.cc.app
                    if app is not None:
                        current_perfs[str(app)] += cg.cc.perf.to('req/s').magnitude * cg.replicas

            # Append the current allocation performances
            for app_name in app_perfs:
                app_perfs[app_name].append(current_perfs[app_name])

            # Prepare for the next allocation change
            previous_time = current_time
        return app_perfs

    def get_cluster_cost(self) -> list[float]:
        """
        Gets the cluster cost in $/hour.
        :return: A list with the cost in $/hour at every second, starting from 0 seconds.
        """
        node_costs = []
        previous_time = -1
        for current_allocation in self.billing_changes:
            current_time, current_nodes = current_allocation
            # Repeat the previous cost when there is a gap between the current and previous times
            if current_time - previous_time > 1:
                node_costs.extend([node_costs[-1]] * (current_time - previous_time - 1))
            # Append the current cost
            billed_nodes = [
                node
                for node in current_nodes
                if NodeStates.get_state(node) in [NodeStates.BILLED, NodeStates.READY, NodeStates.REMOVING]
            ]
            node_costs.append(sum(node.ic.price.magnitude for node in billed_nodes))
            previous_time = current_time
        return node_costs

    def get_queue_waiting_times(self) -> dict[str, list[float]]:
        """
        Get the waiting times of requests in the processing queues. Requests of
        a given application can be served by different containers (servers), with different performance in req/s.
        Each application is modelled as a D/D/m queue with heterogeneous servers:
        - One application has as many servers as application containers.
        - Perfect load balancing, so the queue length of each container is proportional to container's performance.
        :return: One-second samples of queue waiting times.
        """

        app_workloads = self._workload_vectors
        app_performances = self.get_performances()

        # Calculate samples of the queue length. Difference (w-p) may not be multiple of 1 second
        frac_surplus = {app_name: 0.0 for app_name in app_performances}
        queue_length = {app_name: [0] for app_name in app_performances}
        for app in app_workloads:
            app_name = str(app)
            for w, p in zip(app_workloads[app][1:], app_performances[app_name][1:]):
                w_frac = w - int(w)
                if frac_surplus[app_name] >= w_frac:
                    frac_surplus[app_name] -= w_frac
                    w = floor(w)
                else:
                    frac_surplus[app_name] += (1 - w_frac)
                    w = ceil(w)
                queue_length[app_name].append(max(0, queue_length[app_name][-1] + w - p))

        # Samples of waiting times
        waiting_times = {
            app_name: [ql / wperf for ql, wperf in zip(queue_length[app_name], app_performances[app_name])]
            for app_name in queue_length
        }
        return waiting_times

    def write_workload_csv(self, csv_file: str) -> None:
        """
        Write a csv file with the workload for every application and time.
        :param csv_file: csv file to write
        """
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            workloads = self.get_workloads()
            writer.writerow([f'{app_name} (req/s)' for app_name in workloads])
            for row in zip(*workloads.values()):
                writer.writerow(row)

    def write_performance_csv(self, csv_file: str) -> None:
        """
        Write a csv file with the performance for every application and time.
        :param csv_file: csv file to write
        """
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            performances = self.get_performances()
            writer.writerow([f'{app_name} (req/s)' for app_name in performances])
            for row in zip(*performances.values()):
                writer.writerow(row)

    def write_cost_csv(self, csv_file: str) -> None:
        """
        Write a csv file with the cost at any time.
        :param csv_file: csv file to write
        """
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Cost ($/hour)'])
            for cost in self.get_cluster_cost():
                writer.writerow([f'{cost:.2f}'])

    def write_allocations(self, yaml_file: str):
        """
        Write the allocations as a YAML file.
        :param yaml_file: Output YAML file.
        """
        time_alloc = {}
        with open(yaml_file, "w") as f:
            for current_time, alloc in self.performance_changes:
                serializable_alloc = defaultdict(lambda: {})
                for node in alloc:
                    for cg in node.cgs:
                        serializable_alloc[f"{node.ic.name}-{node.id}"][str(cg.cc)] = cg.replicas
                time_alloc[current_time] = dict(serializable_alloc)
            yaml_dump(time_alloc, f)

    @staticmethod
    def plot(dict_values: dict[str, int], title: str = None, unit:str = None):
        """
        Plot a curve per dictionary entry with values at every time second.
        :param dict_values: Dictionary to plot, example: {'app0': [1.4, 3.0, 3.1], 'app1': [1.0, 1.0, 3.1]}.
        :param title: Title of the plot.
        :param unit: Unit for the vertical axis
        """
        plt.figure()
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        color_index = 0
        last_time = len(dict_values[list(dict_values.keys())[0]])
        times = list(range(last_time))
        for label, dict_values in dict_values.items():
            plt.plot(times, dict_values, linestyle='-', color=colors[color_index], label=label)
            color_index += 1
            if color_index == len(colors):
                color_index = 0
        plt.xlabel('Time (s)')
        if unit is not None:
            plt.ylabel(unit)
        if title is not None:
            plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_bar(dict_values: dict[str, list[int]], title: str = None, unit:str = None):
        """
        Plot a bar chart showing non-negative values from a dictionary of time series.

        :param dict_values: Dictionary to plot. Example: {'app0': [0.1, 2.0, 3.1], 'app1': [-1.0, 1.0, 3.1]}.
                            Only zero or positive values are displayed; negative values are ignored.
        :param title: Title of the plot.
        :param unit: Label for the vertical axis (e.g., "MB", "requests", etc.).
        """
        plt.figure()
        # Prepare labels and values
        labels = list(dict_values.keys())
        values = [np.array(dict_values[label]) for label in labels]
        x = np.arange(len(values[0]))

        # Replace negative values with NaN
        masked_values = [np.where(v >= 0, v, np.nan) for v in values]

        # Plot order to show the smallest in the foreground
        order = np.argsort([np.nan_to_num(v, nan=np.inf) for v in zip(*masked_values)], axis=1)

        # Keep track of which labels have already been added to the legend
        shown_labels = set()

        # Plot the highest bar and next the lowest
        for pos in range(len(x)):
            ordered_indices = order[pos]
            for idx in ordered_indices[::-1]:
                val = masked_values[idx][pos]
                if not np.isnan(val):
                    label = labels[idx] if labels[idx] not in shown_labels else "_nolegend_"
                    plt.bar(pos, val, color=f"C{idx}", label=label, zorder=idx, width=20, alpha=0.85)
                    shown_labels.add(labels[idx])

        # Títle and style
        if title:
            plt.title(title)
        plt.xlabel('Time (s)')
        if unit is not None:
            plt.ylabel(unit)
        plt.legend()

        plt.show()

