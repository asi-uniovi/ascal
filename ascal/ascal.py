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
from fcma import RequestsPerTime, Allocation, App, Vm, ContainerGroup
from ascal.autoscalers import Autoscaler
from ascal.hvpredictive import HVPredictiveAutoscaler
from ascal.hreactivehvpredictive import HReactiveHVPredictiveAutoscaler
from ascal.nodestates import NodeStates
from ascal.ascalconfig import AscalConfig

class Ascal:
    """
    This class provides methods to simulate the autoscaling of a system under a given load trace.
    """

    def __init__(self, ascal_config: AscalConfig, log=None, metrics_period:int = 60):
        """
        Ascal constructor.
        :param ascal_config: Configuration for the Ascal problem. It is not currently checked.
        """
        self._workload_vectors = ascal_config.workload_vectors
        self._autoscaler = ascal_config.autoscaler
        self._autoscaler.system = ascal_config.system
        self._autoscaler.apps = ascal_config.apps
        self._autoscaler.log_path = log # A string with the path of the log file
        self._metrics_period = metrics_period # Subsamples in each CPU utilization and queue waiting time sample
        self._cpu_util_subsamples: dict[Vm, list[float]] = {} # Subsamples of node CPU utilization
        self._app_perf_subsamples: dict[App, list[float]] = \
            {app: [] for app in self._autoscaler.apps} # Subsamples of application performance
        self._queue_waiting_time_subsamples: dict[App, list[float]] = \
            {app: [] for app in self._autoscaler.apps} # Subsamples of QWT for each application
        self._pending_workload: dict[App, list[int]] = \
            {app: 0.0 for app in self._autoscaler.apps} # Pending workload for each application
        self.time = -1 # Current simulation time
        self.last_time = len(next(iter(self._workload_vectors.values()))) - 1 # Last simulation time
        self.performance_changes: list[(int, Allocation)] = [] # Pairs time and allocation
        self.billing_changes: list[(int, Allocation)] = [] # Pairs time and allocation
        self.calc_times: dict[str, list[float]] = {"transition_times": [], "total_times": []} # Calculation times
        self.node_recycling_levels: list[float] = [] # List of node recycling levels
        self.container_recycling_levels: list[float] = [] # List of container recycling levels
        self.cpu_util: list[dict[Vm, float]] = [] # List of CPU utilizations for each node
        self.queue_waiting_time_k8s: dict[App, list[float]] = [] # List of queue waiting times for each application
        self.app_perf_k8s: dict[App, list[float]] = [] # List of performances for each application

    def _get_va(self) -> dict[App, float]:
        """
        Get the application's performance in req/s/core.
        :return: Application's performance in req/s/core.
        """
        va = {}
        for app_fm in self._autoscaler.system:
            cores = self._autoscaler.system[app_fm].cores
            perf = self._autoscaler.system[app_fm].perf
            a, _ = app_fm
            va[a] = perf.magnitude / cores.magnitude
        return va

    def _get_k8s_app_cores(self, workload: dict[App, RequestsPerTime], allocation: Allocation,
                           va: dict[App, float]) -> dict[App, dict[Vm, float]]:
        """
        Get application allocated cores in each node for the current workload. It assumes Weighted-Round-Robin 
        load balancing and container allocation cores are commited cores, similar to requests in Kubernetes. 
        :param workload: Workload for each application at the current time.
        :param allocation: Current allocation.
        :param va: Application's performance in req/s/core. 
        :return: Application's CPU cores in each node.
        """

        # The total wortkload comes from the workload at the current time plus the pending
        # workload from previous time
        wa = {app: (workload[app].magnitude + self._pending_workload[app]) for app in workload}

        # Get Va for each application, the application performance in req/s per requested core
        va = self._get_va()

        # List of nodes allocating at least one container that is not being removed (cc.app != None)
        nodes = [
            node for node in allocation 
            if len(node.cgs) > 0 and any(cg.cc.app != None for cg in node.cgs)
        ]

        # Output application cores
        cai = {app: {} for app in workload}

        # Application commited cores in each node. They are analogous to to requested cores in Kubernetes
        ccai = {app:{} for app in wa}
        for node in nodes:
             for cg in node.cgs:
                app = cg.cc.app
                # Containers being removed (associated to None applications) are ignored
                if app is None:
                    continue
                if node not in ccai[app]:
                    ccai[app][node] = 0.0
                ccai[app][node] += cg.cc.cores.magnitude * cg.replicas

        # Total commited cores for applications Aa
        cca = {app: sum(ccai[app][node] for node in ccai[app]) for app in ccai}
        # Node applications. They are removed from the problem as the algorithm progresses
        node_apps = {node: [app for app in ccai if node in ccai[app]] for node in nodes}
        # Total commited cores for applications allocated in nodes Ni
        cci = {node: sum(ccai[app][node] for app in node_apps[node]) for node in nodes}
        # Get Ra, the workload multiplier for applications Aa
        ra = {app: wa[app] / (va[app]*cca[app]) for app in ccai}
        # Node capacities in cores
        ci = {node: node.ic.cores.magnitude for node in nodes}
        # Get Mi, the available cores multiplier for nodes Ni
        mi = {node: ci[node] / cci[node] for node in nodes}

        # Iterate while there are nodes. Nodes are removed from the problem when they exclusively allocate
        # removed applications, or are saturated nodes.
        while len(mi) > 0:
            # While there is one application Ar with Rr <= Mi for all nodes Ni
            while True:
                lowest_mi_node = min(mi, key=lambda node: mi[node])
                lowest_ra_app = min(node_apps[lowest_mi_node], key=lambda app: ra[app])
                if not ra[lowest_ra_app] <= mi[lowest_mi_node]:
                    break
                app = lowest_ra_app

                # For all the nodes allocating the application to remove
                for node in mi:
                    if node in ccai[app]:
                        # Calculate the application allocated cores
                        cai[app][node] = (wa[app] / va[app]) * (ccai[app][node] / cca[app])
                        # Update node parameters in the problem
                        cci[node] = cci[node] - ccai[app][node]
                        ci[node] = ci[node] - cai[app][node]
                        # Remove the application from the node
                        node_apps[node].remove(app)
                # Remove the application from the problem
                del ra[app]
                for node in dict(mi):
                    if len(node_apps[node]) > 0:
                        # Update the available cores multiplier in the node
                        mi[node] = ci[node] / cci[node]
                    else:
                        # Remove nodes without applications
                        del mi[node]

                # Check the termination condition
                if len(mi) == 0:
                    return cai           

            # The node with the lowest Mi cannot provide enough cores for all of its allocated applications 
            # and it is the first to saturate
            lowest_mi_node = min(mi, key=lambda node: mi[node])
            for app in node_apps[lowest_mi_node]:
                # Calculate the application allocated cores
                cai[app][lowest_mi_node] = ccai[app][lowest_mi_node] * mi[lowest_mi_node]
                # Update application parameters in the problem
                cca[app] -= ccai[app][lowest_mi_node]
                wa[app] -= cai[app][lowest_mi_node] * va[app]
                if cca[app] > 0.0:
                    ra[app] = (wa[app] / va[app]) / cca[app]
            # Remove the saturated node from the problem
            del mi[lowest_mi_node]

        return cai
    
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
            
            # Calculate workload predictions for predictive autoscalers
            if self.time == 0 and (isinstance(self._autoscaler, HVPredictiveAutoscaler) or
                                   isinstance(self._autoscaler, HReactiveHVPredictiveAutoscaler)):
                Autoscaler.workload_predictions(self._autoscaler, self._workload_vectors)

            # Time message every 100 seconds to show the progress of the simulation
            if self.time % 100 == 0:
                print(f'Time: {self.time} s')

            # Change workloads to the correct format
            workloads = {}
            for app in self._workload_vectors:
                workload = RequestsPerTime(f"{self._workload_vectors[app][self.time] * 3600}  req/hour")
                workloads[app] = workload

            # Run the autoscaling with the selected autoscaler and the current workloads. 
            # The autoscaler may change the allocation of nodes and containers.
            statistics = self._autoscaler.run(workloads)

            # Application's performance in req/s/core
            va = self._get_va() 

            # Get applications CPU utilizations in each node for the current workloads. It should be noted
            # that pending workloads are added to current workloads to calculate CPU utilizations
            cai = self._get_k8s_app_cores(workloads, self._autoscaler.allocation, va)

            # Update CPU utilization subsamples in each node
            cpu_util = {
                node: sum(cai[a][node] / node.ic.cores.magnitude for a in cai if node in cai[a]) 
                for node in self._autoscaler.allocation
            }
            for node in dict(self._cpu_util_subsamples):
                if node not in cpu_util: # Nodes may change in the allocation
                    del self._cpu_util_subsamples[node]
            for node in cpu_util:
                if node not in self._cpu_util_subsamples:
                    self._cpu_util_subsamples[node] = []
                if len(self._cpu_util_subsamples[node]) == self._metrics_period:
                    self._cpu_util_subsamples[node].pop(0)
                self._cpu_util_subsamples[node].append(cpu_util[node])

            # Update application's performance subsamples
            app_perf = {app: va[app] * sum(cai[app][node] for node in cai[app]) for app in cai}
            for app in app_perf:
                if len(self._app_perf_subsamples[app]) == self._metrics_period:
                    self._app_perf_subsamples[app].pop(0)
                self._app_perf_subsamples[app].append(app_perf[app])

            # Calculate the pending workload for each application assumig 1 second as sampling period
            for app in self._pending_workload:
                self._pending_workload[app] = \
                    round(max(0.0, self._pending_workload[app] + workloads[app].magnitude - app_perf[app]), 6)

            # Update the queue waiting time subsamples in seconds for each application
            for app in self._pending_workload:
                if len(self._queue_waiting_time_subsamples[app]) == self._metrics_period:
                    self._queue_waiting_time_subsamples[app].pop(0)
                if self._pending_workload[app] == 0:
                    self._queue_waiting_time_subsamples[app].append(0.0)
                elif app_perf[app] > 0:
                    self._queue_waiting_time_subsamples[app].append(self._pending_workload[app] / app_perf[app])

            # Update autoscaling info
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
            self.cpu_util = {node: sum(self._cpu_util_subsamples[node]) / len(self._cpu_util_subsamples[node])
                             for node in self._cpu_util_subsamples}
            self.app_perf_k8s = {app: sum(self._app_perf_subsamples[app]) / len(self._app_perf_subsamples[app])
                             for app in self._app_perf_subsamples}
            self.queue_waiting_time_k8s = {
                app: sum(self._queue_waiting_time_subsamples[app]) / len(self._queue_waiting_time_subsamples[app])
                for app in self._queue_waiting_time_subsamples
            }                                                                                                                                    
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

    def get_performances(self, k8s=False) -> dict[str, list[float]]:
        """
        Get application performances.
        :param k8s: If it is False the returned performances are commited performances. The committed 
        performance value ensures each application can handle up to that amount of load. 
        When k8s is True, it returns the real performances (not commited ones) assuming CPU cores are 
        analogous to K8s CPU requests, so application performances can be higher than commited values 
        when other applications experience low load, below their committed performance.
.        :return: For each application the performances in req/s at every second, starting at 0 seconds.
        """
        if k8s:
            return {str(app): [self.app_perf_k8s[app]] for app in self.app_perf_k8s}

        # Application performance is the maximum performance an application can provide, obtained from the
        # cores allocated to its containers
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
    
    def get_node_cpu_util(self) -> dict[Vm, float]:
        """
        Get node CPU utilizations, which depend on allocations an workloads.
        :return: CPU utilizations in [0.0, 1.0].
        """
        return self.cpu_util

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

    def get_queue_waiting_times(self, k8s=False) -> dict[str, list[float]]:
        """
        Get the waiting times of requests in the processing queues. Requests of
        a given application can be served by different containers (servers), with different performance in req/s.
        Each application is modelled as a D/D/m queue with heterogeneous servers:
        - One application has as many servers as application containers.
        - Perfect load balancing, so the queue length of each container is proportional to container's performance.
        :param k8s: If True, it returns queue waiting times assuming CPU cores are analogous to K8s CPU requests, so
        performances can be higher than the commited values.
        :return: One-second samples of queue waiting times.
        """
        if k8s:
            return {str(app): [self.queue_waiting_time_k8s[app]] for app in self.queue_waiting_time_k8s}        
        
        # Queue waiting times are obtained using the commited application performance
        app_performances = self.get_performances()

        # Calculate samples of the queue length. Difference (w-p) may not be multiple of 1 second.
        # This extra complexity could be eliminated, but it is maintained to adhere as closely as 
        # possible to the results of previous scientific publications.
        frac_surplus = {app_name: 0.0 for app_name in app_performances}
        queue_length = {app_name: [0] for app_name in app_performances}
        for app in self._workload_vectors:
            app_name = str(app)
            for w, p in zip(self._workload_vectors[app][1:], app_performances[app_name][1:]):
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

    def write_performance_csv(self, csv_file: str, k8s=False) -> None:
        """
        Write a csv file with the performance for every application and time.
        :param csv_file: csv file to write.
        :param k8s: It writes commited performances when it is False and real performances
        assuming CPU cores analogous to K8s CPU requests otherwise.
        """
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            performances = self.get_performances(k8s)
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

