"""
Main module of the ascal package. It defines class Ascal to calculate the sequence of deployments for
a given autoscaler and applications
"""
from copy import deepcopy
from yaml import safe_load
import matplotlib.pyplot as plt

import csv
from fcma import (
    RequestsPerTime,
    InstanceClassFamily,
    ComputationalUnits,
    Storage,
    AppFamilyPerf,
    Allocation,
    App,
    System
)
from autoscalers import (
    AutoscalerTypes,
    HReactiveAutoscaler,
    HVReactiveAutoscaler,
    HVPredictiveAutoscaler,
    TimedOps
)
from nodestates import NodeStates


class AscalConfig:
    def __init__(self, system: System = None, workload_vectors: dict[App, [RequestsPerTime]] = None,
                 autoscaler_type: AutoscalerTypes = AutoscalerTypes.H_REACTIVE):
        """
        Ascal configuration.
        :param system: System, made up of applications and containers
        :param workload_vectors: Application workloads
        :param autoscaler_type: Autoscaler to be used
        """
        self.system = system
        if self.system is None:
            self.apps = None
        else:
            self.apps = list(set(app for app, _ in self.system.keys()))
        self.workload_vectors = workload_vectors
        if autoscaler_type == AutoscalerTypes.H_REACTIVE:
            self.autoscaler = HReactiveAutoscaler()
        elif autoscaler_type == AutoscalerTypes.HV_REACTIVE:
            self.autoscaler = HVReactiveAutoscaler()
        elif autoscaler_type == AutoscalerTypes.HV_PREDICTIVE:
            self.autoscaler = HVPredictiveAutoscaler()


    @staticmethod
    def get_from_config_yaml(yaml_file:str, ic_family: InstanceClassFamily):
        """
        Get Ascal configuration from a YAML file and a familiy of instance classes.
        :param yaml_file: YAML file with the configuration.
        :param ic_family: Family containing instance classes for nodes with the same base hardware.
        :return: An Ascal configuration object.
        """
        with open(yaml_file, "r") as file:
            data = safe_load(file)
            AscalConfig.validate_config(data)
            config = AscalConfig()

            # Creation/removal times for contaienrs and nodes
            timing_args = TimedOps.TimingArgs(data['node_time_to_billing'], data['node_creation_time'],
                                              data['node_removal_time'], data['container_creation_time'],
                                              data['container_removal_time'])
            # Set the autoscaler
            if data['autoscaler'] == 'h_reactive':
                config.autoscaler = HReactiveAutoscaler(data['autoscalers']['h_reactive']['time_period'],
                                                        data['autoscalers']['h_reactive']['desired_cpu_utilization'],
                                                        data['autoscalers']['h_reactive']['node_utilization_threshold'],
                                                        timing_args)
            elif data['autoscaler'] == 'hv_reactive':
                algorithm = AutoscalerTypes.FCMA
                if data['autoscalers']['hv_reactive']['algorithm'] == 'fcma1':
                    algorithm = AutoscalerTypes.FCMA1
                elif  data['autoscalers']['hv_reactive']['algorithm'] == 'fcma2':
                    algorithm = AutoscalerTypes.FCMA2
                elif data['autoscalers']['hv_reactive']['algorithm'] == 'fcma3':
                    algorithm = AutoscalerTypes.FCMA3
                config.autoscaler = HVReactiveAutoscaler(data['autoscalers']['hv_reactive']['time_period'],
                                                         data['autoscalers']['hv_reactive']['desired_cpu_utilization'],
                                                         timing_args, algorithm,
                                                         data['autoscalers']['hv_reactive']['transition_time_budget'])
            elif data['autoscaler'] == 'hv_predictive':
                algorithm = AutoscalerTypes.FCMA
                if data['autoscalers']['hv_predictive']['algorithm'] == 'fcma1':
                    algorithm = AutoscalerTypes.FCMA1
                elif  data['autoscalers']['hv_predictive']['algorithm'] == 'fcma2':
                    algorithm = AutoscalerTypes.FCMA2
                elif data['autoscalers']['hv_predictive']['algorithm'] == 'fcma3':
                    algorithm = AutoscalerTypes.FCMA3
                config.autoscaler = HVPredictiveAutoscaler(data['autoscalers']['hv_predictive']['prediction_window'],
                                                         data['autoscalers']['hv_predictive']['prediction_percentile'],
                                                         timing_args, algorithm)
            # Set the system
            config.system = {}
            config.apps = []
            for app_name in data['apps']:
                app = App(app_name)
                config.apps.append(app)
                cores = ComputationalUnits(data['apps'][app_name]['container']['cpu'])
                gib = Storage(data['apps'][app_name]['container']['mem'])
                perf = RequestsPerTime(data['apps'][app_name]['container']['perf'])
                aggs = (1,)
                if 'aggs' in data['apps'][app_name]['container']:
                    aggs = tuple(data['apps'][app_name]['container']['aggs'])
                app_family_perf = AppFamilyPerf(cores=cores, mem=gib, perf=perf, aggs=aggs)
                config.system[(app, ic_family)] = app_family_perf
            config.autoscaler.system = config.system

            # Set workloads. For each spplication:
            # 1. Read the set of load samples.
            # 2. Repeat the set of samples "repeat" times.
            # 3. Add "load_offset" to all the samples and multiply the result by "load_mult".
            # 4. Repeat the first sample "time_offset" times at the beginning.
            # 5. Repeat each sample "time_interval" times.
            # 6. Fill at the end using the last sample, or truncate the last samples
            # to achieve the desired number of samples.
            config.workload_vectors = {}
            max_time = data['simulation_time'] - 1
            for app in config.apps:
                app_name = app.name
                processed_load = [] # Application load after being processed
                time_interval = data['apps'][app_name]['load']['time_interval']
                repeat = data['apps'][app_name]['load']['repeat']
                time_offset = data['apps'][app_name]['load']['time_offset']
                load_offset = data['apps'][app_name]['load']['load_offset']
                load_mult = data['apps'][app_name]['load']['load_mult']
                load_file = data['apps'][app_name]['load']['file']
                with open(load_file, newline='') as csv_file:
                    reader = csv.reader(csv_file)
                    loads = list(reader)[1:] * repeat
                    first_processed_load = (int(loads[0][0]) + load_offset) * load_mult
                    time_index = time_offset * time_interval
                    processed_load.extend([first_processed_load] * time_index)
                load_index = 0
                while time_index <= max_time and load_index < len(loads):
                    load_val = (int(loads[load_index][0]) + load_offset) * load_mult
                    processed_load.extend([load_val] * time_interval)
                    load_index += 1
                    time_index += time_interval
                if time_index < max_time:
                    last_processed_load = processed_load[-1]
                    processed_load.extend([last_processed_load] * (max_time - time_index + 1))
                if time_index >= max_time:
                    processed_load = processed_load[0: max_time + 1]
                config.workload_vectors[app] = processed_load

        return config

    @staticmethod
    def validate_config(config):
        """
        Validates the given configuration.
        :param config: The configuration to validate.
        :raises ValueError: If the validation fails with a specific error message.
        """

        def check_fields(data: dict, keys: list[str], types: list[type]):
            """
            Check the fields type and value for the given data.
            :param data: Dictionary with the data to check
            :param keys: List of keys to check
            :param types: List of epected types for the keys
            """
            for i in range(len(keys)):
                if keys[i] not in data:
                    raise ValueError(f"Key '{keys[i]}' is missing in {str(data)}")
                if not isinstance(data[keys[i]], types[i]) or (types[i] == int and data[keys[i]] < 0):
                    raise ValueError(f"Invalid value of key '{keys[i]}'")

        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")

        # Validate autoscalers
        if "autoscalers" not in config:
            raise ValueError("'autoscalers' key is missing")
        for key in config["autoscalers"]:
            if not isinstance(config["autoscalers"][key], dict):
                raise ValueError("Available autoscalers must be dictionaries")
            if key not in ["h_reactive", "hv_reactive", "hv_predictive"]:
                raise ValueError("Valid autoscalers: 'h_reactive', 'hv_reactive', 'hv_predictive'")
            if key == "h_reactive":
                check_fields(config["autoscalers"][key],
                             ["time_period", "desired_cpu_utilization", "node_utilization_threshold"],
                             [int, float, float])
            elif key == "hv_reactive":
                check_fields(config["autoscalers"][key],
                            ["time_period", "desired_cpu_utilization", "algorithm", "transition_time_budget"],
                             [int, float, str, int])
                if config["autoscalers"]["hv_reactive"]["algorithm"] not in ["fcma", "fcma1", "fcma2", "fcma3"]:
                    raise ValueError("Valid algorithms are 'fcma', 'fcma1', 'fcma2' or 'fcma3'")
            elif key == "hv_predictive":
                check_fields(config["autoscalers"][key],
                             ["prediction_window", "prediction_percentile", "algorithm"],
                             [int, int, str])
                if config["autoscalers"]["hv_predictive"]["algorithm"] not in ["fcma", "fcma1", "fcma2", "fcma3"]:
                    raise ValueError("Valid algorithms are 'fcma', 'fcma1', 'fcma2' or 'fcma3'")
        if "autoscaler" not in config:
            raise ValueError("Field 'autoscaler' is missing")
        if config["autoscaler"] not in config["autoscalers"].keys():
            raise ValueError("The selected autoscaler is invalid")

        # Validate temporal parameters
        check_fields(config,
             ["node_time_to_billing", "node_creation_time", "node_removal_time", "container_creation_time",
              "container_removal_time", "simulation_time"], [int, int, int, int, int, int])

        # Validate applications
        if "apps" not in config:
            raise ValueError("key 'apps' is missing")
        if len(config["apps"]) == 0:
            raise ValueError("At least one application is required")
        for key, val in config["apps"].items():
            if not isinstance(key, str):
                raise ValueError("Application names must be strings")
            if "load" not in val:
                raise ValueError(f"'load' key is missing in application '{val}'")
            if "container" not in val:
                raise ValueError(f"'container' key is missing in application '{val}'")
            check_fields(config["apps"][key]["load"],
                         ["file", "time_interval", "repeat", "load_offset", "load_mult", "time_offset"],
                         [str, int, int, int, int, int])
            check_fields(config["apps"][key]["container"],
                         ["cpu", "mem", "perf", "aggs"],
                         [str, str, str, list])
            for v in config["apps"][key]["container"]["aggs"]:
                if not isinstance(v, int) and v >= 1:
                    raise ValueError("Aggregations must be iinteger equal to or higher than zero")
        

class Ascal:
    """
    This class provides methods to simulate the autoscaling of a system under a given load.
    """

    def __init__(self, ascal_config: AscalConfig, log=None):
        """
        Ascal contructor.
        :param ascal_config: Configuration for the Ascal problem.
        """
        self._workload_vectors = ascal_config.workload_vectors
        self._system = ascal_config.system
        self._autoscaler = ascal_config.autoscaler
        self._autoscaler.apps = ascal_config.apps
        self._autoscaler.log_path = log # A string with the path of the log file
        self.time = -1 # Current simulation time
        self.last_time = len(next(iter(self._workload_vectors.values()))) - 1 # Last simulation time
        self.performance_changes: list[(int, Allocation)] = [] # Pairs time and allocation
        self.billing_changes: list[(int, Allocation)] = [] # Pairs time and allocation
        self.calc_times: list[float] = [] # Calculation times

    def run(self, break_point: int = None) -> bool:
        """
        Continue simulating autoscaling until the given breakpoint.
        :param break_point: Simulate until this time. When it is -1 it simulates until the end.
        :return: True when the simulation has reached the end.
        """
        if break_point is None or break_point >= self.time:
            # The breakpoint is placed at the last time in simulation
            break_point = self.last_time
        while self.time < break_point:
            self.time += 1
            if self.time == 0 and hasattr(self._autoscaler, 'prediction_window'):
                self._autoscaler.workload_predictions(self._workload_vectors)
            if self.time % 100 == 0:
                print(f'Time: {self.time} s')
            workloads = {}
            for app in self._workload_vectors:
                workload = RequestsPerTime(f"{self._workload_vectors[app][self.time] * 3600}  req/hour")
                workloads[app] = workload
            # Save the current allocation as an allocation change or a node change. Nodes are considered to change
            # onece it begin to be billed, even if they can not allocate contaoners yet
            performance_changed, billing_changed, calculation_time = self._autoscaler.run(workloads)
            if performance_changed or billing_changed or self.time == break_point:
                allocation_copy = (self.time, deepcopy(self._autoscaler.allocation))
                if performance_changed or self.time == break_point:
                    self.performance_changes.append(allocation_copy)
                if billing_changed or self.time == break_point:
                    self.billing_changes.append(allocation_copy)
            self.calc_times.append(calculation_time)
        self._autoscaler.log_allocation_summary()

    def get_workloads(self) -> dict[str, list[int]]:
        """
        Get application workloads.
        :return: For each application the workloads in req/s at every second, starting from 0 seconds.
        """
        return {str(key): value for key, value in self._workload_vectors.items()}

    def get_performances(self) -> dict[str, list[int]]:
        """
        Gets application performances.
        :return: For each application the performances in req/s at every second, starting at 0 seconds.
        """
        app_perfs = {str(app): [] for app in self._workload_vectors}

        previous_time = -1
        for current_allocation in self.performance_changes:
            current_time, current_nodes = current_allocation
            # Repeat the previous allocation performances
            if current_time - previous_time > 1:
                for app_name, perf in app_perfs.items():
                    app_perfs[app_name].extend([perf[-1]] * (current_time - previous_time - 1))
            # Get application performances for the current allocation
            current_perfs = {str(app): 0 for app in self._workload_vectors}
            for node in current_nodes:
                for cg in node.cgs:
                    app = cg.cc.app
                    if app is None:
                        continue
                    current_perfs[str(app)] += cg.cc.perf.to('req/s').magnitude * cg.replicas

            # Append the current allocation performances
            for app_name in app_perfs:
                app_perfs[app_name].append(current_perfs[app_name])

            # Prepare for the next allocation change
            previous_time = current_time
        return app_perfs

    def get_cluster_cost(self) -> list[int]:
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

    @staticmethod
    def plot(dict_values: dict[str, int], title: str = None, unit:str = None):
        """
        Plot a curve per dictionary entry with values at every time second.
        :param dict_values: Dictionary to plot, example: {'app0': [1.4, 3.0, 3.1], 'app1': [1.0, 1.0, 3.1]}.
        :param title: Title of the plot
        :param unit: Unit for the vertical axis
        """
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
