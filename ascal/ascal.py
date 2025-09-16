"""
Main module of the ascal package. It defines class Ascal to calculate the sequence of allocations for
a given autoscaler and applications
"""

from math import ceil, floor
from copy import deepcopy
from yaml import safe_load
from yaml import dump as yaml_dump
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
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
from ascal.autoscalers import Autoscaler, AllocationSolver, AutoscalerTypes, TimedOps
from ascal.hreactive import HReactiveAutoscaler
from ascal.hvreactive import HVReactiveAutoscaler
from ascal.hvpredictive import HVPredictiveAutoscaler
from ascal.hreactivehvreactive import HReactiveHVReactiveAutoscaler
from ascal.hreactivehvpredictive import HReactiveHVPredictiveAutoscaler
from ascal.nodestates import NodeStates


class AscalConfig:
    def __init__(self, system: System = None, workload_vectors: dict[App, list[RequestsPerTime]] = None,
                 autoscaler_type: AutoscalerTypes = AutoscalerTypes.H_REACTIVE):
        """
        Ascal configuration.
        :param system: System, made up of applications and containers.
        :param workload_vectors: Application workloads.
        :param autoscaler_type: Autoscaler to be used.
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
        elif autoscaler_type == AutoscalerTypes.H_REACTIVE_HV_REACTIVE:
            self.autoscaler = HReactiveHVReactiveAutoscaler()
        elif autoscaler_type == AutoscalerTypes.H_REACTIVE_HV_PREDICTIVE:
            self.autoscaler = HReactiveHVPredictiveAutoscaler()

    @staticmethod
    def _get_check_aggs(data, apps: list[App], aggs) -> dict[App, list[int]]:
        """
        Get the aggregation levels for the applications using the horizontal autoscaler.
        :param data: Data read from the YAML file.
        :param apps: Applications.
        :aggs: Aggregations read from YAML file.
        :return: A list of aggregation levels for each application.
        """
        aggs_dict = {app: [] for app in apps}

        if isinstance(aggs, int):  # Aggregations in the autoscaler can be a single integer
            agg = aggs  # Aggregations are in fact a single aggregation
            for app in apps:
                if agg not in data['apps'][app.name]['container']['aggs']:
                    raise ValueError("Invalid aggregation level in h_autoscaler")
                aggs_dict[app] = [agg]
        elif isinstance(aggs, list):  # Aggregations in the autoscaler can be a list of integers
            for app in apps:
                for agg in aggs:
                    if agg not in data['apps'][app.name]['container']['aggs']:
                        raise ValueError("Invalid aggregation level in h_autoscaler")
                aggs_dict[app] = aggs
        elif isinstance(aggs, dict):  # Aggregations in the autoscaler can be a dictionary
            app_aggs = aggs
            aggs_dict = {app: [1] for app in apps}  # Default aggregations
            app_names = [app.name for app in apps]
            for app_name, aggs in app_aggs.items():
                try:
                    app = apps[app_names.index(app_name)]
                except ValueError:
                    raise ValueError("Invalid app in agg field of h_autoscaler")
                if isinstance(aggs, int):
                    agg = aggs  # Aggregations are in fact a single aggregation
                    if agg not in data['apps'][app.name]['container']['aggs']:
                        raise ValueError("Invalid aggregation level in h_autoscaler")
                    aggs_dict[app] = [agg]
                elif isinstance(aggs, list):
                    for agg in aggs:
                        if agg not in data['apps'][app_name]['container']['aggs']:
                            raise ValueError("Invalid aggregation level in h_autoscaler")
                    aggs_dict[app] = aggs
                else:
                    raise ValueError("Invalid app in agg field of h_autoscaler")
        else:
            raise ValueError("Invalid agg value in h_autoscaler. Use list and int ")
        return aggs_dict

    @staticmethod
    def _set_autoscaler(config, data:dict, timing_args: TimedOps.TimingArgs):
        """
        Set the autoscaler from data dictionary
        :param config: System configuration.
        :param data: Application data.
        :param timing_args: Times to create/remove nodes/containers.
        :return:
        """
        if data['autoscaler'] == 'h_reactive':
            config.autoscaler = HReactiveAutoscaler(
                data['autoscalers']['h_reactive']['time_period'],
                data['autoscalers']['h_reactive']['desired_cpu_utilization'],
                data['autoscalers']['h_reactive']['node_utilization_threshold'],
                AscalConfig._get_check_aggs(data, config.apps, aggs=data["autoscalers"]['h_reactive']["aggs"]),
                timing_args
            )
        elif data['autoscaler'] == 'hv_reactive':
            algorithm = AllocationSolver.FCMA
            if data['autoscalers']['hv_reactive']['algorithm'] == 'fcma1':
                algorithm = AllocationSolver.FCMA1
            elif data['autoscalers']['hv_reactive']['algorithm'] == 'fcma2':
                algorithm = AllocationSolver.FCMA2
            elif data['autoscalers']['hv_reactive']['algorithm'] == 'fcma3':
                algorithm = AllocationSolver.FCMA3
            elif data['autoscalers']['hv_reactive']['algorithm'] == 'mncf':
                algorithm = AllocationSolver.MNCF
            config.autoscaler = HVReactiveAutoscaler(
                data['autoscalers']['hv_reactive']['time_period'],
                data['autoscalers']['hv_reactive']['desired_cpu_utilization'],
                timing_args,
                algorithm,
                data['autoscalers']['hv_reactive']['transition_time_budget'],
                data['autoscalers']['hv_reactive']['hot_node_scale_up']
            )
        elif data['autoscaler'] == 'hv_predictive':
            algorithm = AllocationSolver.FCMA
            if data['autoscalers']['hv_predictive']['algorithm'] == 'fcma1':
                algorithm = AllocationSolver.FCMA1
            elif data['autoscalers']['hv_predictive']['algorithm'] == 'fcma2':
                algorithm = AllocationSolver.FCMA2
            elif data['autoscalers']['hv_predictive']['algorithm'] == 'fcma3':
                algorithm = AllocationSolver.FCMA3
            elif data['autoscalers']['hv_predictive']['algorithm'] == 'mncf':
                algorithm = AllocationSolver.MNCF
            config.autoscaler = HVPredictiveAutoscaler(
                data['autoscalers']['hv_predictive']['prediction_window'],
                data['autoscalers']['hv_predictive']['prediction_percentile'],
                timing_args,
                algorithm,
                data['autoscalers']['hv_predictive']['transition_time_budget'],
                data['autoscalers']['hv_predictive']['hot_node_scale_up']
            )
        elif data['autoscaler'] == 'h_reactive_hv_reactive':
            algorithm = AllocationSolver.FCMA
            if data['autoscalers']['h_reactive_hv_reactive']['hv_algorithm'] == 'fcma1':
                algorithm = AllocationSolver.FCMA1
            elif data['autoscalers']['h_reactive_hv_reactive']['hv_algorithm'] == 'fcma2':
                algorithm = AllocationSolver.FCMA2
            elif data['autoscalers']['h_reactive_hv_reactive']['hv_algorithm'] == 'fcma3':
                algorithm = AllocationSolver.FCMA3
            elif data['autoscalers']['h_reactive_hv_reactive']['hv_algorithm'] == 'mncf':
                algorithm = AllocationSolver.MNCF
            config.autoscaler = HReactiveHVReactiveAutoscaler(
                data['autoscalers']['h_reactive_hv_reactive']['h_time_period'],
                data['autoscalers']['h_reactive_hv_reactive']['desired_cpu_utilization'],
                data['autoscalers']['h_reactive_hv_reactive']['h_node_utilization_threshold'],
                timing_args,
                algorithm,
                data['autoscalers']['h_reactive_hv_reactive']['hv_time_period'],
                data['autoscalers']['h_reactive_hv_reactive']['hv_transition_time_budget'],
                data['autoscalers']['h_reactive_hv_reactive']['hot_node_scale_up']
            )
        elif data['autoscaler'] == 'h_reactive_hv_predictive':
            algorithm = AllocationSolver.FCMA
            if data['autoscalers']['h_reactive_hv_predictive']['hv_algorithm'] == 'fcma1':
                algorithm = AllocationSolver.FCMA1
            elif data['autoscalers']['h_reactive_hv_predictive']['hv_algorithm'] == 'fcma2':
                algorithm = AllocationSolver.FCMA2
            elif data['autoscalers']['h_reactive_hv_predictive']['hv_algorithm'] == 'fcma3':
                algorithm = AllocationSolver.FCMA3
            elif data['autoscalers']['h_reactive_hv_predictive']['hv_algorithm'] == 'mncf':
                algorithm = AllocationSolver.MNCF
            config.autoscaler = HReactiveHVPredictiveAutoscaler(
                data['autoscalers']['h_reactive_hv_predictive']['h_time_period'],
                data['autoscalers']['h_reactive_hv_predictive']['h_desired_cpu_utilization'],
                data['autoscalers']['h_reactive_hv_predictive']['h_node_utilization_threshold'],
                timing_args,
                algorithm,
                data['autoscalers']['h_reactive_hv_predictive']['hv_prediction_window'],
                data['autoscalers']['h_reactive_hv_predictive']['hv_prediction_percentile'],
                data['autoscalers']['h_reactive_hv_predictive']['hv_transition_time_budget'],
                data['autoscalers']['h_reactive_hv_predictive']['hot_node_scale_up']
            )

    @staticmethod
    def _set_apps_workload(config, data:dict):
        """
        Set application's workload from data dictionary.
        :param config: System configuration.
        :param data: Data dictionary.
        """

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
            processed_load = []  # Application load after being processed
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

    @staticmethod
    def get_from_config_yaml(yaml_file:str, ic_family: InstanceClassFamily):
        """
        Get Ascal configuration from a YAML file and a family of instance classes.
        :param yaml_file: YAML file with the configuration.
        :param ic_family: Family containing instance classes for nodes with the same base hardware.
        :return: An Ascal configuration object.
        """
        with open(yaml_file, "r") as file:
            data = safe_load(file)
            AscalConfig.validate_config(data)
            config = AscalConfig()

            config.system = {}
            config.apps = []
            app_names = []
            for app_name in data['apps']:
                app = App(app_name)
                config.apps.append(app)
                app_names.append(app.name)
                aggs = tuple(data['apps'][app_name]['container']['aggs'])
                cores = ComputationalUnits(data['apps'][app_name]['container']['cpu'])
                mem_agg1 = Storage(data['apps'][app_name]['container']['mem'])
                gib = tuple(mem_agg1 for _ in aggs)
                perf = RequestsPerTime(data['apps'][app_name]['container']['perf'])
                app_family_perf = AppFamilyPerf(cores=cores, mem=gib, perf=perf, aggs=aggs)
                config.system[(app, ic_family)] = app_family_perf

        config.autoscaler.system = config.system

        # Creation/removal times for containers and nodes
        timing_args = data["timing_args"]
        timing_args = TimedOps.TimingArgs(timing_args['node_time_to_billing'], timing_args['node_creation_time'],
                                          timing_args['node_removal_time'], timing_args['hot_node_scale_up_time'],
                                          timing_args['container_creation_time'],
                                          timing_args['container_removal_time'])
        # Set the autoscaler
        AscalConfig._set_autoscaler(config, data, timing_args)

        # Set application's workload
        try:
            AscalConfig._set_apps_workload(config, data)
        except:
            raise ValueError("Error reading the workload file")

        return config

    @staticmethod
    def _check_fields(data: dict, keys: list[str], types: list[type]):
        """
        Check the fields type and value for the given data.
        :param data: Dictionary with the data to check.
        :param keys: List of keys to check.
        :param types: List of expected types for the keys.
        """
        for i in range(len(keys)):
            if keys[i] not in data:
                raise ValueError(f"Key '{keys[i]}' is missing in {str(data)}")
            if not isinstance(data[keys[i]], types[i]) or (types[i] == int and data[keys[i]] < 0):
                raise ValueError(f"Invalid value of key '{keys[i]}'")

    @staticmethod
    def _validate_autoscalers(config):
        """
        Validate autoscaler parameters in the configuration.
        :param config: Configuration.
        :raises ValueError: When a validation fails.
        """
        valid_autoscalers = ["h_reactive", "hv_reactive", "hv_predictive",
                             "h_reactive_hv_reactive", "h_reactive_hv_predictive"]
        for key in config["autoscalers"]:
            if not isinstance(config["autoscalers"][key], dict):
                raise ValueError("Available autoscalers must be dictionaries")
            if key not in valid_autoscalers:
                raise ValueError(f"Valid autoscalers: {valid_autoscalers}")

            if key == "h_reactive":
                properties = ["time_period", "desired_cpu_utilization", "node_utilization_threshold"]
                AscalConfig._check_fields(config["autoscalers"][key], properties,[int, float, float])
                # Check properties. Note that autoscaler aggs are checked by _get_check_aggs()
                if set(properties) | {"aggs"} != set(config["autoscalers"][key]):
                    raise ValueError(f"Invalid or missing property in {key}")
                if config["autoscalers"][key]["time_period"] < 0:
                    raise ValueError("Time period must be possitive")
                if config["autoscalers"][key]["desired_cpu_utilization"] < 0.1:
                    raise ValueError("Desired CPU utilization must be >= 0.1")
                if config["autoscalers"][key]["node_utilization_threshold"] < 0.1:
                    raise ValueError("Desired CPU utilization must be >= 0.1")

            elif key == "hv_reactive":
                properties = ["time_period", "desired_cpu_utilization", "algorithm", "transition_time_budget",
                              "hot_node_scale_up"]
                if set(properties) != set(config["autoscalers"][key]):
                    raise ValueError(f"Invalid or missing property in {key}")
                AscalConfig._check_fields(config["autoscalers"][key], properties,[int, float, str, int, bool])
                if config["autoscalers"][key]["time_period"] < 0:
                    raise ValueError("Time period must be possitive")
                if config["autoscalers"][key]["desired_cpu_utilization"] < 0.1:
                    raise ValueError("Desired CPU utilization must be >= 0.1")
                if config["autoscalers"][key]["algorithm"] not in ["fcma", "fcma1", "fcma2", "fcma3", "mncf"]:
                    raise ValueError("Valid algorithms are 'fcma', 'fcma1', 'fcma2', 'fcma3' or 'mncf'")

            elif key == "hv_predictive":
                properties = ["prediction_window", "prediction_percentile", "algorithm", "transition_time_budget",
                              "hot_node_scale_up"]
                if set(properties) != set(config["autoscalers"][key]):
                    raise ValueError(f"Invalid or missing property in {key}")
                AscalConfig._check_fields(config["autoscalers"][key], properties,[int, int, str, int, bool])
                if config["autoscalers"][key]["prediction_window"] < 10:
                    raise ValueError("Prediction window must be >= 10")
                if config["autoscalers"][key]["prediction_percentile"] < 0.1:
                    raise ValueError("Prediction percentile must be >= 0.1")
                if config["autoscalers"][key]["algorithm"] not in ["fcma", "fcma1", "fcma2", "fcma3", "mncf"]:
                    raise ValueError("Valid algorithms are 'fcma', 'fcma1', 'fcma2', 'fcma3' or 'mncf'")

            elif key == "h_reactive_hv_reactive":
                properties = ["h_time_period", "h_node_utilization_threshold", "desired_cpu_utilization",
                              "hv_time_period", "hv_algorithm", "hv_transition_time_budget", "hot_node_scale_up"]
                if set(properties) != set(config["autoscalers"][key]):
                    raise ValueError(f"Invalid or missing property in {key}")
                AscalConfig._check_fields(config["autoscalers"][key], properties,
                                          [int, float, float, int, str, int, bool])
                h_time_period = config["autoscalers"][key]["h_time_period"]
                hv_time_period = config["autoscalers"][key]["hv_time_period"]
                if h_time_period == 0 or hv_time_period % h_time_period > 0 or hv_time_period / h_time_period < 2:
                    raise ValueError("H time period must be a multiple 2x or higher of h time period")
                if config["autoscalers"][key]["desired_cpu_utilization"] < 0.1:
                    raise ValueError("Desired CPU utilization must be >= 0.1")
                if config["autoscalers"][key]["h_node_utilization_threshold"] < 0.1:
                    raise ValueError("Desired CPU utilization must be >= 0.1")
                if config["autoscalers"][key]["hv_algorithm"] not in ["fcma", "fcma1", "fcma2", "fcma3", "mncf"]:
                    raise ValueError("Valid algorithms are 'fcma', 'fcma1', 'fcma2', 'fcma3' or 'mncf'")

            elif key == "h_reactive_hv_predictive":
                properties = ["h_time_period", "h_node_utilization_threshold", "h_desired_cpu_utilization",
                              "hv_prediction_window", "hv_prediction_percentile", "hv_algorithm",
                              "hv_transition_time_budget", "hot_node_scale_up"]
                if set(properties) != set(config["autoscalers"][key]):
                    raise ValueError(f"Invalid or missing property in {key}")
                AscalConfig._check_fields(config["autoscalers"][key], properties,
                                          [int, float, float, int, int, str, int, bool])
                h_time_period = config["autoscalers"][key]["h_time_period"]
                hv_prediction_window = config["autoscalers"][key]["hv_prediction_window"]
                hv_prediction_percentile = config["autoscalers"][key]["hv_prediction_percentile"]
                if h_time_period == 0 or hv_prediction_window % h_time_period > 0 or \
                        hv_prediction_window / h_time_period < 2:
                    raise ValueError("H time period must be a multiple 2x or higher of HV prediction window")
                if hv_prediction_percentile < 50 or hv_prediction_percentile > 100:
                    raise ValueError("Prediction percentile must be in [50, 100]")
                if config["autoscalers"][key]["h_desired_cpu_utilization"] < 0.1:
                    raise ValueError("Desired CPU utilization must be >= 0.1")
                if config["autoscalers"][key]["h_node_utilization_threshold"] < 0.1:
                    raise ValueError("Desired CPU utilization must be >= 0.1")
                if config["autoscalers"][key]["hv_algorithm"] not in ["fcma", "fcma1", "fcma2", "fcma3", "mncf"]:
                    raise ValueError("Valid algorithms are 'fcma', 'fcma1', 'fcma2', 'fcma3' or 'mncf'")

    @staticmethod
    def _validate_timing_args(config):
        """
        Validate timing arguments in the configuration.
        :param config: Configuration.
        :raises ValueError: When a validation fails.
        """
        properties = ["node_time_to_billing", "node_creation_time", "node_removal_time",
                      "container_creation_time", "container_removal_time", "hot_node_scale_up_time"]
        if set(properties) != set(config["timing_args"]):
            raise ValueError(f"Invalid or missing property in 'timing_args'")
        AscalConfig._check_fields(config["timing_args"],
             ["node_time_to_billing", "node_creation_time", "node_removal_time", "container_creation_time",
              "container_removal_time", "hot_node_scale_up_time"], [int, int, int, int, int, int])

        # Node creation time can not be lower than hot node scale-up time
        if config["timing_args"]["node_creation_time"] < config["timing_args"]["hot_node_scale_up_time"]:
            raise ValueError("Node creation time can not be lower than hot node scale-up time")

        # Avoid a problem with priorities while firing container removal events at the same time:
        # begin removal, start grace period and end of removal
        if config["timing_args"]["node_removal_time"] == 0:
            raise ValueError("Node removal time must be possitive")

    @staticmethod
    def _validate_apps(config):
        """
        Validate applications in the configuration.
        :param config: Configuration.
        :raises ValueError: When a validation fails.
        """
        if len(config["apps"]) == 0:
            raise ValueError("At least one application is required")
        properties = ["load", "container"]
        for key, val in config["apps"].items():
            if set(properties) != set(config["apps"][key]):
                raise ValueError(f"Invalid or missing property in apps: {key}")
            load_properties = ["file", "time_interval", "repeat", "load_offset", "load_mult", "time_offset"]
            if set(load_properties) != set(val["load"]):
                raise ValueError(f"Invalid or missing property in apps: {key}: load")
            AscalConfig._check_fields(config["apps"][key]["load"],
                         ["file", "time_interval", "repeat", "load_offset", "load_mult", "time_offset"],
                         [str, int, int, int, (int, float), int])
            container_properties = ["cpu", "mem", "perf", "aggs"]
            if set(container_properties) != set(val["container"]):
                raise ValueError(f"Invalid or missing property in apps: {key}: container")
            AscalConfig._check_fields(config["apps"][key]["container"],
                         ["cpu", "mem", "perf", "aggs"],
                         [str, str, str, list])
            for v in config["apps"][key]["container"]["aggs"]:
                if not (isinstance(v, int) and v >= 1):
                    raise ValueError(f"{key}: Aggregations must be integer equal to or higher than zero")
            if len(set(config["apps"][key]["container"]["aggs"])) != len(config["apps"][key]["container"]["aggs"]):
                raise ValueError(f"{key}: Repeated aggregations")

    @staticmethod
    def validate_config(config):
        """
        Validates the given configuration.
        :param config: The configuration to validate.
        :raises ValueError: If the validation fails with a specific error message.
        """
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")

        # Check root properties in the configuration
        properties = ["autoscalers", "autoscaler", "timing_args", "apps", "simulation_time"]
        if set(config) != set(properties):
            raise ValueError("Invalid root property in configuration")
        if config["autoscaler"] not in config["autoscalers"].keys():
            raise ValueError("The selected autoscaler is invalid")
        AscalConfig._check_fields(config,["simulation_time"], [int])

        # Validate all the autoscalers
        AscalConfig._validate_autoscalers(config)

        # Validate the timing arguments
        AscalConfig._validate_timing_args(config)

        # Validate applications
        AscalConfig._validate_apps(config)


class Ascal:
    """
    This class provides methods to simulate the autoscaling of a system under a given load trace.
    """

    def __init__(self, ascal_config: AscalConfig, log=None):
        """
        Ascal constructor.
        :param ascal_config: Configuration for the Ascal problem.
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
            # Save the current allocation as an allocation change or a node change. Nodes are considered to change
            # onece it begin to be billed, even if they can not allocate contaoners yet
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

