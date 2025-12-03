"""
Get and check Ascal configuration from a YAML file.
"""

from yaml import safe_load
import csv
from fcma import (
    RequestsPerTime,
    InstanceClassFamily,
    ComputationalUnits,
    Storage,
    AppFamilyPerf,
    App,
    System
)
from ascal.autoscalers import AllocationSolver, AutoscalerTypes, TimedOps
from ascal.hreactive import HReactiveAutoscaler
from ascal.hvreactive import HVReactiveAutoscaler
from ascal.hvpredictive import HVPredictiveAutoscaler
from ascal.hreactivehvreactive import HReactiveHVReactiveAutoscaler
from ascal.hreactivehvpredictive import HReactiveHVPredictiveAutoscaler

class AscalConfig:
    def __init__(self, system: System = None, workload_vectors: dict[App, list[RequestsPerTime]] = None,
                 autoscaler_type: AutoscalerTypes = AutoscalerTypes.H_REACTIVE):
        """
        Ascal configuration.
        :param system: Application performance parameters for pairs application and instance class family.
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
    def get_from_config_yaml(yaml_file:str, ic_family: InstanceClassFamily) -> 'AscalConfig':
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

        # Times for operations on containers and nodes. Hot node scale-up time defaults to node creation time and
        # hot container scale time defaults to 1 time unit if not provided.
        timing_args = data["timing_args"]
        timing_args = TimedOps.TimingArgs(timing_args['node_time_to_billing'], 
                                          timing_args['node_creation_time'],
                                          timing_args['node_removal_time'], 
                                          timing_args.get('hot_node_scale_up_time', 
                                                          timing_args['node_creation_time']),
                                          timing_args['container_creation_time'],
                                          timing_args['container_removal_time'],
                                          timing_args.get('hot_container_scale_time', 1))
        # Set the autoscaler
        AscalConfig._set_autoscaler(config, data, timing_args)

        # Set application's workload
        try:
            AscalConfig._set_apps_workload(config, data)
        except:
            raise ValueError("Error reading the workload file")

        return config

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

        # Validate autoscalers
        AscalConfig._validate_autoscalers(config)

        # Validate timing arguments
        AscalConfig._validate_timing_args(config)

        # Validate applications
        AscalConfig._validate_apps(config)

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
        Set the autoscaler from data dictionary.
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
                data['autoscalers']['h_reactive']['replica_scale_down_stabilization_time'],
                data['autoscalers']['h_reactive']['node_scale_down_stabilization_time'],
                AscalConfig._get_check_aggs(data, config.apps, aggs=data["autoscalers"]['h_reactive']["aggs"]),
                timing_args
            )
        elif data['autoscaler'] == 'hv_reactive':
            algorithm = AllocationSolver.FCMA
            if data['autoscalers']['hv_reactive']['allocation'] == 'fcma1':
                algorithm = AllocationSolver.FCMA1
            elif data['autoscalers']['hv_reactive']['allocation'] == 'fcma2':
                algorithm = AllocationSolver.FCMA2
            elif data['autoscalers']['hv_reactive']['allocation'] == 'fcma3':
                algorithm = AllocationSolver.FCMA3
            elif data['autoscalers']['hv_reactive']['allocation'] == 'mncf':
                algorithm = AllocationSolver.MNCF
            config.autoscaler = HVReactiveAutoscaler(
                data['autoscalers']['hv_reactive']['time_period'],
                data['autoscalers']['hv_reactive']['desired_cpu_utilization'],
                timing_args,
                algorithm,
                data['autoscalers']['hv_reactive']['transition_time_budget'],
                data['autoscalers']['hv_reactive']['hot_node_scale_up'],
                data['autoscalers']['hv_reactive']['hot_container_scale'] 
            )
        elif data['autoscaler'] == 'hv_predictive':
            algorithm = AllocationSolver.FCMA
            if data['autoscalers']['hv_predictive']['allocation'] == 'fcma1':
                algorithm = AllocationSolver.FCMA1
            elif data['autoscalers']['hv_predictive']['allocation'] == 'fcma2':
                algorithm = AllocationSolver.FCMA2
            elif data['autoscalers']['hv_predictive']['allocation'] == 'fcma3':
                algorithm = AllocationSolver.FCMA3
            elif data['autoscalers']['hv_predictive']['allocation'] == 'mncf':
                algorithm = AllocationSolver.MNCF
            config.autoscaler = HVPredictiveAutoscaler(
                data['autoscalers']['hv_predictive']['prediction_window'],
                data['autoscalers']['hv_predictive']['prediction_percentile'],
                timing_args,
                algorithm,
                data['autoscalers']['hv_predictive']['transition_time_budget'],
                data['autoscalers']['hv_predictive']['hot_node_scale_up'],
                data['autoscalers']['hv_predictive']['hot_container_scale']
            )
        elif data['autoscaler'] == 'h_reactive_hv_reactive':
            algorithm = AllocationSolver.FCMA
            if data['autoscalers']['h_reactive_hv_reactive']['hv_allocation'] == 'fcma1':
                algorithm = AllocationSolver.FCMA1
            elif data['autoscalers']['h_reactive_hv_reactive']['hv_allocation'] == 'fcma2':
                algorithm = AllocationSolver.FCMA2
            elif data['autoscalers']['h_reactive_hv_reactive']['hv_allocation'] == 'fcma3':
                algorithm = AllocationSolver.FCMA3
            elif data['autoscalers']['h_reactive_hv_reactive']['hv_allocation'] == 'mncf':
                algorithm = AllocationSolver.MNCF
            config.autoscaler = HReactiveHVReactiveAutoscaler(
                data['autoscalers']['h_reactive_hv_reactive']['h_time_period'],
                data['autoscalers']['h_reactive_hv_reactive']['desired_cpu_utilization'],
                data['autoscalers']['h_reactive_hv_reactive']['h_node_utilization_threshold'],
                data['autoscalers']['h_reactive_hv_reactive']['h_replica_scale_down_stabilization_time'],
                data['autoscalers']['h_reactive_hv_reactive']['h_node_scale_down_stabilization_time'],
                timing_args,
                algorithm,
                data['autoscalers']['h_reactive_hv_reactive']['hv_time_period'],
                data['autoscalers']['h_reactive_hv_reactive']['hv_transition_time_budget'],
                data['autoscalers']['h_reactive_hv_reactive']['hv_hot_node_scale_up'],
                data['autoscalers']['h_reactive_hv_reactive']['hv_hot_container_scale']
            )
        elif data['autoscaler'] == 'h_reactive_hv_predictive':
            algorithm = AllocationSolver.FCMA
            if data['autoscalers']['h_reactive_hv_predictive']['hv_allocation'] == 'fcma1':
                algorithm = AllocationSolver.FCMA1
            elif data['autoscalers']['h_reactive_hv_predictive']['hv_allocation'] == 'fcma2':
                algorithm = AllocationSolver.FCMA2
            elif data['autoscalers']['h_reactive_hv_predictive']['hv_allocation'] == 'fcma3':
                algorithm = AllocationSolver.FCMA3
            elif data['autoscalers']['h_reactive_hv_predictive']['hv_allocation'] == 'mncf':
                algorithm = AllocationSolver.MNCF
            config.autoscaler = HReactiveHVPredictiveAutoscaler(
                data['autoscalers']['h_reactive_hv_predictive']['h_time_period'],
                data['autoscalers']['h_reactive_hv_predictive']['h_desired_cpu_utilization'],
                data['autoscalers']['h_reactive_hv_predictive']['h_node_utilization_threshold'],
                data['autoscalers']['h_reactive_hv_predictive']['h_replica_scale_down_stabilization_time'],
                data['autoscalers']['h_reactive_hv_predictive']['h_node_scale_down_stabilization_time'],
                timing_args,
                algorithm,
                data['autoscalers']['h_reactive_hv_predictive']['hv_prediction_window'],
                data['autoscalers']['h_reactive_hv_predictive']['hv_prediction_percentile'],
                data['autoscalers']['h_reactive_hv_predictive']['hv_transition_time_budget'],
                data['autoscalers']['h_reactive_hv_predictive']['hv_hot_node_scale_up'],
                data['autoscalers']['h_reactive_hv_predictive']['hv_hot_container_scale']
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
    def _validate_h_reactive(config):
        """
        Validate h_reactive autoscaler parameters in the configuration.
        :param config: Configuration.
        :raises ValueError: When a validation fails.
        """
        properties = ["time_period", "desired_cpu_utilization", "node_utilization_threshold",
                      "replica_scale_down_stabilization_time", "node_scale_down_stabilization_time"]
        if set(properties) | {"aggs"} != set(config["autoscalers"]["h_reactive"]):
            raise ValueError("Invalid or missing property in h_reactive")
        AscalConfig._check_fields(config["autoscalers"]["h_reactive"], properties,
                                  [int, float, float, int, int])
        if config["autoscalers"]["h_reactive"]["time_period"] < 0:
            raise ValueError("Time period must be possitive in h_reactive")
        if config["autoscalers"]["h_reactive"]["desired_cpu_utilization"] < 0.1:
            raise ValueError("Desired CPU utilization must be >= 0.1 in h_reactive")
        if config["autoscalers"]["h_reactive"]["node_utilization_threshold"] < 0.1:
            raise ValueError("Node utilization must be >= 0.1 in h_reactive")
        if config["autoscalers"]["h_reactive"]["replica_scale_down_stabilization_time"] < 0:
            raise ValueError("Replica scale-down stabilization time must be >= 0 in h_reactive")
        if config["autoscalers"]["h_reactive"]["node_scale_down_stabilization_time"] < 0:
            raise ValueError("Node scale-down stabilization time must be >= 0 in h_reactive")

    @staticmethod
    def _validate_hv_reactive(config):
        """
        Validate hv_reactive autoscaler parameters in the configuration.
        :param config: Configuration.
        :raises ValueError: When a validation fails.
        """
        mandatory = ["time_period", "desired_cpu_utilization", "transition_time_budget"]
        optional = ["allocation", "hot_node_scale_up", "hot_container_scale"]
        if "allocation" not in config["autoscalers"]["hv_reactive"]:
            config["autoscalers"]["hv_reactive"]["allocation"] = "fcma"
        if "hot_node_scale_up" not in config["autoscalers"]["hv_reactive"]:
            config["autoscalers"]["hv_reactive"]["hot_node_scale_up"] = False
        if "hot_container_scale" not in config["autoscalers"]["hv_reactive"]:
            config["autoscalers"]["hv_reactive"]["hot_container_scale"] = False
        AscalConfig._check_fields(config["autoscalers"]["hv_reactive"], mandatory ,[int, float, int])
        AscalConfig._check_fields(config["autoscalers"]["hv_reactive"], optional, [str, bool, bool])
        if set(mandatory + optional) != set(list(config["autoscalers"]["hv_reactive"].keys()) + optional):
            raise ValueError(f"Invalid property in hv_reactive need to be removed in hv_reactive")
        if config["autoscalers"]["hv_reactive"]["time_period"] < 0:
            raise ValueError("Time period must be possitive in hv_reactive")
        if config["autoscalers"]["hv_reactive"]["desired_cpu_utilization"] < 0.1:
            raise ValueError("Desired CPU utilization must be >= 0.1 in hv_reactive")
        if config["autoscalers"]["hv_reactive"]["transition_time_budget"] < 0:
            raise ValueError("Transition time budget must be >= 0 in hv_reactive")
        if config["autoscalers"]["hv_reactive"]["allocation"] not in ["fcma", "fcma1", "fcma2", "fcma3", "mncf"]:
            raise ValueError("Valid allocations in hv_reactive are 'fcma', 'fcma1', 'fcma2', 'fcma3' or 'mncf'")

    @staticmethod
    def _validate_hv_predictive(config):
        mandatory = ["prediction_window", "prediction_percentile", "transition_time_budget"]
        optional = ["allocation", "hot_node_scale_up", "hot_container_scale"]   
        if "allocation" not in config["autoscalers"]["hv_predictive"]:
            config["autoscalers"]["hv_predictive"]["allocation"] = "fcma"
        if "hot_node_scale_up" not in config["autoscalers"]["hv_predictive"]:
            config["autoscalers"]["hv_predictive"]["hot_node_scale_up"] = False
        if "hot_container_scale" not in config["autoscalers"]["hv_predictive"]:
            config["autoscalers"]["hv_predictive"]["hot_container_scale"] = False
        if set(mandatory + optional) != set(list(config["autoscalers"]["hv_predictive"].keys()) + optional):
            raise ValueError(f"Invalid property in hv_predictive need to be removed")
        AscalConfig._check_fields(config["autoscalers"]["hv_predictive"], mandatory,[int, int, int])
        AscalConfig._check_fields(config["autoscalers"]["hv_predictive"], optional, [str, bool, bool])
        if config["autoscalers"]["hv_predictive"]["prediction_window"] < 10:
            raise ValueError("Prediction window must be >= 10 in hv_predictive")
        if config["autoscalers"]["hv_predictive"]["prediction_percentile"] == 0:
            raise ValueError("Prediction percentile must be >= 0.1 in hv_predictive")
        if config["autoscalers"]["hv_predictive"]["transition_time_budget"] < 0:
            raise ValueError("Transition time budget must be >= 0 in hv_predictive")
        if config["autoscalers"]["hv_predictive"]["allocation"] not in ["fcma", "fcma1", "fcma2", "fcma3", "mncf"]:
            raise ValueError("Valid allocations in hv_predictive are 'fcma', 'fcma1', 'fcma2', 'fcma3' or 'mncf'")
        
    @staticmethod
    def _validate_h_reactive_hv_reactive(config):
        """
        Validate h_reactive_hv_reactive autoscaler parameters in the configuration.
        :param config: Configuration.
        :raises ValueError: When a validation fails.
        """
        mandatory = ["h_time_period", "h_replica_scale_down_stabilization_time",
                     "h_node_scale_down_stabilization_time", 
                     "h_node_utilization_threshold", "desired_cpu_utilization",
                     "hv_time_period", "hv_transition_time_budget"] 
        optional = ["hv_allocation", "hv_hot_node_scale_up", "hv_hot_container_scale"]
        if "hv_allocation" not in config["autoscalers"]["h_reactive_hv_reactive"]:
            config["autoscalers"]["h_reactive_hv_reactive"]["hv_allocation"] = "fcma"
        if "hv_hot_node_scale_up" not in config["autoscalers"]["h_reactive_hv_reactive"]:
            config["autoscalers"]["h_reactive_hv_reactive"]["hv_hot_node_scale_up"] = False
        if "hv_hot_container_scale" not in config["autoscalers"]["h_reactive_hv_reactive"]:
            config["autoscalers"]["h_reactive_hv_reactive"]["hv_hot_container_scale"] = False
        if set(mandatory + optional) != set(list(config["autoscalers"]["h_reactive_hv_reactive"].keys()) + optional):
            raise ValueError(f"Invalid property in h_reactive_hv_reactive need to be removed")
        AscalConfig._check_fields(config["autoscalers"]["h_reactive_hv_reactive"], mandatory,
                                  [int, int, int, float, float, int, int])
        AscalConfig._check_fields(config["autoscalers"]["h_reactive_hv_reactive"], optional, [str, bool, bool])
        h_time_period = config["autoscalers"]["h_reactive_hv_reactive"]["h_time_period"]
        hv_time_period = config["autoscalers"]["h_reactive_hv_reactive"]["hv_time_period"]
        if h_time_period <= 0:
            raise ValueError("H time period must be possitive in h_reactive_hv_reactive")    
        if hv_time_period % h_time_period > 0 or hv_time_period / h_time_period < 2:
            raise ValueError("HV time period must be a multiple 2x or higher of H time period "
                             "in h_reactive_hv_reactive")
        if config["autoscalers"]["h_reactive_hv_reactive"]["desired_cpu_utilization"] < 0.1:
            raise ValueError("Desired CPU utilization must be >= 0.1 in h_reactive_hv_reactive")
        if config["autoscalers"]["h_reactive_hv_reactive"]["h_node_utilization_threshold"] < 0.1:
            raise ValueError("Node utilization threshold must be >= 0.1 in h_reactive_hv_reactive")
        if config["autoscalers"]["h_reactive_hv_reactive"]["h_replica_scale_down_stabilization_time"] < 0:
            raise ValueError("Replica scale-down stabilization time must be >= 0 in h_reactive_hv_reactive")
        if config["autoscalers"]["h_reactive_hv_reactive"]["h_node_scale_down_stabilization_time"] < 0:
            raise ValueError("Node scale-down stabilization time must be >= 0 in h_reactive_hv_reactive")
        if config["autoscalers"]["h_reactive_hv_reactive"]["hv_transition_time_budget"] < 0:
            raise ValueError("Transition time budget must be >= 0 in h_reactive_hv_reactive")
        if config["autoscalers"]["h_reactive_hv_reactive"]["hv_allocation"] not in\
              ["fcma", "fcma1", "fcma2", "fcma3", "mncf"]:
            raise ValueError("Valid allocations in h_reactive_hv_reactive are "
                             "'fcma', 'fcma1', 'fcma2', 'fcma3' or 'mncf'")
    
    @staticmethod
    def _validate_h_reactive_hv_predictive(config):
        """
        Validate h_reactive_hv_predictive autoscaler parameters in the configuration.
        :param config: Configuration.
        :raises ValueError: When a validation fails.
        """
        mandatory = ["h_time_period", "h_replica_scale_down_stabilization_time",
                     "h_node_scale_down_stabilization_time", 
                     "h_node_utilization_threshold", "h_desired_cpu_utilization",
                     "hv_prediction_window", "hv_prediction_percentile", "hv_transition_time_budget"] 
        optional = ["hv_allocation", "hv_hot_node_scale_up", "hv_hot_container_scale"]
        if "hv_allocation" not in config["autoscalers"]["h_reactive_hv_predictive"]:
            config["autoscalers"]["h_reactive_hv_predictive"]["hv_allocation"] = "fcma"
        if "hv_hot_node_scale_up" not in config["autoscalers"]["h_reactive_hv_predictive"]:
            config["autoscalers"]["h_reactive_hv_predictive"]["hv_hot_node_scale_up"] = False
        if "hv_hot_container_scale" not in config["autoscalers"]["h_reactive_hv_predictive"]:
            config["autoscalers"]["h_reactive_hv_predictive"]["hv_hot_container_scale"] = False
        if set(mandatory + optional) != \
            set(list(config["autoscalers"]["h_reactive_hv_predictive"].keys()) + optional):
            raise ValueError(f"Invalid property in h_reactive_hv_predictive need to be removed")
        AscalConfig._check_fields(config["autoscalers"]["h_reactive_hv_predictive"], mandatory,
                                  [int, int, int, float, float, int, int, int])
        AscalConfig._check_fields(config["autoscalers"]["h_reactive_hv_predictive"], optional, [str, bool, bool])
        h_time_period = config["autoscalers"]["h_reactive_hv_predictive"]["h_time_period"]
        hv_prediction_window = config["autoscalers"]["h_reactive_hv_predictive"]["hv_prediction_window"]
        hv_prediction_percentile = config["autoscalers"]["h_reactive_hv_predictive"]["hv_prediction_percentile"]
        if h_time_period == 0:
            raise(ValueError("H time period must be possitive in h_reactive_hv_predictive"))
        if hv_prediction_window % h_time_period > 0 or hv_prediction_window / h_time_period < 2:
            raise ValueError("HV prediction window must be a multiple 2x or higher of H time period")
        if hv_prediction_percentile < 50 or hv_prediction_percentile > 100:
            raise ValueError("Prediction percentile must be in [50, 100]")
        if config["autoscalers"]["h_reactive_hv_predictive"]["h_desired_cpu_utilization"] < 0.1:
            raise ValueError("Desired CPU utilization must be >= 0.1 in h_reactive_hv_predictive")
        if config["autoscalers"]["h_reactive_hv_predictive"]["h_node_utilization_threshold"] < 0.1:
            raise ValueError("Node utilization threshold must be >= 0.1 in h_reactive_hv_predictive")
        if config["autoscalers"]["h_reactive_hv_predictive"]["h_replica_scale_down_stabilization_time"] < 0:
            raise ValueError("Replica scale-down stabilization time must be >= 0 in h_reactive_hv_predictive")
        if config["autoscalers"]["h_reactive_hv_predictive"]["h_node_scale_down_stabilization_time"] < 0:
            raise ValueError("Node scale-down stabilization time must be >= 0 in h_reactive_hv_predictive")
        if config["autoscalers"]["h_reactive_hv_predictive"]["hv_transition_time_budget"] < 0:
            raise ValueError("Transition time budget must be >= 0 in h_reactive_hv_predictive")
        if config["autoscalers"]["h_reactive_hv_predictive"]["hv_allocation"] not in\
              ["fcma", "fcma1", "fcma2", "fcma3", "mncf"]:
            raise ValueError("Valid allocations in h_reactive_hv_predictive are "
                             "'fcma', 'fcma1', 'fcma2', 'fcma3' or 'mncf'")

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
                AscalConfig._validate_h_reactive(config)
            elif key == "hv_reactive":
                AscalConfig._validate_hv_reactive(config)
            elif key == "hv_predictive":
                AscalConfig._validate_hv_predictive(config)
            elif key == "h_reactive_hv_reactive":
                AscalConfig._validate_h_reactive_hv_reactive(config)
            elif key == "h_reactive_hv_predictive":
                AscalConfig._validate_h_reactive_hv_predictive(config)
                
    @staticmethod
    def _validate_timing_args(config):
        """
        Validate timing arguments in the configuration.
        :param config: Configuration.
        :raises ValueError: When a validation fails.
        """
        mandatory = ["node_time_to_billing", "node_creation_time", "node_removal_time",
                     "container_creation_time", "container_removal_time"]
        optional = ["hot_node_scale_up_time", "hot_container_scale_time"]
        if "hot_node_scale_up_time" not in config["timing_args"]:
            config["timing_args"]["hot_node_scale_up_time"] = config["timing_args"]["node_creation_time"]
        if "hot_container_scale_time" not in config["timing_args"]:
            config["timing_args"]["hot_container_scale_time"] = 1 
        if set(mandatory + optional) != set(list(config["timing_args"].keys()) + optional):
            raise ValueError(f"Invalid property in 'timing_args need to be removed")
        AscalConfig._check_fields(config["timing_args"], mandatory + optional,
              [int, int, int, int, int, int, int])
        if config["timing_args"]["node_time_to_billing"] < 0:
            raise ValueError("Node time to billing must be possitive")
        check_non_negative = mandatory + optional
        check_non_negative.remove("node_time_to_billing")
        for key in check_non_negative:
            if config["timing_args"][key] <= 0:
                raise ValueError(f"Timing argument {key} must be possitive")

        # Node creation time can not be lower than hot node scale-up time
        if config["timing_args"]["node_creation_time"] < config["timing_args"]["hot_node_scale_up_time"]:
            raise ValueError("Node creation time can not be lower than hot node scale-up time")

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

