"""
Example of autoscaling problem
"""

from numpy import percentile as percentile
from ascal import AscalConfig, Ascal
from examples import aws_eu_west_1_c5m5r5

import os
script_dir = os.path.dirname(os.path.abspath(__file__))

config_file = "config.yaml"
log_file = "config.log"

# Check config.yaml existence
if not os.path.exists(config_file):
    raise FileNotFoundError(f"{config_file} not found in {os.getcwd()}")

# Read the problem configuration file and validate it.
# It is possible to validate any configuration with method validate_config().
ascal_config = AscalConfig.get_from_config_yaml(config_file, aws_eu_west_1_c5m5r5.c5_m5_r5_fm)

# Create the autoscaling problem
ascal_problem = Ascal(ascal_config, log=log_file)

# Last time that can be simulated (last time in the trace)
# Simulating time unit is seconds, so 3600 means 1 hour. Time starting from 0
last_time = ascal_problem.last_time 
print(f'Time range of the simulation: 0 - {last_time} seconds')

# Run the autoscaling problem until the end. The argument of run() method is the last simulation time in seconds
# Simulating time unit is seconds, so 3600 means 1 hour. Time starting from 0
ascal_problem.run()

# Write workloads, performance and cost into csv files
ascal_problem.write_workload_csv('workloads.csv')
ascal_problem.write_performance_csv('performances.csv')
ascal_problem.write_cost_csv('cost.csv')

# Write allocations in a YAML file
ascal_problem.write_allocations('allocations.yaml')

# Get application overloads as workload/performance
workloads = ascal_problem.get_workloads()
performances = ascal_problem.get_performances()
overloads = {app: [w/p for w, p in zip(workloads[app], performances[app])] for app in workloads}

# Get queue waiting times relative to service times, assuming each container is a server in a heterogenous D/D/n queue
relative_queue_waiting_times = ascal_problem.get_relative_queue_waiting_times()
percentiles99 = {
    app_name: percentile(waiting_times, 99)
    for app_name, waiting_times in relative_queue_waiting_times.items()
}
for app_name in dict(relative_queue_waiting_times):
    relative_queue_waiting_times[f"{app_name} 99% percentile = {percentiles99[app_name]:.3f}"] =\
        relative_queue_waiting_times.pop(app_name)

# Plot autoscaling information
ascal_problem.plot(ascal_problem.get_workloads(), "Application Workloads", "req/s")
ascal_problem.plot(ascal_problem.get_performances(), "Application Performances", "req/s")
cluster_cost = ascal_problem.get_cluster_cost()
total_cost_str = f"total cost = {sum(cluster_cost)/3600:.3f} $"
ascal_problem.plot({total_cost_str: cluster_cost}, "Cluster Cost", "$/hour")
ascal_problem.plot(overloads, "Application Overloads")
ascal_problem.plot(relative_queue_waiting_times, "Relative queue waiting times")

# Useful properties
last_time = ascal_problem.last_time # Last time that can be simulated
current_time =  ascal_problem.time # Current simulated time in range [0, last_time]
billing_changes = ascal_problem.billing_changes # Dictionary with times and cluster state on billing changes
performance_changes = ascal_problem.performance_changes # Dictionary with times and cluster state on allocation changes
calculation_times = ascal_problem.calc_times # Calculation times to obtain new allocations

# Recycling levels for Horizontal/Vertical autoscalers
node_recycling_levels, container_recycling_levels = ascal_problem.get_recycling_levels()

# Plot times to calculate transitions
transition_times = calculation_times["transition_times"]
ascal_problem.plot({'_nolegend_': transition_times}, "Transition times", "Seconds")

# Plot recyclings
ascal_problem.plot_bar({'nodes': node_recycling_levels, 'containers': container_recycling_levels},
                   "Recyclings", "Recycling value")