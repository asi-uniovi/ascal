# Example of autoscaling problem
from ascal import AscalConfig, Ascal
from examples import aws_eu_west_1_c5m5r5

config_file = "config.yaml"
log_file = "config.log"

# Read the problem configuration file and validate it.
# It is possible to validate any configuration with method validate_config().
ascal_config = AscalConfig.get_from_config_yaml(config_file, aws_eu_west_1_c5m5r5.c5_m5_r5_fm)

# Create the autoscaling problem
ascal_problem = Ascal(ascal_config, log=log_file)

# Run the autoscaling problem until the end. The argument of run() method is the last simulation time
ascal_problem.run()

# Write workloads, performance and cost into csv files
ascal_problem.write_workload_csv('workloads.csv')
ascal_problem.write_performance_csv('performances.csv')
ascal_problem.write_cost_csv('cost.csv')

# Get application overloads (workload/performance)
workloads = ascal_problem.get_workloads()
performances = ascal_problem.get_performances()

overloads = {app: [w/p for w, p in zip(workloads[app], performances[app])] for app in workloads}

# Plot autoscaling information
ascal_problem.plot(ascal_problem.get_workloads(), "Application Workloads", "req/s")
ascal_problem.plot(ascal_problem.get_performances(), "Application Performances", "req/s")
cluster_cost = ascal_problem.get_cluster_cost()
total_cost_str = f"total cost = {sum(cluster_cost)/3600:.3f} $"
ascal_problem.plot({total_cost_str: cluster_cost}, "Cluster Cost", "$/hour")
ascal_problem.plot(overloads, "Application Overloads")

# Useful properties
last_time = ascal_problem.last_time # Last time that can be simulated
time =  ascal_problem.time # Current simulated time in range [0, last_time]
billing_changes = ascal_problem.billing_changes # Dictionary with times and cluster state on billing changes
performance_changes = ascal_problem.performance_changes # Dictionary with times and cluster state on allocation changes
calculation_times = ascal_problem.calc_times # Calculation times to obtain new allocations

# Plot calculation times
ascal_problem.plot({'times': calculation_times}, "Calculation times", "Seconds")

