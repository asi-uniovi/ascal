"""
Generate the solution for any YAML file in problem's directory.
The generated solutions, stored in solution's directory will be compared with those obtained later from tests.
"""

import os
import shutil
from yaml import dump as yaml_dump
from collections import defaultdict
from pathlib import Path
from fcma import Allocation
from ascal import AscalConfig, Ascal
import aws_eu_west_1_c5m5r5

PROBLEMS_DIR = "../problems"
NEW_SOLUTIONS_DIR = "../new-solutions"

def write_allocations(output_file: str, allocations: list[(int, Allocation)]):
    """
    Write the allocations as a YAML file in the output file.
    :param output_file: Output file.
    :param allocations: Allocations
    """
    time_alloc = {}
    with open(f"{NEW_SOLUTIONS_DIR}/{output_file}", "w") as f:
        for current_time, alloc in allocations:
            serializable_alloc = defaultdict(lambda: {})
            for node in alloc:
                for cg in node.cgs:
                    serializable_alloc[f"{node.ic.name}-{node.id}"][str(cg.cc)] = cg.replicas
            time_alloc[current_time] = dict(serializable_alloc)
        yaml_dump(time_alloc, f)

# Create the new solutions directory
if os.path.exists(NEW_SOLUTIONS_DIR):
    shutil.rmtree(NEW_SOLUTIONS_DIR)  # Remove the entire directory and its contents
os.makedirs(NEW_SOLUTIONS_DIR)  # Recreate the empty directory

# List of problem YAML files
problem_files = [f.name for f in Path(PROBLEMS_DIR).glob('*.yaml') if f.is_file()]

# Generate problem solutions
for problem_file in problem_files:
    file_path = f"{PROBLEMS_DIR}/{problem_file}"
    print(f"\n{'-'*50}\nSolving problem {file_path}\n{'-'*50}")
    ascal_config = AscalConfig.get_from_config_yaml(file_path, aws_eu_west_1_c5m5r5.c5_m5_r5_fm)
    problem_name = f"{NEW_SOLUTIONS_DIR}/{problem_file[:-len('.yaml')] }"
    sol_file = problem_name + "-alloc.yaml"
    log_file = problem_name + ".log" # Log files are not used for testing, but to explain allocation differences
    ascal_problem = Ascal(ascal_config, log=log_file)
    ascal_problem.run()
    write_allocations(sol_file, ascal_problem.performance_changes)

    """ Uncomment to show the plots
    
    workloads = ascal_problem.get_workloads()
    performances = ascal_problem.get_performances()
    overloads = {app: [w / p for w, p in zip(workloads[app], performances[app])] for app in workloads}
    cluster_cost = ascal_problem.get_cluster_cost()
    total_cost_str = f"total cost = {sum(cluster_cost) / 3600:.3f} $"
    node_recycling_levels, container_recycling_levels = ascal_problem.get_recycling_levels()
    ascal_problem.plot(workloads, "Application Workloads", "req/s")
    ascal_problem.plot(performances, "Application Performances", "req/s")
    ascal_problem.plot({total_cost_str: cluster_cost}, "Cluster Cost", "$/hour")
    ascal_problem.plot(overloads, "Application Overloads")
    ascal_problem.plot_bar({'nodes': node_recycling_levels, 'containers': container_recycling_levels},
                           "Recyclings", "Recycling value")
    
    """
