"""
Generate the solution for any YAML file in problem's directory.
The generated solutions, stored in solution's directory will be compared  with those obtained later from tests.
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

def write_allocations(file_name: str, allocations: list[(int, Allocation)]):
    time_alloc = {}
    with open(f"{NEW_SOLUTIONS_DIR}/{file_name}-alloc.yaml", "w") as f:
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
problem_file_names = [f.name for f in Path(PROBLEMS_DIR).glob('*.yaml') if f.is_file()]

# Generate problem solutions
for problem_file_name in problem_file_names:
    file_path = f"{PROBLEMS_DIR}/{problem_file_name}"
    ascal_config = AscalConfig.get_from_config_yaml(file_path, aws_eu_west_1_c5m5r5.c5_m5_r5_fm)
    ascal_problem = Ascal(ascal_config)
    ascal_problem.run()
    write_allocations(problem_file_name, ascal_problem.performance_changes)


