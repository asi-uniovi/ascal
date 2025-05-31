"""
Test ASCAL based on the sequence of allocations
"""

import os
from yaml import safe_load as yaml_safe_load
from collections import defaultdict
from pathlib import Path
import pytest
from fcma import Allocation
from ascal import AscalConfig, Ascal
import aws_eu_west_1_c5m5r5

PROBLEMS_DIR = "../problems"
SOLUTIONS_DIR = "../solutions"

def read_allocations(file_name) -> dict:
    with open(f"{SOLUTIONS_DIR}/{file_name}-alloc.yaml", "r") as f:
        allocations = yaml_safe_load(f)
    return allocations

def calc_allocations(allocations: list[(int, Allocation)]):
    time_alloc = {}
    for current_time, alloc in allocations:
        serializable_alloc = defaultdict(lambda: {})
        for node in alloc:
            for cg in node.cgs:
                serializable_alloc[f"{node.ic.name}-{node.id}"][str(cg.cc)] = cg.replicas
        time_alloc[current_time] = dict(serializable_alloc)
    return time_alloc

# Change the current directory
os.chdir("tests/test_gen")

# Generate list of test cases from YAML files
problem_file_names = [
    f.name for f in Path(PROBLEMS_DIR).glob('*.yaml') if f.is_file()
]

@pytest.mark.parametrize("problem_file_name", problem_file_names)
def test_problem_allocation(problem_file_name):
    file_path = f"{PROBLEMS_DIR}/{problem_file_name}"
    ascal_config = AscalConfig.get_from_config_yaml(file_path, aws_eu_west_1_c5m5r5.c5_m5_r5_fm)
    ascal_problem = Ascal(ascal_config)
    ascal_problem.run()
    calculated_allocs = calc_allocations(ascal_problem.performance_changes)
    solution_allocs = read_allocations(problem_file_name)
    assert calculated_allocs == solution_allocs, f"Check failed for {problem_file_name}"
