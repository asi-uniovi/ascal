"""
Test ASCAL based on the sequence of allocations
"""

from yaml import safe_load as yaml_safe_load
from collections import defaultdict
from pathlib import Path
import pytest
from fcma import Allocation, Vm
from ascal import AscalConfig, Ascal
from test_gen import aws_eu_west_1_c5_m5_r5
from test_gen import aws_eu_west_1_c7a_m7a_r7a

TESTS_DIR = Path(__file__).resolve().parent
PROBLEMS_DIR = TESTS_DIR / "problems"
SOLUTIONS_DIR = TESTS_DIR / "solutions"

def read_allocations(file_name) -> dict:
    """
    Read allocation stored as a YAML file.
    :param file_name: File with the allocations.
    :return: A dictionary obtained from the YAML.
    """
    with open(SOLUTIONS_DIR / file_name, "r") as f:
        allocations = yaml_safe_load(f)
    return allocations

def allocations_to_serializable_dict(allocations: list[(int, Allocation)]):
    """
    Get a serializable dictionary from a list of allocations.
    :param allocations: A list of allocations.
    :return: A serializable dictionary.
    """
    time_alloc = {}
    for current_time, alloc in allocations:
        serializable_alloc = defaultdict(lambda: {})
        for node in alloc:
            for cg in node.cgs:
                serializable_alloc[f"{node.ic.name}-{node.id}"][str(cg.cc)] = cg.replicas
        time_alloc[current_time] = dict(serializable_alloc)
    return time_alloc

# Generate list of problem file names to feed test_problem_allocation()
problem_file_names = [
    f.name for f in PROBLEMS_DIR.glob("*.yaml") if f.is_file()
]

@pytest.mark.parametrize("problem_file_name", problem_file_names)
def test_problem_allocation(problem_file_name, monkeypatch):
    """
    Test a problem given by a YAML file.
    :param problem_file_name: YAML file defining a problem
    :param monkeypatch: Pytest fixture used to change the working directory.
    """
    monkeypatch.chdir(TESTS_DIR / "test_gen")
    file_path = PROBLEMS_DIR / problem_file_name
    Vm.reset_ids()
    ic_families = [
        aws_eu_west_1_c5_m5_r5.c5_m5_r5_fm, 
        aws_eu_west_1_c7a_m7a_r7a.c7a_m7a_r7a_fm
    ]
    ascal_config = AscalConfig.get_from_config_yaml(file_path, ic_families)
    ascal_problem = Ascal(ascal_config)
    ascal_problem.run()
    calculated_allocs = allocations_to_serializable_dict(ascal_problem.allocation_changes)
    solution_allocs = read_allocations(f"{problem_file_name[:-len('.yaml')]}-alloc.yaml")
    assert calculated_allocs == solution_allocs, f"Check failed for {problem_file_name}"
