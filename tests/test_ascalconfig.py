import pytest
from fcma import AppFamilyPerf, ComputationalUnits

from ascal.ascalconfig import AscalConfig
from ascal.autoscalers import AllocationSolver


def test_container_resource_values_are_parsed_per_aggregation():
    container = {
        "aggs": [1, 2, 4],
        "mem": "1 GiB, 1.5 GiB, 2 GiB",
        "perf": "10 req/s, 15 req/s, 20 req/s",
    }

    memories = AscalConfig._get_container_memories(container)
    performances = AscalConfig._get_container_performances(container)
    app_family = AppFamilyPerf(
        cores=ComputationalUnits("1 core"),
        mem=memories,
        perf=performances,
        aggs=tuple(container["aggs"]),
    )

    assert [memory.to("GiB").magnitude for memory in memories] == [1, 1.5, 2]
    assert [performance.to("req/s").magnitude for performance in performances] == [10, 15, 20]
    assert [performance.to("req/s").magnitude for performance in app_family.perf] == [10, 15, 20]


def test_single_performance_value_expands_linearly():
    performances = AscalConfig._get_container_performances({
        "aggs": [1, 2, 4],
        "perf": "10 req/s",
    })

    assert [performance.to("req/s").magnitude for performance in performances] == [10, 20, 40]


@pytest.mark.parametrize("perf", [
    "10 req/s, 20 req/s",
    "10 req/s, 15 req/s, 12 req/s",
    "10 req/s, 0 req/s, 20 req/s",
])
def test_invalid_performance_vectors_are_rejected(perf):
    with pytest.raises(ValueError):
        AscalConfig._get_container_performances({
            "aggs": [1, 2, 4],
            "perf": perf,
        })


def test_linear_memory_constraint_depends_on_selected_allocators():
    container = {
        "aggs": [1, 2, 4],
        "mem": "1 GiB, 1.5 GiB, 2 GiB",
        "perf": "10 req/s, 15 req/s, 20 req/s",
    }

    with pytest.raises(ValueError, match="Linear memory"):
        AscalConfig._validate_linear_memory_container_constraint(
            {**container, "mem": "1 GiB, 2.1 GiB, 3.8 GiB"},
            {AllocationSolver.MNCF},
        )
    AscalConfig._validate_linear_memory_container_constraint(container, set())


def test_app_validation_accepts_nonproportional_performance():
    config = {
        "autoscaler": "hv_reactive",
        "autoscalers": {"hv_reactive": {"algorithm": "fcma"}},
        "apps": {
            "app": {
                "load": {
                    "file": "trace.csv",
                    "time_interval": 1,
                    "repeat": 1,
                    "load_offset": 0,
                    "load_mult": 1,
                    "time_offset": 0,
                },
                "container": {
                    "cpu": "1 core",
                    "mem": "1 GiB, 1.5 GiB",
                    "perf": "10 req/s, 20 req/s",
                    "aggs": [1, 2],
                },
            }
        },
    }

    AscalConfig._validate_apps(config)
    config["apps"]["app"]["container"]["perf"] = "10 req/s, 15 req/s"
    AscalConfig._validate_apps(config)


def test_horizontal_autoscaler_uses_mncf_constraints():
    config = {
        "autoscaler": "h_reactive",
        "autoscalers": {"h_reactive": {}},
    }

    assert AscalConfig._get_selected_allocation_solvers(config) == {AllocationSolver.MNCF}

    mixed_config = {
        "autoscaler": "h_reactive_hv_reactive",
        "autoscalers": {
            "h_reactive_hv_reactive": {"hv_algorithm": "fcma2"},
        },
    }
    assert AscalConfig._get_selected_allocation_solvers(mixed_config) == {
        AllocationSolver.MNCF,
        AllocationSolver.FCMA2,
    }
