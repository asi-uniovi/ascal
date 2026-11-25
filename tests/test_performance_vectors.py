from types import SimpleNamespace

from fcma import (
    App,
    AppFamilyPerf,
    ContainerGroup,
    ComputationalUnits,
    ContainerClass,
    CurrencyPerTime,
    InstanceClass,
    InstanceClassFamily,
    RequestsPerTime,
    Storage,
    Vm,
)

from ascal.helper import Vmt, get_cgs_from_workload
from ascal.ascal import Ascal


def test_get_cc_perf_interpolates_fragment_performance():
    family = InstanceClassFamily("test-family")
    instance_class = InstanceClass(
        "test-instance",
        CurrencyPerTime("1 usd/hour"),
        ComputationalUnits("8 cores"),
        Storage("8 GiB"),
        family,
    )
    node = Vmt(Vm(instance_class))
    container_class = ContainerClass(
        app=App("test-app"),
        ic=None,
        fm=family,
        cores=ComputationalUnits("1 core"),
        mem=(Storage("1 GiB"), Storage("1.5 GiB"), Storage("2 GiB")),
        perf=(
            RequestsPerTime("10 req/s"),
            RequestsPerTime("15 req/s"),
            RequestsPerTime("40 req/s"),
        ),
        aggs=(1, 2, 4),
        id=1,
    )
    node.fragments[container_class.id] = (2, 3)

    assert node.get_cc_perf(container_class, 0).to("req/s").magnitude == 0
    assert node.get_cc_perf(container_class, 3).to("req/s").magnitude == 25
    assert node.get_cc_perf(container_class, 4).to("req/s").magnitude == 30


def test_mncf_container_groups_support_nonproportional_performance():
    app = App("test-app")
    family = InstanceClassFamily("test-family")
    system = {
        (app, family): AppFamilyPerf(
            cores=ComputationalUnits("1 core"),
            mem=(Storage("1 GiB"), Storage("1.5 GiB"), Storage("2 GiB")),
            perf=(
                RequestsPerTime("10 req/s"),
                RequestsPerTime("15 req/s"),
                RequestsPerTime("30 req/s"),
            ),
            aggs=(1, 2, 4),
        )
    }

    groups = get_cgs_from_workload(
        system,
        {app: RequestsPerTime("35 req/s")},
    )[family]

    assert [(group.cc.agg_level, group.replicas) for group in groups] == [(4, 1), (1, 1)]
    assert sum(group.cc.perfv * group.replicas for group in groups).to("req/s").magnitude >= 35


def test_cpu_performance_rate_uses_the_current_nonlinear_allocation():
    app = App("test-app")
    family = InstanceClassFamily("test-family")
    app_family_perf = AppFamilyPerf(
        cores=ComputationalUnits("1 core"),
        mem=(Storage("1 GiB"), Storage("1.5 GiB"), Storage("2 GiB")),
        perf=(
            RequestsPerTime("10 req/s"),
            RequestsPerTime("15 req/s"),
            RequestsPerTime("30 req/s"),
        ),
        aggs=(1, 2, 4),
    )
    instance_class = InstanceClass(
        "test-instance",
        CurrencyPerTime("1 usd/hour"),
        ComputationalUnits("8 cores"),
        Storage("8 GiB"),
        family,
    )
    container_class = ContainerClass(
        app=app,
        ic=instance_class,
        fm=family,
        cores=ComputationalUnits("2 cores"),
        mem=app_family_perf.mem,
        perf=app_family_perf.perf,
        aggs=app_family_perf.aggs,
        agg_level=2,
    )
    node = Vm(instance_class)
    node.cgs.append(ContainerGroup(container_class, 2))
    ascal = object.__new__(Ascal)
    ascal._autoscaler = SimpleNamespace(
        system={(app, family): app_family_perf},
        allocation=[node],
    )

    assert ascal._get_perf_per_core()[app] == 15 / 2
