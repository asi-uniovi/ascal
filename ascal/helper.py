"""
A bunch of auxiliary methods for atoscaling
"""
from copy import deepcopy
from math import ceil
from json import dumps
from collections import defaultdict, Counter
from fcma import Allocation, App, RequestsPerTime, Vm, ContainerClass, InstanceClass, ContainerGroup, System

from ascal.recycling import Recycling

class Vmt:
    """
    Node class for transitions, with direct access to the number of replicas of a container class.
    """
    def __init__(self, vm: Vm):
        """
        Create a node for transitions from a standard node.
        :param vm: A standard node.
        """
        self.vm = vm
        self.ic = vm.ic
        self.id = vm.id
        self.free_mem = vm.free_mem
        self.free_cores = vm.free_cores
        self.replicas = defaultdict(lambda: 0, {cg.cc: cg.replicas for cg in vm.cgs})

    def __str__(self) -> str:
        """
        String representation of the node.
        :return: The string representation.
        """
        return f"Vmt-{self.ic.name}[{self.id}]"
    
    def __repr__(self) -> str:
        """
        String representation of the node.
        :return: The string representation.
        """
        return self.__str__()

    def clear(self) -> 'Vmt':
        """
        Remove all the containers allocated in the node.
        """
        self.free_cores = self.ic.cores
        self.free_mem = self.ic.mem
        self.replicas = defaultdict(lambda: 0)
        return self

    def is_empty(self) -> bool:
        """
        Check whether the node is empty.
        :return: Return True when the node is empty.
        """
        for cc, replicas in self.replicas.items():
            if replicas > 0:
                return False
        return True

    def upgrade(self, other: 'Vmt'):
        """
        Upgrade a node to a bigger instance class in the same family.
        :param other: Other node.
        :raise ValueError: If the node upgrade is not valid.
        """
        if self.ic.family != other.ic.family or self.ic.cores > other.ic.cores or self.ic.mem > other.ic.mem:
            raise ValueError("Invalid node upgrade")
        self.id = other.id
        self.free_mem += other.ic.mem - self.ic.mem
        self.free_cores += other.ic.cores - self.ic.cores
        self.ic = other.ic

class RecyclingVmt:
    """
    Node and container's recycling class using Vmt nodes.
    """
    def __init__(self, recycling: Recycling, vm_to_vmt: dict[Vm, Vmt]):
        """
        Create a recycling object with Vmt nodes from a recycling with Vm nodes.
        :param recycling: A recycling in the Vm format.
        :param vm_to_vmt: A dictionary with Vm keys and Vmt values.
        """
        self.obsolete_nodes: list[Vmt] = [
            vm_to_vmt[vm]
            for vm in recycling.obsolete_nodes
        ]
        self.recycled_node_pairs: dict[Vmt, Vmt] = {
            vm_to_vmt[vm1]: vm_to_vmt[vm2]
            for vm1, vm2 in recycling.recycled_node_pairs.items()
        }
        self.upgraded_node_pairs: dict[Vmt, Vmt] = {
            vm_to_vmt[vm1]: vm_to_vmt[vm2]
            for vm1, vm2 in recycling.upgraded_node_pairs.items()
        }
        self.new_nodes: list[Vmt] = [
            vm_to_vmt[vm]
            for vm in recycling.new_nodes
        ]
        self.obsolete_containers: dict[Vmt, dict[ContainerClass, int]] = {
            vm_to_vmt[vm]: cc_replicas
            for vm, cc_replicas in recycling.obsolete_containers.items()
        }
        self.recycled_containers: dict[Vmt, dict[ContainerClass, int]] = {
            vm_to_vmt[vm]: cc_replicas
            for vm, cc_replicas in recycling.recycled_containers.items()
        }
        self.new_containers: dict[Vmt, dict[ContainerClass, int]] = {
            vm_to_vmt[vm]: cc_replicas
            for vm, cc_replicas in recycling.new_containers.items()
        }
        self.node_recycling_level: float = recycling.node_recycling_level
        self.container_recycling_level: float = recycling.container_recycling_level

def get_min_max_perf(alloc1: Allocation, alloc2: Allocation) ->\
        tuple[dict[App, RequestsPerTime], dict[App, RequestsPerTime]]:
    """
    Calculate the minimum and maximum application's performances of two allocations.
    :param alloc1: One allocation.
    :param alloc2: Another allocation.
    :return: The minimum and maximum application's performances.
    """

    def _get_alloc_perf(alloc: Allocation) -> dict[App, RequestsPerTime]:
        """
        Calculate the performance per application in a given allocation.
        :param alloc: Allocation.
        :return: A dictionary with performance per application.
        """
        perf = defaultdict(lambda: RequestsPerTime("0 req/s"))
        for n in alloc:
            for cgg in n.cgs:
                perf[cgg.cc.app] += cgg.replicas * cgg.cc.perf
        return perf

    perf1 = _get_alloc_perf(alloc1)
    perf2 = _get_alloc_perf(alloc2)
    zero_perf = RequestsPerTime("0 req/s")
    return (
        {app: min(perf1.get(app, zero_perf), perf2.get(app, zero_perf)) for app in perf1 | perf2},
        {app: max(perf1.get(app, zero_perf), perf2.get(app, zero_perf)) for app in perf1 | perf2},
    )

def get_min_max_load(load1: dict[App, RequestsPerTime], load2: dict[App, RequestsPerTime])\
        -> tuple[dict[App, RequestsPerTime], dict[App, RequestsPerTime]]:
    """
    Calculate the minimum and maximum application's loads between two allocations.
    :param load1: One load.
    :param load2: Another load.
    :return: The minimum and maximum application's loads.
    """

    zero_load = RequestsPerTime("0 req/s")
    return (
        {app: min(load1.get(app, zero_load), load2.get(app, zero_load)) for app in load1},
        {app: max(load1.get(app, zero_load), load2.get(app, zero_load)) for app in load1},
    )

def get_vmt_allocation_signature(alloc: list[Vmt]) -> Counter:
    """
    Get a signature to compare allocations.
    :param alloc: Allocation.
    :return: Signature.
    """
    serializable_alloc = []
    for node in alloc:
        serializable_node = {
            'ic': node.ic.name,
            'replicas': {str(c): rep for c, rep in node.replicas.items()}
        }
        serializable_alloc.append(serializable_node)
    return Counter([dumps(node, sort_keys=True) for node in serializable_alloc])

def get_app_perf_surplus(min_perf: dict[App, RequestsPerTime], alloc: list[Vmt]) -> dict[App, RequestsPerTime]:
    """
    Calculate application's performance surplus.
    :param min_perf: Minimum application's performance.
    :param alloc: Allocation.
    :return: Application's performance surplus.
    """
    app_perf_surplus = {app: -min_perf[app] for app in min_perf}
    for node in alloc:
        for cc, replicas in node.replicas.items():
            app_perf_surplus[cc.app] += replicas * cc.perf
    return app_perf_surplus

def get_app_ccs(system: System, app_aggs: dict[App, list[int]] = None) -> dict[App, list[ContainerClass]]:
    """
    Get a list of container classes for each application in the system.
    :param system: Performance parameters for pairs application and family.
    :param app_aggs: Application aggregations. Use None to consider all the application
    aggregations in system.
    :return: A dictionary with a list of containers for each application.
    """

    # Container classes for applications sorted by decreasing aggregation
    app_ccs = {}
    for app_icf, perf in system.items():
        app = app_icf[0]
        icf = app_icf[1]
        if app_aggs is not None:
            assert set(app_aggs[app]).issubset(set(perf.aggs)), f"Invalid aggregation levels for {app}"
            aggs = app_aggs[app]
        else:
            aggs = perf.aggs
        if app not in app_ccs:
            app_ccs[app] = []
        for agg in aggs:
            cc = ContainerClass(
                app=app,
                ic=None,
                fm=icf,
                cores=system[(app, icf)].cores * agg,
                mem=system[(app, icf)].mem[0],
                perf=system[(app, icf)].perf * agg,
                agg_level=agg,
                aggs=aggs
            )
            app_ccs[app].append(cc)
    return app_ccs

def get_cgs_from_workload(system: System, workloads: dict[App, RequestsPerTime],
                          app_aggs: dict[App, list[int]] = None) -> list[ContainerGroup]:
    """
    Get the container groups for a system and a given workload.
    :param system: Performance parameters for pairs application and family.
    :param workloads: Workload for each application in the system.
    :param app_aggs: Application aggregations. Use None to consider all the application
    aggregations in system.
    :return: A list of container groups
    """
    cgs = []
    # Get application's container classes sorted by decreasing aggregation levels
    app_ccs = get_app_ccs(system, app_aggs)
    for app in app_ccs:
        app_ccs[app].sort(key=lambda c: c.agg_level, reverse=True)

    for app in workloads:
        workload = workloads[app]
        agg1_perf = app_ccs[app][-1].perf / app_ccs[app][-1].agg_level
        agg1_replicas = int(ceil(workload.to('req/s') / agg1_perf.to('req/s')))
        if agg1_replicas <= app_ccs[app][-1].agg_level:
            cgs.append(ContainerGroup(app_ccs[app][-1], 1))
            continue
        for cc in app_ccs[app]:
            if cc == app_ccs[app][-1]:
                cc_replicas = min(1, int(ceil(agg1_replicas / cc.agg_level)))
            else:
                cc_replicas = int(agg1_replicas // cc.agg_level)
            if cc_replicas > 0:
                cgs.append(ContainerGroup(cc, cc_replicas))
                agg1_replicas -= cc_replicas * cc.agg_level
            if agg1_replicas == 0:
                break
    return cgs

def get_required_nodes(ic_list: list[InstanceClass], cgs: list[ContainerGroup], allocate:bool = False) -> Allocation:
    """
    Get the required nodes to allocate the containers.
    :param ic_list: List of available instance classes to allocate containers.
    :param cgs: Container groups, defined by container classes and number of replicas.
    :param allocate: In addition to get the required nodes, allocate containers on the nodes.
    :return: A list with the required nodes.
    """

    # Constant used to deal with numerical approximations
    delta = 0.000001

    required_nodes = []

    # Sort available instance classes by increasing prices
    ics = list(ic_list)
    ics.sort(key=lambda i: i.price)

    # Simulate allocation using the minimum number of nodes of the more expensive instance class
    more_expensive_ic = ics[-1]
    new_node = Vm(more_expensive_ic)
    required_nodes.append(new_node)
    for cg in deepcopy(cgs):
        while cg.replicas > 0:
            # Allocate as many replicas as possible in the current nodes
            allocated_replicas = 0
            for node in required_nodes:
                allocatable_replicas_cpu = int((node.free_cores / cg.cc.cores).magnitude + delta)
                allocatable_replicas_mem = int((node.free_mem / cg.cc.mem[0]).magnitude + delta)
                allocated_replicas = min(cg.replicas, allocatable_replicas_cpu, allocatable_replicas_mem)
                if allocated_replicas > 0:
                    node.free_cores -= allocated_replicas * cg.cc.cores
                    node.free_mem -= allocated_replicas * cg.cc.mem[0]
                    node.cgs.append(ContainerGroup(cg.cc, allocated_replicas))
                    cg.replicas -= allocated_replicas
                    if cg.replicas == 0:
                        break
            if allocated_replicas == 0:
                # Add a new node
                new_node = Vm(more_expensive_ic)
                required_nodes.append(new_node)

    # Try to reduce the cost of the nodes
    for node in list(required_nodes):
        node_price = node.ic.price
        cpu_usage = node.ic.cores - node.free_cores
        mem_usage = node.ic.mem - node.free_mem
        for ic in ics:
            if ic.cores >= cpu_usage and ic.mem >= mem_usage and ic.price < node_price:
                lowest_cost_ic = ic
                cheaper_node = Vm(lowest_cost_ic)
                cheaper_node.cgs = node.cgs
                cheaper_node.free_cores = cheaper_node.ic.cores - cpu_usage
                cheaper_node.free_mem = cheaper_node.ic.mem - mem_usage
                # Replace the node by a cheaper node
                index = required_nodes.index(node)
                required_nodes[index] = cheaper_node
                break

    # Remove all the containers of the required nodes when allocation is not required
    if not allocate:
        for node in required_nodes:
            node.clear()

    return required_nodes

def mncf_allocation(system: System, workloads: dict[App, RequestsPerTime]) -> Allocation:
    """
    Calculate an allocation using the Minimum Node Cost Fit (MNCF) allocation algorithm.
    :param system: A dictionary of tuples (application, instance class family) with performance data.
    :param workloads: The workload for each application.
    :return: The allocation, as a list of nodes with allocated containers.
    """
    cgs = get_cgs_from_workload(system, workloads)

    # Get container groups per family
    fm_cgs = defaultdict(lambda: [])
    for cg in cgs:
        fm_cgs[cg.cc.fm].append(cg)

    # Calculate allocations using a single family
    allocations = {fm: get_required_nodes(fm.ics, cgs, allocate=True) for fm, cgs in fm_cgs.items()}
    allocations_cost = {fm: sum((node.ic.price for node in allocation)) for fm, allocation in allocations.items()}

    # We are interested in the family with the lowest cost
    min_cost_fm = None
    for fm in allocations:
        if min_cost_fm is None or allocations_cost[fm] < allocations_cost[min_cost_fm]:
            min_cost_fm = fm

    return allocations[min_cost_fm]
