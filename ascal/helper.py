"""
A bunch of auxiliary methods for atoscaling
"""
from copy import copy, deepcopy
from math import ceil
from dataclasses import replace
from numpy import interp
from collections import defaultdict, Counter
from fcma import (
    Allocation, App, RequestsPerTime, Vm, ContainerClass, InstanceClass, 
    InstanceClassFamily, ContainerGroup, System, Storage, 
    CurrencyPerTime, ComputationalUnits, Storage
)
from ascal.recycling import Recycling

_DELTA = 0.000001

def copy_set_class_family(ic: InstanceClass, fm: InstanceClassFamily) -> InstanceClass:
    """
    Return a copy of an instance class with the given family. Note that using replace(ic, family=fm)
    would execute the _post_init method that woul add a new instance class to the instance class family.
    :param ic: An instance class.
    :param family: An instance class family.
    :return: A copy of the instance class with the given family."""
    result = copy(ic)
    object.__setattr__(result, "family", fm)
    return result

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
        self.fragments = defaultdict(lambda: (0, 0))

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

    def get_cc_mem(self, cc: ContainerClass, replicas_fragments) -> Storage:
        """
        Get the memory allocated to a container class in the node for the given number of replicas/fragments. 
        It assumes fragments are evenly distributed 
        among all the replicas.
        :param cc: A container/fragment class.
        :param replicas_fragments: The number of replicas/fragments to consider.
        :return: The memory allocated to the container class.
        """
        if replicas_fragments == 0:
            return Storage("0 gibibytes")

        # If the replicas are not fragmented
        if cc.id == 0:
            replicas = replicas_fragments
            assert replicas >= 0, "Invalid number of replicas"
            return replicas * cc.memv

        # If the replicas are fragmented
        replicas, _ = self.fragments[cc.id]
        fragments = replicas_fragments
        assert replicas > 0 and 0 <= fragments <= replicas * cc.aggs[-1], "Invalid number of replicas/fragments"
        avg_agg_level = fragments / replicas
        aggs = [0] + list(cc.aggs)
        mem = [0] + [mem.to("gibibytes").magnitude for mem in cc.mem]
        avg_agg_level = fragments / replicas

        return Storage(f"{interp(avg_agg_level, aggs, mem) * replicas} gibibytes")

    def get_replicas_to_free_mem(self, cc: ContainerClass, free_mem:Storage) -> tuple[int, Storage]:
        """
        Get the minimum number of replicas/fragments of a container class that must be removed in order to
        free a given amount of memory. It assumes aggregation levels and memory are sorted by 
        increasing value and fragments are evenly distribuded among all the replicas.
        :param cc: The container class.
        :param free_mem: The memory to free.
        :return: The number of replicas/fragments and the memory that would be freed.
        """
       # If the replicas are not fragmented
        if cc.id == 0:
            replicas = int(ceil((free_mem / cc.memv).magnitude - _DELTA))
            return replicas, replicas * cc.memv

        # If the replicas are fragmented
        replicas, initial_fragments = self.fragments[cc.id]
        assert 0 <= initial_fragments <= replicas * cc.aggs[-1], \
            "Invalid number of replicas/fragments"
        if initial_fragments == 0 or free_mem.magnitude <= 0:
            return 0, Storage("0 gibibytes")

        initial_cc_mem = self.get_cc_mem(cc, initial_fragments)
        if initial_cc_mem <= free_mem:
            return initial_fragments, initial_cc_mem

        aggs = [0] + list(cc.aggs)
        mem = [0] + [mem.to("gibibytes").magnitude for mem in cc.mem]
        remaining_mem = (initial_cc_mem - free_mem).to("gibibytes").magnitude
        final_fragments = int(replicas * interp(remaining_mem, mem, aggs) + _DELTA)
        removed_fragments = initial_fragments - final_fragments
        return removed_fragments, initial_cc_mem - self.get_cc_mem(cc, final_fragments)

    def get_allocatable_replicas(self, cc: ContainerClass,
                                 released_cpu:ComputationalUnits=ComputationalUnits("0 core"),
                                 released_mem:Storage=Storage("0 B")) -> int:
        """
        Get the maximum number of replicas/fragments of a container class that can be allocated in the node
        after releasing the given amount of CPU and memory.
        :param cc: The container class.
        :param released_cpu: The amount of CPU to be released.
        :param released_mem: The amount of memory to be released.
        :return: The maximum number of replicas/fragments that can be allocated.
        """
        available_cores = self.free_cores + released_cpu
        available_mem = self.free_mem + released_mem
        cpu_allocatable_replicas_fragments = int((available_cores / cc.cores).magnitude + _DELTA)

        # If the replicas are not fragmented
        if cc.id == 0:
            mem_allocatable_replicas = int((available_mem / cc.memv).magnitude + _DELTA)
            return min(cpu_allocatable_replicas_fragments, mem_allocatable_replicas)

        # If the replicas are fragmented
        replicas, init_fragments = self.fragments[cc.id]
        assert replicas > 0 and 0 <= init_fragments <= replicas * cc.aggs[-1], "Invalid number of replicas/fragments"
        aggs = [0] + list(cc.aggs)
        mem = [0] + [mem.to("gibibytes").magnitude for mem in cc.mem]
        current_mem = self.get_cc_mem(cc, init_fragments)
        available_mem = (current_mem + available_mem).to("gibibytes").magnitude
        total_fragments_mem = min(replicas * cc.aggs[-1], int(replicas * interp(available_mem, mem, aggs) + _DELTA))
        return min(cpu_allocatable_replicas_fragments, total_fragments_mem - init_fragments)

    def compact_similar_ccs_replicas(self, cc: ContainerClass) -> int:
        """
        Compact the replicas of similar container classes allocated in the node.
        :param cc: The container class of reference.
        :return: The number of replicas compacted.
        """
        ccs = [cc1 for cc1 in self.replicas if similar_ccs(cc, cc1)]
        if len(ccs) == 0:
            return 0
        if len(ccs) > 1 or ccs[0] != cc:
            compacted_replicas = 0
            for cc1 in ccs:
                compacted_replicas += self.replicas[cc1]
                del self.replicas[cc1]
            self.replicas[cc] = compacted_replicas
        return self.replicas[cc]

    def remove_similar_ccs_replicas(self, cc: ContainerClass, replicas: int):
        """
        Remove replicas of similar container classes allocated in the node.
        :param cc: The container class of reference.
        :param replicas: Number of replicas to remove.
        """
        if replicas < 1:
            return
        op_str = f'Remove containers ({self}, {cc}, {replicas})'
        self.compact_similar_ccs_replicas(cc)
        if self.replicas[cc] < replicas:
            raise ValueError(f'{op_str} -> Invalid container removal. Not enough similar containers are allocated')
        elif self.replicas[cc] == replicas:
            del self.replicas[cc]
        else:
            self.replicas[cc] -= replicas
        self.free_cores += replicas * cc.cores
        self.free_mem += replicas * cc.memv
        if (self.free_cores - self.ic.cores).magnitude > _DELTA:
            raise ValueError(f'{op_str} -> Invalid container removal. Too many cores')
        if (self.free_mem - self.ic.mem).magnitude > _DELTA:
            raise ValueError(f'{op_str} -> Invalid container removal. Too many mem')

    def alloc_similar_ccs_replicas(self, cc: ContainerClass, replicas: int):
        """
        Allocate replicas to a similar container class allocated in the node.
        :param cc: The container class of reference.
        :param replicas: Number of replicas to allocate.
        """
        if replicas < 1:
            return
        self.compact_similar_ccs_replicas(cc)
        if cc in self.replicas:
            self.replicas[cc] += replicas
        else:
            self.replicas[cc] = replicas
        self.free_cores -= cc.cores * replicas
        self.free_mem -= cc.memv * replicas
        op_str = f'Allocate containers ({self}, {cc}, {replicas})'
        if self.free_cores.magnitude < -_DELTA:
            raise ValueError(f'{op_str} -> Invalid container addition. Not enough cores are available')
        if self.free_mem.magnitude < -_DELTA:
            raise ValueError(f'{op_str} -> Invalid container removal. Not enough memory is available')

    def scale_similar_ccs_replicas(self, cc: ContainerClass, replicas: int, multiplier:float):
        """
        Scale replicas of similar container classes allocated in the node.
        :param cc: The container class of reference.
        :param replicas: Number of replicas to scale.
        :param multiplier: Multiplier to scale the replicas.
        """
        if replicas < 1:
            return
        op_str = f'Scale containers ({self}, {cc}, {replicas}, {multiplier})'
        # Remove the replicas to scale
        try:
            self.remove_similar_ccs_replicas(cc, replicas)
        except ValueError as e:
            raise ValueError(f'{op_str} -> Invalid container scale. Not enough similar containers are allocated')
        # Add the replicas after the scaling
        try:
            self.alloc_similar_ccs_replicas(cc * multiplier, replicas)
        except ValueError as e:
            raise ValueError(f'{op_str} -> Invalid container scale. Not enough resources are available')

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
        self.scaled_containers: dict[Vmt, dict[ContainerClass, ContainerClass, int]] = {
            vm_to_vmt[vm]: {(cc1_cc2[0], cc1_cc2[1]): replicas for cc1_cc2, replicas in cc1_cc2_replicas.items()}
            for vm, cc1_cc2_replicas in recycling.scaled_containers.items()
        }
        self.node_recycling_level: float = recycling.node_recycling_level
        self.container_recycling_level: float = recycling.container_recycling_level

def similar_ccs(cc1: ContainerClass, cc2: ContainerClass, ignore_perf: bool=False)->bool:
    """
    Check if the similarity of two container classes. Two container classes are similar when the are associated
    to the same instance class family and application, require the same computational resources and provide
    the same performance. 
    :param cc1: One container class.
    :param cc2: Another container class.
    :param ignore_perf: If True, ignore the performance of the container classes when checking for similarity.
    :return: True if the container classes are similar.
    """
    return cc1.fm == cc2.fm and cc1.app == cc2.app and abs(cc1.cores - cc2.cores).magnitude < _DELTA and \
        abs(cc1.memv - cc2.memv).magnitude < _DELTA and\
        (abs(cc1.perf.to("req/s") - cc2.perf.to("req/s")).magnitude < _DELTA or ignore_perf)

def compact_similar_cgs(node: Vm, cc: ContainerClass, ignore_perf: bool=False) -> ContainerGroup:
    """
    Compact the replicas of similar container groups allocated in the node.
    :param vm: The node where the container groups are allocated.
    :param cc: The container class of reference.
    :param ignore_perf: If True, ignore the performance of the container classes when checking for similarity.
    :return: The container group with the compacted replicas of similar container classes.
    """
    cgs = [cg for cg in node.cgs if similar_ccs(cg.cc, cc)]
    if len(cgs) == 0:
        same_perf_cgs = False
        if ignore_perf:
            cgs = [cg for cg in node.cgs if similar_ccs(cg.cc, cc, ignore_perf=True)]
        if len(cgs) == 0:
            return None
    else:
        same_perf_cgs = True
    if len(cgs) > 1 or cgs[0].cc != cc:
        compacted_replicas = 0
        for cg in cgs:
            compacted_replicas += cg.replicas
            node.cgs.remove(cg)
        if same_perf_cgs:
            cg = ContainerGroup(cc, compacted_replicas)
        node.cgs.append(cg)
        return cg
    else:
        return cgs[0]
    
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
    allocation_signature = []
    for node in alloc:
        container_replicas = defaultdict(int)
        for cc, replicas in node.replicas.items():
            if replicas > 0:
                container_signature = (
                    cc.app.name,
                    cc.fm.name,
                    float(cc.agg_level),
                    float(cc.cores.to("core").magnitude),
                    float(cc.memv.to("gibibytes").magnitude),
                    float(cc.perf.to("req/s").magnitude),
                )
                container_replicas[container_signature] += replicas
        node_signature = (node.ic.name, tuple(sorted(container_replicas.items())))
        allocation_signature.append(node_signature)
    return Counter(allocation_signature)

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
    for app, surplus in app_perf_surplus.items():
        if -_DELTA <= surplus.to("req/s").magnitude < _DELTA:
            app_perf_surplus[app] = RequestsPerTime("0 req/s")
    return app_perf_surplus

def get_icf_app_ccs(system: System,
                    icf_app_aggs: dict[InstanceClassFamily, dict[App, list[int]]] = None) -> \
        dict[InstanceClassFamily, dict[App, list[ContainerClass]]]:
    """
    Get a list of container classes for each instance class family and application in the system.
    :param system: Performance parameters for pairs application and family.
    :param icf_app_aggs: Aggregation levels for each instance class family and application.
    Use None to consider all the application aggregations in system.
    :return: A dictionary with a list of containers for each instance class family and application.
    """

    # Container classes for applications sorted by decreasing aggregation
    icf_app_ccs = {}
    for app_icf, perf in system.items():
        app = app_icf[0]
        icf = app_icf[1]
        if icf_app_aggs is not None:
            if icf not in icf_app_aggs:
                continue
            aggs = icf_app_aggs[icf].get(app, perf.aggs)
            assert set(aggs).issubset(set(perf.aggs)), \
                f"Invalid aggregation levels for {app} in {icf}"
        else:
            aggs = perf.aggs
        mem = tuple(perf.mem[perf.aggs.index(agg)] for agg in aggs)
        if icf not in icf_app_ccs:
            icf_app_ccs[icf] = {}
        if app not in icf_app_ccs[icf]:
            icf_app_ccs[icf][app] = []
        for agg in aggs:
            cc = ContainerClass(
                app=app,
                ic=None,
                fm=icf,
                cores=system[(app, icf)].cores * agg,
                perf=system[(app, icf)].perf * agg,
                agg_level=agg,
                mem=mem,
                aggs=aggs
            )
            icf_app_ccs[icf][app].append(cc)
    return icf_app_ccs

def get_cgs_from_workload(system: System, workloads: dict[App, RequestsPerTime],
                          icf_app_aggs: dict[InstanceClassFamily, dict[App, list[int]]] = None) -> \
                          dict[InstanceClassFamily, list[ContainerGroup]]:
    """
    Get for each instance class family the container groups providing the performance demanded
    by the given workload.
    :param system: Performance parameters for pairs application and family.
    :param workloads: Workload for each application in the system.
    :param icf_app_aggs: Instance class family and application aggregations. Use None to consider
    all the application aggregations in system.
    :return: Container groups indexed by instance class family.
    """
    cgs = {}
    # Get application's container classes sorted by decreasing aggregation levels
    icf_app_ccs = get_icf_app_ccs(system, icf_app_aggs)
    for icf in icf_app_ccs:
        for app in icf_app_ccs[icf]:
            icf_app_ccs[icf][app].sort(key=lambda c: c.agg_level, reverse=True)

    required_apps = set(workloads)
    for icf, app_ccs_by_app in icf_app_ccs.items():
        if required_apps.issubset(app_ccs_by_app):
            cgs[icf] = []

    for app in workloads:
        workload = workloads[app]
        for icf in cgs:
            app_ccs = icf_app_ccs[icf][app]
            agg1_perf = app_ccs[-1].perf / app_ccs[-1].agg_level
            agg1_replicas = int(ceil(workload.to('req/s') / agg1_perf.to('req/s')))
            if agg1_replicas <= app_ccs[-1].agg_level:
                cgs[icf].append(ContainerGroup(app_ccs[-1], 1))
                continue
            for cc in app_ccs:
                if cc == app_ccs[-1]:
                    cc_replicas = int(ceil(agg1_replicas / cc.agg_level))
                else:
                    cc_replicas = int(agg1_replicas // cc.agg_level)
                if cc_replicas > 0:
                    cgs[icf].append(ContainerGroup(cc, cc_replicas))
                    agg1_replicas -= cc_replicas * cc.agg_level
                if agg1_replicas == 0:
                    break
    return cgs

def get_required_nodes(fm: InstanceClassFamily, cgs: list[ContainerGroup], allocate:bool = False) -> Allocation:
    """
    Get the required nodes to allocate the containers.
    :param fm: The instance class family for the nodes.
    :param cgs: Container groups, defined by container classes and number of replicas.
    :param allocate: In addition to get the required nodes, allocate containers on the nodes.
    :return: A list with the required nodes.
    """

    # Constant used to deal with numerical approximations
    delta = _DELTA

    required_nodes = []

    # Sort available instance classes by increasing prices
    ics = list(fm.ics)
    ics.sort(key=lambda i: i.price)

    # Simulate allocation using the minimum number of nodes of the more expensive instance class
    more_expensive_ic = ics[-1]
    new_node = Vm(more_expensive_ic)
    new_node.ic = copy_set_class_family(more_expensive_ic, fm)
    required_nodes.append(new_node)
    for cg in deepcopy(cgs):
        while cg.replicas > 0:
            # Allocate as many replicas as possible in the current nodes
            allocated_replicas = 0
            for node in required_nodes:
                allocatable_replicas_cpu = int((node.free_cores / cg.cc.cores).magnitude + delta)
                allocatable_replicas_mem = int((node.free_mem / cg.cc.memv).magnitude + delta)
                allocated_replicas = min(cg.replicas, allocatable_replicas_cpu, allocatable_replicas_mem)
                if allocated_replicas > 0:
                    node.free_cores -= allocated_replicas * cg.cc.cores
                    node.free_mem -= allocated_replicas * cg.cc.memv
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
                cheaper_node.ic = replace(lowest_cost_ic, family=fm)
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

def get_allocation_cost(allocation: list[Vm]) -> CurrencyPerTime:
    """
    Get the cost of an allocation.
    :param allocation: An allocation, as a list of nodes with allocated containers.
    :return: The cost of the allocation.
    """
    return sum((node.ic.price for node in allocation))

def mncf_allocation(system: System, workloads: dict[App, RequestsPerTime],
                    icf_app_aggs: dict[InstanceClassFamily, dict[App, list[int]]] = None) -> Allocation:
    """
    Calculate an allocation using the Minimum Node Cost Fit (MNCF) allocation algorithm.
    :param system: A dictionary of tuples (application, instance class family) with performance data.
    :param workloads: The workload for each application.
    :param icf_app_aggs: Aggregation levels for each instance class family and application.
    :return: The allocation, as a list of nodes with allocated containers.
    """
    cgs_by_fm = get_cgs_from_workload(system, workloads, icf_app_aggs)
    if not cgs_by_fm:
        raise ValueError("No instance class family supports all applications in the workload")

    # Calculate allocations using a single family
    allocations = {
        fm: get_required_nodes(fm, family_cgs, allocate=True)
        for fm, family_cgs in cgs_by_fm.items()
    }
    allocations_cost = {fm: sum((node.ic.price for node in allocation)) for fm, allocation in allocations.items()}

    # We are interested in the family with the lowest cost
    min_cost_fm = None
    for fm in allocations:
        if min_cost_fm is None or allocations_cost[fm] < allocations_cost[min_cost_fm]:
            min_cost_fm = fm

    return allocations[min_cost_fm]
