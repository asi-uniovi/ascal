"""
A bunch of auxiliary methods for atoscaling
"""

from collections import defaultdict
from fcma import Allocation, App, RequestsPerTime, Vm, ContainerClass
from recycling import Recycling

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
        Check wether the node is empty.
        :return: Return True when the node is empty.
        """
        for cc, replicas in self.replicas.items():
            if replicas > 0:
                return False
        return True

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
        {app: min(perf1.get(app, zero_perf), perf2.get(app, zero_perf)) for app in perf1},
        {app: max(perf1.get(app, zero_perf), perf2.get(app, zero_perf)) for app in perf1},
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
