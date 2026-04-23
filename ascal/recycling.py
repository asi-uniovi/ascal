""" Recycling module for calculating node and container recycling levels
    between two allocations using different algorithms.
    It supports various solvers including Hungarian, greedy, and ILP methods.
    The recycling level is calculated based on the number of containers and nodes
    that can be recycled between the initial and final allocations.
    The module also supports partitioning the problem to reduce complexity.
    However, after experimentation, greedy, partition and ILP implementations are not used, 
    since they do not reduce calculation times versus the Hungarian solver.
"""

from collections import defaultdict
from collections.abc import Callable
from typing import TypeAlias
import numpy as np
from scipy.optimize import linear_sum_assignment
from pulp import (
    LpVariable,
    lpSum,
    LpProblem,
    LpMaximize,
    LpBinary,
    COIN_CMD,
    PULP_CBC_CMD,
    PulpSolverError,
    LpStatusOptimal,
)
from pulp import value as pulp_value
from fcma import Allocation, Vm, ContainerClass, ContainerGroup, InstanceClass
from fcma.helper import _solve_cbc_patched

COIN_CMD.solve_CBC = _solve_cbc_patched
RecyclingSolverType: TypeAlias = Callable[
    [list[Vm], list[Vm], bool, int],
    tuple[list[tuple[Vm, Vm]], float]
]
MAX_ILP_TIME_SECS = 100
MAX_ILP_ERROR = 0.02

# The node recycling level between one node and its scale-up version is multiplied by this factor
# to favour recycling between nodes with identical instance classes
UP_SCALE_PENALTY_FACTOR = 0.000001

class Recycling:
    """
    Class to calculate the recyclings between two consecutive allocations.
    """

    # Invalid recycling value
    INVALID_RECYCLING = -1

    @staticmethod
    def _group_nodes_by_ic(alloc: Allocation) -> dict:
        """
        Group nodes in an allocation by their instance class (IC).
        :param alloc: List of VM nodes.
        :return: Dictionary mapping instance class to list of nodes with that class.
        """
        ic_nodes = {}
        for node in alloc:
            ic_nodes.setdefault(node.ic, []).append(node)
        return ic_nodes

    @staticmethod
    def _get_scaled_pairs(initial_containers: list[tuple[ContainerClass, int]],
                          final_containers: list[tuple[ContainerClass, int]], scale_up=False) \
                            -> list[tuple[ContainerClass, ContainerClass, int]]:
        """
        Get scaled-up or scaled-down container pairs between the lists of initial and final containers, 
        removing the paired replicas from the lists.
        :param initial_containers: Initial containers.
        :param final_containers: Final containers.
        :param scale_up: True for containers scale-up, False for containers scale-down
        :return: A list of tuples (cc1, cc2, replicas) with the initial container class, final container class
        and the scaled number of replicas.
        """
        # Sort the containers by increasing/decreasing number of cores
        initial_containers.sort(key = lambda x: x[0].cores.magnitude, reverse=scale_up)
        final_containers.sort(key = lambda x: x[0].cores.magnitude, reverse=scale_up)
        i = 0
        j = 0
        scaled_up_pairs = []
        while i < len(initial_containers) and j < len(final_containers):
            if scale_up:
                scale_condition = initial_containers[i][0].cores.magnitude < final_containers[j][0].cores.magnitude  
            else:
                scale_condition = initial_containers[i][0].cores.magnitude > final_containers[j][0].cores.magnitude
            if scale_condition:
                scaled_replicas = min(initial_containers[i][1], final_containers[j][1])
                scaled_up_pairs.append((initial_containers[i][0], final_containers[j][0], scaled_replicas))
                initial_containers[i] = (initial_containers[i][0], initial_containers[i][1] - scaled_replicas)
                final_containers[j] = (final_containers[j][0], final_containers[j][1] - scaled_replicas)
                if initial_containers[i][1] == 0:
                    i += 1                    
                if final_containers[j][1] == 0:
                    j += 1
            else:
                i += 1
        # Remove containers with no replicas
        initial_containers[:] = [c for c in initial_containers if c[1] > 0]
        final_containers[:] = [c for c in final_containers if c[1] > 0]

        return scaled_up_pairs

    def __init__(self, initial_alloc: Allocation, final_alloc: Allocation, 
                 hot_node_scale_up: bool = False, hot_container_scale: bool = False):
        """
        Constructor for the Recycling class. Computes node and container recycling between two allocations
        using the same instance class family.
        :param initial_alloc: Initial allocation.
        :param final_alloc: Final allocation.
        :param hot_node_scale_up: Set when hot node scaling-up is possible.
        :param hot_container_scale: Set when hot container scaling is possible.
        """

        # Check that we are working with a single instance class family
        assert len(set(node.ic.family for node in initial_alloc + final_alloc)) == 1, "Invalid recycling problem"

        self._hot_node_scale_up = hot_node_scale_up
        self._hot_container_scale = hot_container_scale
        self._initialize_fields()

        if self._hot_node_scale_up:
            # Calculate a recycling that includes all the nodes
            node_recyclings = self.calculate_node_recycling(initial_alloc, final_alloc)
            self.obsolete_nodes.extend(node_recyclings["obsolete"])
            self.new_nodes.extend(node_recyclings["new"])
            for initial_node, final_node in node_recyclings["recycled_pairs"]:
                if initial_node.ic != final_node.ic:
                    self.upgraded_node_pairs[initial_node] = final_node
                else:
                    self.recycled_node_pairs[initial_node] = final_node
        else:
            # Calculate one recycling by instance class
            initial_ic_nodes = Recycling._group_nodes_by_ic(initial_alloc)
            final_ic_nodes = Recycling._group_nodes_by_ic(final_alloc)
            initial_recyclable, final_recyclable = \
                self._classify_nodes_by_recycling(initial_ic_nodes, final_ic_nodes)
            for ic in initial_recyclable:
                node_recyclings = self.calculate_node_recycling(initial_recyclable[ic], final_recyclable[ic])
                self.obsolete_nodes.extend(node_recyclings["obsolete"])
                self.new_nodes.extend(node_recyclings["new"])
                for initial_node, final_node in node_recyclings["recycled_pairs"]:
                    self.recycled_node_pairs[initial_node] = final_node

        self._calculate_container_recycling()
        self._calculate_recycling_levels(initial_alloc)
        self._assign_final_node_ids()

    def _valid_node_recycling(self, initial_node: Vm, final_node: Vm) -> bool:
        """
        Return whether the initial node may be recycled to the final node.
        :param initial_node: Initial node.
        :param final_node: Final node.
        :return: True when it is a valid recycling.
        """
        if initial_node.ic.family != final_node.ic.family:
            return False
        if self._hot_node_scale_up:
            if initial_node.ic.cores > final_node.ic.cores or initial_node.ic.mem > final_node.ic.mem:
                return False
        else:
            if initial_node.ic != final_node.ic:
                return False
        return True

    def _get_container_pairs(self, initial_containers: list[ContainerGroup], final_containers: list[ContainerGroup]) \
                                -> list[tuple[ContainerClass, ContainerClass, int]]:
        """
        Get the pairs of containers that can be recycled between the initial and final lists of containers.
        :param initial_containers: Initial containers.
        :param final_containers: Final containers.
        :param hot_container_scale: Set if hot container scaling is enabled.
        return: A list of tuples (container class, container class, replicas). The first element of the tuple is
        the container class in the initial node, the second element is the container class in the final node and
        the third the number of replicas. If the first element is None, the second refers to new containers. 
        If the second element is None, the first refers to obsolete containers. If the number of cores of both 
        container classes is the same, it is a direct recycling, otherwise it is a scaled-up or scaled-down recycling.
        """
        container_pairs = []

        # Transform the lists of container groups into dictionaries
        initial_containers = {cg.cc: cg.replicas for cg in initial_containers} 
        final_containers = {cg.cc: cg.replicas for cg in final_containers} 

        # Find replicas with common container classes in the initial and final nodes
        initial_ccs = list(initial_containers.keys())
        for cc in initial_ccs:
            if cc in final_containers:
                common_replicas = min(initial_containers[cc], final_containers[cc])
                container_pairs.append((cc, cc, common_replicas))
                initial_containers[cc] -= common_replicas
                if initial_containers[cc] == 0:
                    del initial_containers[cc]
                final_containers[cc] -= common_replicas
                if final_containers[cc] == 0:
                    del final_containers[cc]

        # If hot scaling of containers is disabled
        if not self._hot_container_scale:
            # Append obsolete replicas
            for cc, replicas in initial_containers.items():
                container_pairs.append((cc, None, replicas))
            # Append new replicas
            for cc, replicas in final_containers.items():
                container_pairs.append((None, cc, replicas))
        # If hot scaling of containers is enabled, find scaled-up and scaled-down recyclings
        else:
            # Dictionaries where keys are apps and values are lists of tuples (container class, replicas) 
            initial_containers_app = defaultdict(list)
            final_containers_app = defaultdict(list)
            for cc, replicas in initial_containers.items():
                initial_containers_app[cc.app].append((cc, replicas))
            for cc, replicas in final_containers.items():
                final_containers_app[cc.app].append((cc, replicas))
            # Extend the container pairs with scale-up, scale-down and obsolete ones
            for app in initial_containers_app:
                if app in final_containers_app:
                    # Get the scale-up pairs, removing the paired items from the lists
                    # Note that scale-up recycling is priorized
                    pairs = Recycling._get_scaled_pairs(initial_containers_app[app], final_containers_app[app],
                                                        scale_up=True)
                    container_pairs.extend(pairs)
                    # Append scaled-down pairs, removing the paired items from the lists
                    pairs = Recycling._get_scaled_pairs(initial_containers_app[app], final_containers_app[app],
                                                        scale_up=False)
                    container_pairs.extend(pairs)
                # Append pairs with obsolete replicas
                for cc, replicas in initial_containers_app[app]:
                    container_pairs.append((cc, None, replicas))
            # Append pairs with new replicas
            for app in final_containers_app:
                 for cc, replicas in final_containers_app[app]:
                     container_pairs.append((None, cc, replicas))

        return container_pairs

    def node_pair_recycling_level(self, initial_node: Vm, final_node: Vm) -> float:
        """
        Calculate the recycling level between two nodes. The recycling level is 1.0 when both nodes come
        from the same instance class and allocate the same containers.
        :param initial_node: Initial node.
        :param final_node: Final node.
        :return: The recycling level. It is zero when the nodes can not be recycled.
        """
        if not self._valid_node_recycling(initial_node, final_node):
            return 0

        # Recycling between a node and its scaled-up version has a penalty
        node_scale_up_penalty = 0.5 * (initial_node.ic.cores / final_node.ic.cores + \
                                       initial_node.ic.mem / final_node.ic.mem).magnitude

        recycling_level = 0
        ic_cores = (initial_node.ic.cores - initial_node.free_cores).magnitude
        ic_mem = (initial_node.ic.mem - initial_node.free_mem).magnitude
        container_pairs = self._get_container_pairs(initial_node.cgs, final_node.cgs)
        for cc1, cc2, replicas in container_pairs:
            if cc1 is not None and cc2 is not None:
                cc1_cores = cc1.cores.magnitude
                cc1_mem = cc1.memv.magnitude
                cc2_cores = cc2.cores.magnitude
                container_scale_penalty = min(cc1_cores, cc2_cores) / cc1_cores 
                recycling_level += (0.5 * node_scale_up_penalty * replicas * container_scale_penalty *
                                    (cc1_cores / ic_cores + cc1_mem / ic_mem))
        return recycling_level

    def hungarian_solver(self, initial_nodes: list[Vm], final_nodes: list[Vm], partitions: int = 1) \
                            -> tuple[list[tuple[Vm, Vm]], float]:
        """
        Calculate the recyclings between the list of initial nodes and the list of final nodes using the
        Hungarian algorithm.
        :param initial_nodes: Initial nodes.
        :param final_nodes: Final nodes.
        :param partitions: Number of problem partitions.
        :return: A list of pairs (initial node, final node) that recycle each other, as well as the recycling level.
        """
        if type(partitions) is not int or partitions < 1:
            raise ValueError("Invalid partition value")
        if partitions > 1:
            return self._partition_solver(initial_nodes, final_nodes, self.hungarian_solver, partitions)
        benefit_matrix = [
            [
                self.node_pair_recycling_level(initial_node, final_node)
                for final_node in final_nodes
            ]
            for initial_node in initial_nodes
        ]
        np_benefit_matrix = np.array(benefit_matrix)

        # Convert the benefit matrix to a cost matrix
        max_val = np.max(np_benefit_matrix)
        np_cost_matrix = max_val - np_benefit_matrix

        # Pad the cost matrix to square with zeroes
        n_rows, n_cols = np_cost_matrix.shape
        size = max(n_rows, n_cols)
        np_padded_cost_matrix = np.full((size, size), 0.0)
        np_padded_cost_matrix[:n_rows, :n_cols] = np_cost_matrix

        # Apply Hungarian algorithm
        initial_node_indexes, final_node_indexes = linear_sum_assignment(np_padded_cost_matrix)

        # Filter indexes
        valid_index_pairs = [
            (i, j)
            for i, j in zip(initial_node_indexes, final_node_indexes)
            if i < n_rows and j < n_cols and \
               self._valid_node_recycling(initial_nodes[i], final_nodes[j])
        ]

        # Get the recycled node pairs
        node_recyclings = [
            (initial_nodes[i], final_nodes[j])
            for i, j in valid_index_pairs
        ]

        # Get the sum of node recyclings for the optimal solution
        total_node_recycling = sum([benefit_matrix[i][j] for i, j in valid_index_pairs])

        return node_recyclings, total_node_recycling / len(initial_nodes)

    def greedy_solver(self, initial_nodes: list[Vm], final_nodes: list[Vm], dummy: int = 1) \
                        -> tuple[list[tuple[Vm, Vm]], float]:
        """
        Calculate the recyclings between the list of initial nodes and the list of final nodes using a
        greedy algorithm. In each  step, it selects the node pair with the highest recycling value, removes
        the nodes from the list and repeats until no more node pairs can be recycled.
        :param initial_nodes: Initial nodes.
        :param final_nodes: Final nodes.
        :return: A list of pairs (initial node, final node) that recycle each other, as well as the recycling level.
        """
        # Calculate recycling levels for each node pair
        node_pair_recycling_levels = [
            (initial_node, final_node,
            self.node_pair_recycling_level(initial_node, final_node))
            for initial_node in initial_nodes
            for final_node in final_nodes
            if self._valid_node_recycling(initial_node, final_node)
        ]

        # Sort the pairs by decreasing recycling level
        node_pair_recycling_levels.sort(key=lambda pair: pair[2], reverse=True)
        recycled_initial_nodes = set()
        recycled_final_nodes = set()
        pair_index = 0

        # Remove node pairs including an initial node or final node of a previous node pair (with higher
        # recycling level)
        min_length = min(len(initial_nodes), len(final_nodes))
        while len(node_pair_recycling_levels) > min_length:
            initial_node, final_node, _ =  node_pair_recycling_levels[pair_index]
            if initial_node in recycled_initial_nodes or final_node in recycled_final_nodes:
                node_pair_recycling_levels.pop(pair_index)
            else:
                recycled_initial_nodes.add(initial_node)
                recycled_final_nodes.add(final_node)
                pair_index += 1

        # Get the recycled node pairs
        node_recyclings = [(r[0], r[1]) for r in node_pair_recycling_levels]

        # Get the sum of node recyclings for the solution
        total_node_recycling = sum(node_pair_recycling_levels[i][2] for i in range(len(node_pair_recycling_levels)))

        return node_recyclings, total_node_recycling / len(initial_nodes)

    def ilp_solver(self, initial_nodes: list[Vm], final_nodes: list[Vm], partitions: int = 1) \
                    -> tuple[list[tuple[Vm, Vm]], float]:
        """
        Calculate the recyclings between the list of initial nodes and the list of final nodes using an
        Integer Linear Programming (ILP) solver.
        :param initial_nodes: Initial nodes.
        :param final_nodes: Final nodes.
        :param partitions: Number of problem partitions.
        :return: A list of pairs (initial node, final node) that recycle each other, as well as the recycling level.
        """
        if type(partitions) is not int or partitions < 1:
            raise ValueError("Invalid partition value")
        if partitions > 1:
            return self._partition_solver(initial_nodes, final_nodes, partitions, self.ilp_solver)

        # Define the ILP problem and variables
        prob = LpProblem('Node recycling', LpMaximize)
        var_indexes = [
            (i, j)
            for i in range(len(initial_nodes))
            for j in range(len(final_nodes))
            if self._valid_node_recycling(initial_nodes[i], final_nodes[j])
        ]
        z_vars = LpVariable.dicts('Z', indices=var_indexes, cat=LpBinary)

        # Calculate recycling level for each node pair
        node_pair_recycling_levels = {
            (i, j): self.node_pair_recycling_level(initial_nodes[i], final_nodes[j])
            for (i, j) in var_indexes
        }

        # Objective function
        prob += (lpSum(node_pair_recycling_levels[(i, j)] * z_vars[(i, j)] for (i, j) in var_indexes), "Sum_recyclings")

        # Constraints on initial nodes: a node can be recycled only once
        for i in range(len(initial_nodes)):
            prob += (lpSum(z_vars[(i, j)] for m, j in var_indexes if m == i) <= 1, f"Recycle_initial_node_{i}_once")
        for j in range(len(final_nodes)):
            prob += (lpSum(z_vars[(i, j)] for i, n in var_indexes if n == j) <= 1, f"Recycle_final_node_{j}_once")

        # Solve the ILP problem using CBC
        try:
            prob.solve(PULP_CBC_CMD(msg=0, gapRel=MAX_ILP_ERROR, timeLimit=MAX_ILP_TIME_SECS), use_mps=False)
        # If the solver failed to solve
        except PulpSolverError as _:
            return None, 0
        if prob.status != LpStatusOptimal or prob.sol_status != LpStatusOptimal:
            return None, 0

        # If the solver has succeded
        best_node_pairs = [
            (initial_nodes[index_pair[0]], final_nodes[index_pair[1]])
            for index_pair in var_indexes
            if z_vars[(index_pair[0], index_pair[1])].value() > 0
        ]
        return best_node_pairs, pulp_value(prob.objective) / len(initial_nodes)

    def _split_nodes(self, initial_nodes: list[Vm], final_nodes: list[Vm], n_partitions: int = 2) \
                        -> list[tuple[list[Vm], list[Vm]]]:
        """
        Split one list of initial nodes and a list of final nodes, reducing the complexity of calculating the
        complexity of node recycling, at the cost of reducing the container recycling level.
        :param initial_nodes: Initial nodes.
        :param final_nodes: Final nodes.
        :param n_partitions: Number of partitions.
        :return: A list of partitions.
        """
        min_length = min(len(initial_nodes), len(final_nodes))
        # Splitting is not required in this case
        if min_length == 1:
            return [(initial_nodes, final_nodes)]

        # Get the recycling problem solution from a greedy solver
        node_recyclings, _ = self.greedy_solver(initial_nodes, final_nodes)

        # Split the node pairs
        if min_length <= n_partitions:
            return [([node_recyclings[i][0]],[node_recyclings[i][1]]) for i in range(min_length)]
        partitions = []
        partition_size = min_length // n_partitions
        first_partition_index = 0
        for partition_index in range(n_partitions):
            last_partition_index = first_partition_index + partition_size
            if partition_index == n_partitions - 1:
                last_partition_index = min_length
            inodes = [node_recyclings[i][0] for i in range(first_partition_index, last_partition_index)]
            fnodes = [node_recyclings[i][1] for i in range(first_partition_index, last_partition_index)]
            partitions.append((inodes, fnodes))
            first_partition_index += partition_size
        return partitions

    def _partition_solver(self, initial_nodes: list[Vm], final_nodes: list[Vm], 
                          base_solver: RecyclingSolverType = hungarian_solver, n_partitions: int = 2) \
                                -> tuple[list[tuple[Vm, Vm]], float]:
        """
        Calculate the recyclings between the list of initial nodes and the list of final nodes using the partition
        simplification and the given base solver.
        :param initial_nodes: Initial nodes.
        :param final_nodes: Final nodes.
        :param partitions: Number of problem partitions.
        :return: A list of pairs (initial node, final node) that recycle each other, as well as the recycling level.
        """
        partitions = self._split_nodes(initial_nodes, final_nodes, n_partitions)
        recycling_node_pairs = []
        recycling_sum = 0
        partition_initial_nodes = []
        partition_final_nodes = []
        for partition in partitions:
            partition_initial_nodes.extend(partition[0])
            partition_final_nodes.extend(partition[1])
            # The recycling calculation is performed on each partition recursively.
            # Neither new nor obsolete nodes are possible, since partitions have the
            # same number of initial and final nodes
            recycling = base_solver(partition[0], partition[1], 1)
            recycling_node_pairs.extend(recycling[0])
            recycling_sum += recycling[1] * len(partition[0])
        return recycling_node_pairs, recycling_sum / len(initial_nodes)

    def calculate_node_recycling(self, initial_nodes: list[Vm], final_nodes: list[Vm],
                                 solver: RecyclingSolverType = hungarian_solver,
                                 partitions: int = 1) -> dict[str, any]:
        """
        Calculate the node recycling from a list of initial nodes and a list of final nodes.
        :param initial_nodes: Initial nodes.
        :param final_nodes: Final nodes.
        :param solver: Solver to calculate node recycling.
        :param partitions: Number of problem partitions.
        :return: A dictionary with "obsolete", "recycled", "new" and "level" keys storing: a list of obsolete nodes
        (initial nodes not recycled), a list of tuples (initial node, final node) that are recycled, a list of
        new nodes (final nodes not recycled) and a measurement of node recycling.
        """
        recycling_node_pairs, recycling_level = \
            solver(self, initial_nodes, final_nodes, partitions)
        if recycling_node_pairs is not None:
            obsolete_nodes = [node for node in initial_nodes if node not in [pair[0] for pair in recycling_node_pairs]]
            new_nodes = [node for node in final_nodes if node not in [pair[1] for pair in recycling_node_pairs]]
            return {"obsolete": obsolete_nodes, "recycled_pairs": recycling_node_pairs, "new": new_nodes,
                    "level": recycling_level}

    def _calculate_container_recycling(self):
        """
        Calculate obsolete containers, new containers, recycled containers and scaled containers.
        """
        for node in self.obsolete_nodes:
            self.obsolete_containers[node] = {}
            for cg in node.cgs:
                self.obsolete_containers[node][cg.cc] = cg.replicas
        for node in self.new_nodes:
            self.new_containers[node] = {}
            for cg in node.cgs:
                self.new_containers[node][cg.cc] = cg.replicas
        for initial_node, final_node in (self.recycled_node_pairs | self.upgraded_node_pairs).items():
            self.obsolete_containers[initial_node] = {}
            self.new_containers[initial_node] = {}
            self.recycled_containers[initial_node] = {} 

        # Recycled and upgraded nodes can contain recycled containers, new containers, obsolete containers 
        # and scaled containers
        for initial_node, final_node in (self.recycled_node_pairs | self.upgraded_node_pairs).items():
            container_pairs = self._get_container_pairs(initial_node.cgs, final_node.cgs)
            for initial_cc, final_cc, replicas in container_pairs:
                if initial_cc is None:
                    self.new_containers[initial_node][final_cc] = replicas
                elif final_cc is None:
                    self.obsolete_containers[initial_node][initial_cc] = replicas
                elif initial_cc == final_cc:
                    self.recycled_containers[initial_node][initial_cc] = replicas
                else:
                    if node not in self.scaled_containers:
                        self.scaled_containers[initial_node] = {(initial_cc, final_cc): replicas}
                    else:    
                        self.scaled_containers[initial_node][(initial_cc, final_cc)] = replicas

    def _calculate_recycling_levels(self, initial_alloc: Allocation):
        """
        Calculate node and container recycling levels.
        :param initial_alloc: Initial allocation.
        """
        initial_node_cores = sum(
            initial_node.ic.cores
            for initial_node in initial_alloc
        )
        initial_node_mem = sum(
            initial_node.ic.mem
            for initial_node in initial_alloc
        )
        initial_container_cores = sum(
            cg.cc.cores * cg.replicas
            for initial_node in initial_alloc
            for cg in initial_node.cgs
        )
        initial_container_mem = sum(
            cg.cc.memv * cg.replicas
            for initial_node in initial_alloc
            for cg in initial_node.cgs
        )
        recycled_node_cores = sum(
            initial_node.ic.cores
            for initial_node, _ in self.recycled_node_pairs.items()
        )
        recycled_node_mem = sum(
            initial_node.ic.mem
            for initial_node, _ in self.recycled_node_pairs.items()
        )
        recycled_container_cores = sum(
            cc.cores * replicas
            for _, cc_replicas in self.recycled_containers.items()
            for cc, replicas in cc_replicas.items()
        )
        recycled_container_mem = sum(
            cc.memv * replicas
            for _, cc_replicas in self.recycled_containers.items()
            for cc, replicas in cc_replicas.items()
        )
        if len(self.recycled_node_pairs) == 0:
            self.node_recycling_level = 0
            self.container_recycling_level = 0
        else:
            self.node_recycling_level = 0.5 * (recycled_node_cores / initial_node_cores +
                                               recycled_node_mem / initial_node_mem).magnitude
            self.container_recycling_level = 0.5 * ((recycled_container_cores / initial_container_cores).magnitude +
                                                    (recycled_container_mem / initial_container_mem).magnitude)

    def _initialize_fields(self):
        """
        Initialize all internal data structures to their default empty state.
        This includes recycled node/container mappings and recycling level metrics.
        """
        self.obsolete_nodes: list[Vm] = []
        self.recycled_node_pairs: dict[Vm, Vm] = {}
        self.upgraded_node_pairs: dict[Vm, Vm] = {}
        self.new_nodes: list[Vm] = []
        self.obsolete_containers: dict[Vm, dict[ContainerClass, int]] = {}
        self.recycled_containers: dict[Vm, dict[ContainerClass, int]] = {}
        self.scaled_containers: dict[Vm, dict[tuple[ContainerClass, ContainerClass], int]] = {} 
        self.new_containers: dict[Vm, dict[ContainerClass, int]] = {}
        self.node_recycling_level: float = 0
        self.container_recycling_level: float = 0

    def _classify_nodes_by_recycling(self, initial_ic_nodes: dict[InstanceClass, Vm],
                                     final_ic_nodes: dict[InstanceClass, Vm]) -> tuple[dict, dict]:
        """
        Determine which nodes are:
            - Obsolete: present only in initial allocation.
            - New: present only in final allocation.
            - Recyclable: present in both.
        :param initial_ic_nodes: Dict of initial nodes grouped by instance class.
        :param final_ic_nodes: Dict of final nodes grouped by instance class.
        :return: Tuple with two dicts: recyclable initial and recyclable final nodes.
        """
        initial_recyclable = {}
        final_recyclable = {}
        for ic, nodes in initial_ic_nodes.items():
            if ic not in final_ic_nodes:
                self.obsolete_nodes.extend(nodes)
            else:
                initial_recyclable[ic] = nodes
        for ic, nodes in final_ic_nodes.items():
            if ic not in initial_ic_nodes:
                self.new_nodes.extend(nodes)
            else:
                final_recyclable[ic] = nodes
        return initial_recyclable, final_recyclable

    def _assign_final_node_ids(self):
        """
        Update the IDs of final nodes to ensure:
            - Recycled and upgraded nodes inherit the ID of their corresponding initial node.
            - New nodes receive an incremented ID following the highest used in that instance class.
        """
        last_initial_node_ids = defaultdict(lambda: -1)
        for node in self.obsolete_nodes:
            last_initial_node_ids[node.ic] = max(last_initial_node_ids[node.ic], node.id)
        for init_node, final_node in self.recycled_node_pairs.items():
            final_node.id = init_node.id # Recycled node pairs have the same ID
            last_initial_node_ids[init_node.ic] = max(last_initial_node_ids[init_node.ic], init_node.id)
        for _, final_node in self.upgraded_node_pairs.items():
            last_initial_node_ids[final_node.ic] += 1
            final_node.id = last_initial_node_ids[final_node.ic]
        for node in self.new_nodes:
            last_initial_node_ids[node.ic] += 1
            node.id = last_initial_node_ids[node.ic]
