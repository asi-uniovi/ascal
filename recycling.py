from collections import defaultdict
from itertools import permutations
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
from fcma.helper import _solve_cbc_patched
COIN_CMD.solve_CBC = _solve_cbc_patched
from fcma import Allocation, Vm, ContainerClass

# These values define the recycling algorithm to use as a function of n1 and n2,
# being n1 and n2 the maximum and minimum number of nodes, respectively, evaluated
# among the number of nodes in the initial and final allocations.
# 1) P(n1, n2) = n1! / (n1 - n2)! < ILP_SOLVER_THRESHOLD => Combinatorial solver
# 2) ILP_SOLVER_THRESHOLD <= P(n1, n2) = n1! / (n1 - n2)! => ILP solver
# In addition, implement problem partition with PARTITION_SOLVER_DIVIDER when the ILP solver
# times are higher than MAX_PROBLEM_TIME_SECS.
ILP_SOLVER_THRESHOLD = 10000
MAX_ILP_TIME_SECS = 1
PARTITION_SOLVER_DIVIDER = 4
MAX_ILP_ERROR = 0.02

class Recycling:
    """
    Class to calculate the recyclings between two consecutive allocations.
    """
    @staticmethod
    def node_pair_recycling_level(initial_node: Vm, final_node: Vm) -> float:
        """
        Calculate the recycling level between an initial and a final node. The recycling level
        is 1.0 when both nodes come from the same instance class and all the conatiners in the initial
        node are allocated in the final node.
        :param initial_node: Initial node.
        :param final_node: Final node.
        :return: The recycling level.
        """
        if initial_node.ic != final_node.ic:
            return 0
        recycling_level = 0
        ic_cores = (initial_node.ic.cores - initial_node.free_cores).magnitude
        ic_mem = (initial_node.ic.mem - initial_node.free_mem).magnitude
        for cg1 in initial_node.cgs:
            for cg2 in final_node.cgs:
                if cg1.cc == cg2.cc:
                    recycling_level += 0.5 * min(cg1.replicas, cg2.replicas) * (cg1.cc.cores.magnitude / ic_cores +
                                                                                cg1.cc.mem[0].magnitude / ic_mem)
                    break
        return recycling_level

    @staticmethod
    def _split_nodes(initial_nodes: list[Vm], final_nodes: list[Vm]) -> list[tuple[list[Vm], list[Vm]]]:
        """
        Split one list of initial nodes and a list of final nodes, reducing the complexity of calculating the
        complexity of node recycling, at the cost of reducing the container recycling level.
        :param initial_nodes: Initial nodes.
        :param final_nodes: Final nodes.
        :return: A list of partitions.
        """
        min_length = min(len(initial_nodes), len(final_nodes))
        # Splitting is not required in this case
        if min_length == 1:
            return [(initial_nodes, final_nodes)]

        # Calculate recycling levels for each node pair
        node_pair_recycling_levels = [
            (initial_node, final_node,  Recycling.node_pair_recycling_level(initial_node, final_node))
            for initial_node in initial_nodes
            for final_node in final_nodes
        ]

        # Sort the pairs by decreasing recycling level
        node_pair_recycling_levels.sort(key=lambda pair: pair[2], reverse=True)
        recycled_initial_nodes = []
        recycled_final_nodes = []
        pair_index = 0

        # Remove node pairs including an initial node or final node of a previous node pair (with higher
        # recycling level).
        while len(node_pair_recycling_levels) > min_length:
            initial_node, final_node, _ =  node_pair_recycling_levels[pair_index]
            if initial_node in recycled_initial_nodes or final_node in recycled_final_nodes:
                node_pair_recycling_levels.pop(pair_index)
            else:
                recycled_initial_nodes.append(initial_node)
                recycled_final_nodes.append(final_node)
                pair_index += 1

        # Split the node pairs
        partition_size = min_length // PARTITION_SOLVER_DIVIDER
        partitions = []
        first_partition_index = 0
        for partition_index in range(PARTITION_SOLVER_DIVIDER):
            last_partition_index = first_partition_index + partition_size
            if partition_index == PARTITION_SOLVER_DIVIDER - 1:
                last_partition_index = min_length
            inodes = [node_pair_recycling_levels[i][0] for i in range(first_partition_index, last_partition_index)]
            fnodes = [node_pair_recycling_levels[i][1] for i in range(first_partition_index, last_partition_index)]
            partitions.append((inodes, fnodes))
            first_partition_index += partition_size

        return partitions

    @staticmethod
    def _combinatorial_solver(initial_nodes: list[Vm], final_nodes: list[Vm]) -> tuple[list[tuple[Vm, Vm]], float]:
        """
        Calculate the recyclings between the list of initial nodes and the list of final nodes analyzing all
        the possible recyclings.
        :param initial_nodes: Initial nodes.
        :param final_nodes: Final nodes.
        :return: A list of pairs (initial node, final node) that recycle each other, as well as the recycling level.
        """
        if len(initial_nodes) >= len(final_nodes):
            larger_node_list = initial_nodes
            shorter_node_list = final_nodes
        else:
            larger_node_list = final_nodes
            shorter_node_list = initial_nodes
        perms = list(permutations(larger_node_list, len(shorter_node_list)))
        best_perm = None
        best_perm_recycling_level = 0

        # Get the permutation with the maximum recycling level
        for perm in perms:
            perm_recycling_level = 0
            node_index = 0
            for node_in_larger_list in perm:
                node_in_shorter_list = shorter_node_list[node_index]
                perm_recycling_level += Recycling.node_pair_recycling_level(node_in_larger_list, node_in_shorter_list)
                node_index += 1
            if perm_recycling_level >= best_perm_recycling_level:
                best_perm_recycling_level = perm_recycling_level
                best_perm = perm

        # Return the best permutation
        if len(initial_nodes) >= len(final_nodes):
            return ([(best_perm[i], final_nodes[i]) for i in range(len(final_nodes))],
                    best_perm_recycling_level / len(initial_nodes))
        else:
            return ([(initial_nodes[i], best_perm[i]) for i in range(len(initial_nodes))],
                    best_perm_recycling_level / len(initial_nodes))

    @staticmethod
    def _ilp_solver(initial_nodes: list[Vm], final_nodes: list[Vm]) -> tuple[list[tuple[Vm, Vm]], float]:
        """
        Calculate the recyclings between the list of initial nodes and the list of final nodes using and
        Integer Linnear Programming (ILP) solver.
        :param initial_nodes: Initial nodes.
        :param final_nodes: Final nodes.
        :return: A list of pairs (initial node, final node) that recycle each other, as well as the recycling level.
        """

        # Calculate recycling leve for each node pair
        node_pair_recycling_levels = {
            (i,j): Recycling.node_pair_recycling_level(initial_nodes[i], final_nodes[j])
            for i in range(len(initial_nodes))
            for j in range(len(final_nodes))
        }

        # Define the ILP problem and variables
        prob = LpProblem('Node recycling', LpMaximize)
        var_indexes = [(i,j) for i in range(len(initial_nodes)) for j in range(len(final_nodes))]
        z_vars = LpVariable.dicts('Z', indices=var_indexes, cat=LpBinary)

        # Objective function
        prob += (lpSum(node_pair_recycling_levels[(i,j)] * z_vars[(i, j)] for (i,j) in var_indexes), "Sum_recyclings")

        # Constraints on initial nodes: an initial node may be on a single recycling only
        for i in range(len(initial_nodes)):
            prob += (lpSum(z_vars[(i, j)] for j in range(len(final_nodes))) <= 1, f"Recycle_initial_node_{i}_once")
        for j in range(len(final_nodes)):
            prob += (lpSum(z_vars[(i, j)] for i in range(len(initial_nodes))) <= 1, f"Recycle_final_node_{j}_once")

        # Solve the ILP problem using CBC
        try:
            prob.solve(PULP_CBC_CMD(msg=0, gapRel= MAX_ILP_ERROR, timeLimit=MAX_ILP_TIME_SECS), use_mps=False)
        # If the solver failed to solve
        except PulpSolverError as _:
            return None, 0
        if prob.status != LpStatusOptimal or prob.sol_status != LpStatusOptimal:
            return None, 0

        # if the solver has succeded
        best_node_pairs = [
            (initial_nodes[index_pair[0]], final_nodes[index_pair[1]])
            for index_pair in var_indexes
            if z_vars[(index_pair[0], index_pair[1])].value() > 0
        ]
        return best_node_pairs, pulp_value(prob.objective) / len(initial_nodes)

    @staticmethod
    def _calculate_node_recycling(initial_nodes: list[Vm], final_nodes: list[Vm]) -> dict[str, any]:
        """
        Calculate the node recycling from a list of initial nodes and a list of final nodes.
        :param initial_nodes: Initial nodes.
        :param final_nodes: Final nodes.
        :return: A dictionary with "removed", "recycled", "new" and "level" keys storing: a list of removed nodes
        (initial nodes not recycled), a list of tuples (initial node, final node) that are recycled, a list of
        new nodes (final nodes not recycled) and a measurement of node recycling.
        """
        longer_list_length = max(len(initial_nodes), len(final_nodes))
        shorter_list_lenght = min(len(initial_nodes), len(final_nodes))

        # Calculate the number of possible recyclings as permutations(longer_list_length, short_list_length)
        n_possible_recyclings = 1
        for i in range(shorter_list_lenght):
            n_possible_recyclings *= longer_list_length
            longer_list_length -= 1
            if n_possible_recyclings >= ILP_SOLVER_THRESHOLD:
                break

        # If the number of possible recyclings is affordable then use the combinatorial solver
        if n_possible_recyclings < ILP_SOLVER_THRESHOLD:
            recycling_node_pairs, recycling_level = \
                Recycling._combinatorial_solver(initial_nodes, final_nodes)
        # When the number of possible recyclings is too high then solve as an ILP problem
        else:
            recycling_node_pairs, recycling_level = Recycling._ilp_solver(initial_nodes, final_nodes)
        # If the combinatorial solver or the ILP solver provide a solution
        if recycling_node_pairs is not None:
            removed_nodes = [node for node in initial_nodes if node not in [pair[0] for pair in recycling_node_pairs]]
            new_nodes = [node for node in final_nodes if node not in [pair[1] for pair in recycling_node_pairs]]
            return {"removed": removed_nodes, "recycled_pairs": recycling_node_pairs, "new": new_nodes,
                    "level": recycling_level}

        # If the problem was too large for the ILP solver then partition the node recycling problem.
        # Each partition is a tuple with a reduced number of initial and final nodes.
        partitions = Recycling._split_nodes(initial_nodes, final_nodes)
        recycling_node_pairs = []
        recycling_sum = 0
        partition_initial_nodes = []
        partition_final_nodes = []
        for partition in partitions:
            partition_initial_nodes.extend(partition[0])
            partition_final_nodes.extend(partition[1])
            # The recycling calculation is performed on each partition recursively.
            # Neither new nor removed nodes are possible, since partitions have the
            # same number of initial and final nodes
            recycling = Recycling._calculate_node_recycling(partition[0], partition[1])
            recycling_node_pairs.extend(recycling["recycled_pairs"])
            recycling_sum += recycling["level"] * len(partition[0])
        removed_nodes = [node for node in initial_nodes if node not in partition_initial_nodes]
        new_nodes = [node for node in final_nodes if node not in partition_final_nodes]
        return {"removed": removed_nodes, "recycled_pairs": recycling_node_pairs, "new": new_nodes,
                "level": recycling_sum / len(initial_nodes)}

    def _calculate_container_recycling(self):
        """
        Calculate the removed containers, new containers and recycled containers from the list of removed nodes,
        new nodes and recycled node pairs.
        """
        self.removed_containers = {}
        for node in self.removed_nodes:
            self.removed_containers[node] = defaultdict(lambda: 0)
            for cg in node.cgs:
                self.removed_containers[node][cg.cc] = cg.replicas
        self.new_containers = {}
        for node in self.new_nodes:
            self.new_containers[node] = defaultdict(lambda: 0)
            for cg in node.cgs:
                self.new_containers[node][cg.cc] = cg.replicas
        self.recycled_containers = {node: defaultdict(lambda: 0) for node, _ in self.recycled_node_pairs.items()}

        for initial_node, final_node in self.recycled_node_pairs.items():
            for cg1 in initial_node.cgs:
                found_cc_in_final_node = False
                for cg2 in final_node.cgs:
                    # Containers in the same container class for both the initial and final nodes
                    # may be recycled, removed or new, depending on the number
                    if str(cg1.cc) == str(cg2.cc):
                        self.recycled_containers[initial_node][cg1.cc] = min(cg1.replicas, cg2.replicas)
                        if cg1.replicas > cg2.replicas:
                            if initial_node not in self.removed_containers:
                                self.removed_containers[initial_node] = defaultdict(lambda: 0)
                            self.removed_containers[initial_node][cg1.cc] += cg1.replicas - cg2.replicas
                        elif cg1.replicas < cg2.replicas:
                            if initial_node not in self.new_containers:
                                self.new_containers[initial_node] = defaultdict(lambda : 0)
                            self.new_containers[initial_node][cg1.cc] +=  cg2.replicas - cg1.replicas
                        found_cc_in_final_node = True
                        break
                # If the container class appears only in the initial node
                if not found_cc_in_final_node:
                    if initial_node not in self.removed_containers:
                        self.removed_containers[initial_node] =  defaultdict(lambda : 0)
                    self.removed_containers[initial_node][cg1.cc] += cg1.replicas
            # Find containers in the final node in container classes that do not appear in the initial nodes
            for cg2 in final_node.cgs:
                found_cc_in_initial_node = False
                for cg1 in initial_node.cgs:
                    if str(cg1.cc) == str(cg2.cc):
                        found_cc_in_initial_node = True
                        break
                if not found_cc_in_initial_node:
                    if initial_node not in self.new_containers:
                        self.new_containers[initial_node] = defaultdict(lambda : 0)
                    self.new_containers[initial_node][cg2.cc] += cg2.replicas

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
            cg.cc.mem[0] * cg.replicas
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
            cc.mem[0] * replicas
            for _, cc_replicas in self.recycled_containers.items()
            for cc, replicas in cc_replicas.items()
        )
        if len(self.recycled_node_pairs) == 0:
            self.node_recycling_level = 0
        else:
            self.node_recycling_level = 0.5 * (recycled_node_cores / initial_node_cores +
                                               recycled_node_mem / initial_node_mem).magnitude
        if len(self.recycled_containers) == 0:
            self.container_recycling_level = 0
        else:
            self.container_recycling_level = 0.5 * (recycled_container_cores / initial_container_cores +
                                                    recycled_container_mem / initial_container_mem).magnitude

    def __init__(self, initial_alloc: Allocation, final_alloc: Allocation):
        """
        Constructor for the Recycling class. The Reycling class object defines properties to get the new nodes,
        removed nodes, node recycling pairs, removed containers, new containers and recycled containers.
        :param initial_alloc: Initial allocation.
        :param final_alloc: Final allocation
        """
        self.removed_nodes: list[Vm] = [] # Removed nodes. They come from the initial nodes.
        self.recycled_node_pairs: dict[Vm, Vm] = {} # Recyclings for the initial nodes
        self.new_nodes: list[Vm] = [] # New nodes. They come from the final nodes.
        self.removed_containers:  dict[Vm, dict[ContainerClass, int]] = {} # From removed and recycled nodes
        self.recycled_containers: dict[Vm, dict[ContainerClass, int]] = {} # From recycled nodes
        self.new_containers: dict[Vm, dict[ContainerClass, int]] = {} # From new and recycled nodes
        # All the recycling levels in [0, 1]
        self.node_recycling_level: float = 0 # Node recycling level. Considering all the instance classes
        self.container_recycling_level: float = 0 # Container recycling level. Considering all the instance classes

        # One dictionary with the initial nodes for each instance class
        initial_ic_nodes = {}
        for node in initial_alloc:
            if node.ic not in initial_ic_nodes:
                initial_ic_nodes[node.ic] = [node]
            else:
                initial_ic_nodes[node.ic].append(node)

        # One dictionary with the final nodes for each instance class
        final_ic_nodes = {}
        for node in final_alloc:
            if node.ic not in final_ic_nodes:
                final_ic_nodes[node.ic] = [node]
            else:
                final_ic_nodes[node.ic].append(node)

        # Some removed and new nodes are known in advance since they come from instance classes that appear
        # only in the initial or final nodes. In addition, get all the nodes that may be recycled
        initial_recyclable_nodes = {}
        final_recyclable_nodes = {}
        for ic, nodes in initial_ic_nodes.items():
            if ic not in final_ic_nodes:
                self.removed_nodes.extend(nodes)
            else:
                initial_recyclable_nodes[ic] = nodes
        for ic, nodes in final_ic_nodes.items():
            if ic not in initial_ic_nodes:
                self.new_nodes.extend(nodes)
            else:
                final_recyclable_nodes[ic] = nodes

        # Nodes that may be recyclable will be divided into:
        # - Removed => Recyclable initial nodes that are not finally recycled.
        # - New => Recyclable final nodes that do not come from recycling an initial node.
        # - Recycled => Pairs of recyclable initial and final nodes that are finally recycled.
        for ic in initial_recyclable_nodes:
            # Calculate additional removed and new nodes, recycled nodes and the recycling level
            node_recyclings = Recycling._calculate_node_recycling(initial_recyclable_nodes[ic],
                                                                  final_recyclable_nodes[ic])
            self.removed_nodes.extend(node_recyclings["removed"])
            for recycled_pair in node_recyclings["recycled_pairs"]:
                self.recycled_node_pairs[recycled_pair[0]] = recycled_pair[1]
            self.new_nodes.extend(node_recyclings["new"])

        # Calculate removed, recycled and new containers
        self._calculate_container_recycling()

        # Calculate recycling levels
        self._calculate_recycling_levels(initial_alloc)

