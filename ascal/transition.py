"""
Implement synchronous transitions between two allocations as a list of commands.
A synchronous transition defines predefined times to start node/container creations/removals, assuming
fixed durations to perform these operations. A command starts when the previous command completes
(and the node creation time has elapsed if the command flag "sync_on_nodes_creation" is set). A command ends
and the next can be processed after completing its container removals and allocations, so the time required
to execute a command is the sum of one container removal time and one container creation time.
Note that node creations and removals execute in background.
Within a command operations are executed following these restrictions:

        - Node creations and upgrades. Start inmediately with the command. Only the first command can include them.
        - Container removals. Start inmediately with the command.
        - Node removals. Start after completing all the container removals in the command.
        - Container allocations. Start after completing all the container removals in the command.

While useful for theoretical analysis, synchronous transitions are too restrictive in practice,
as creation and removal times are not fixed, and assuming their worst-case durations can be overly pessimistic.
Nevertheless, an asynchronous transition can be derived from a synchronous transition with small changes:

- Move container allocations in created nodes to the first command, i.e, the node creations command. Although
container allocations and node creations are triggered at the same time, container allocations are suspended
until the nodes really exist. Note that the command with container allocations in created nodes is identified by a flag
"sync_on_nodes_creation" enabled.

- Container removals, node removals and container allocations in a command are triggered with the command. However,
any individual node removal is suspended until its containers have been removed. Similarly, any individual
container allocation is suspended until the destination node exist (if the node required to be created) and has enough
computational resources.

"""
from math import ceil, floor
from json import loads
from collections import defaultdict
from dataclasses import dataclass, field
from fcma import Fcma, SolvingPars, System, Allocation, App, Vm, ContainerClass, RequestsPerTime, InstanceClass
from ascal.timedops import TimedOps
from ascal.recycling import Recycling
from ascal.helper import (
    get_min_max_perf,
    Vmt,
    RecyclingVmt,
    get_allocation_signature,
    get_app_perf_surplus
)


@dataclass
class Command:
    """
    The transition between an initial allocation and a final allocation is a sequence of commands.
    Each command can include several types of operations with nodes and containers.
    """
    allocate_containers: list[tuple[Vm|Vmt, ContainerClass, int]] = field(default_factory=list)
    remove_containers: list[tuple[Vm|Vmt, ContainerClass, int]] = field(default_factory=list)
    remove_nodes: list[Vm|Vmt] = field(default_factory=list)
    # Only the first command can create or upgrade nodes. Some commands must delay command execution until
    # completing node creations or node upgrades in the first command
    create_nodes: list[Vm|Vmt] = field(default_factory=list)
    upgrade_nodes: list[tuple[Vm, InstanceClass]] = field(default_factory=list)
    sync_on_nodes_creation: bool = False
    sync_on_nodes_upgrade: bool = False

    def extend(self, command: 'Command'):
        """
        Extend with a new command, appending all the container and node operations.
        :param command: Command to extend.
        """
        self.allocate_containers.extend(command.allocate_containers)
        self.remove_containers.extend(command.remove_containers)
        self.create_nodes.extend(command.create_nodes)
        self.remove_nodes.extend(command.remove_nodes)
        self.upgrade_nodes.extend(command.upgrade_nodes)

    def vmt_to_vm(self):
        """
        Replaces the Vmt nodes in the command by Vm nodes.
        """
        self.allocate_containers = [(node.vm, cc, replicas) for node, cc, replicas in self.allocate_containers]
        self.remove_containers = [(node.vm, cc, replicas) for node, cc, replicas in self.remove_containers]
        self.create_nodes = [node.vm for node in self.create_nodes]
        self.remove_nodes = [node.vm for node in self.remove_nodes]
        self.upgrade_nodes = [(initial_node.vm, final_ic) for initial_node, final_ic in self.upgrade_nodes]

    def is_null(self) -> bool:
        """
        Check if the command is null.
        :return: True when the command is null.
        """
        return (len(self.allocate_containers) == 0 and len(self.remove_containers) == 0 and
                len(self.create_nodes) == 0 and len(self.remove_nodes) == 0 and len(self.upgrade_nodes) == 0)

    def get_container_command_time(self, timing_args: TimedOps.TimingArgs) -> int:
        """
        Get the time required to complete a command ignoring node operations.
        :param timing_args: Times for creation/removal of nodes/containers.
        :return: The time to complete the command.
        """
        container_command_time = 0
        if len(self.remove_containers) > 0:
            container_command_time += timing_args.container_removal_time
        if len(self.allocate_containers) > 0:
            container_command_time += timing_args.container_creation_time
        return container_command_time

    def replace_nodes(self, node_pairs: dict[Vm, Vm]) -> 'Command':
        """
        Replace the nodes in the command by their counterparts in the node pairs.
        :param node_pairs: A dictionary with the final node for each initial node.
        :return: A new command with the replacement.
        """
        new_command = Command()
        new_command.sync_on_nodes_creation = self.sync_on_nodes_creation
        for node, cc, replicas in self.allocate_containers:
            if node in node_pairs:
                new_command.allocate_containers.append((node_pairs[node], cc, replicas))
            else:
                new_command.allocate_containers.append((node, cc, replicas))
        for node, cc, replicas in self.remove_containers:
            if node in node_pairs:
                new_command.remove_containers.append((node_pairs[node], cc, replicas))
            else:
                new_command.remove_containers.append((node, cc, replicas))
        for node in self.create_nodes:
            if node in node_pairs:
                new_command.create_nodes.append(node_pairs[node])
            else:
                new_command.create_nodes.append(node)
        for node in self.remove_nodes:
            if node in node_pairs:
                new_command.remove_nodes.append(node_pairs[node])
            else:
                new_command.remove_nodes.append(node)
        return new_command

class Transition:
    """
    Class to perform transitions between initial allocation and final allocations for a given system.
    """

    # Constant used to deal with numerical approximations
    _DELTA = 0.000001

    def __init__(self, timing_args: TimedOps.TimingArgs, system: System, time_limit: int = None,
                 hot_node_scale_up: bool = False):
        """
        Creates an object for transition between two allocations.
        :param timing_args: Creation and removal times for containers and nodes.
        :param system: System performance and computational requirements.
        :param time_limit: Maximum time to carry out transitions. By default, this is set to the node creation time.
        :param hot_node_scale_up: Set to enable hot node scaling-up.
        Anyway, transition times can be longer, specially when they are set to a value less than the node creation
        time and new nodes need to be created.
        """
        self._timing_args = timing_args
        self._system = system
        self._recycling = None
        self._recycling_vm = None
        self._current_alloc: list[Vmt] = None
        self._unalloc_node_cs: list[tuple[Vmt, ContainerClass, int]]  = None
        self._app_unalloc_perf: defaultdict[App, RequestsPerTime]  = None
        self._app_perf_surplus:  defaultdict[App, RequestsPerTime]  = None
        self._app_perf_increment: defaultdict[App, RequestsPerTime]  = None
        self._allocatable_cs_next_step: list[tuple[Vm, ContainerClass, int]] = None
        self._unallocated_containers_in_new_nodes: list[tuple[Vm, ContainerClass, int]] = None
        self._time_limit = time_limit if time_limit is not None else self._timing_args.node_creation_time
        self._hot_node_scale_up = hot_node_scale_up
        self._commands: list[Command] = None
        self._sync_on_next_alloc_upgraded_nodes = True

    def get_worst_case_transition_time(self) -> int:
        """
        Get the worst-case transition time excluding node removals.
        :return: The worst-case transition time.
        """
        # - If time_limit == 0: wait for node creation + allocate containers in new/tmp nodes +
        #   remove_allocate + remove obsolete containers/nodes.
        # - If time_limit > 0: wait for max(NCT , ceil(limit/CRCA) * CRCA) + allocate containers in
        #   new/tmp nodes + remove_allocate + remove obsolete containers/nodes, where NCT is the
        # node creation time and CRCA is the time to remove containers plus allocating containers.
        # In the worst-case, node removals occur at the end of transition.

        if self._time_limit == 0:
            return self._timing_args.node_creation_time + self._timing_args.container_creation_time + \
                   (self._timing_args.container_removal_time + self._timing_args.container_creation_time) + \
                   self._timing_args.container_removal_time + self._timing_args.node_removal_time

        if self._time_limit > 0:
            crca = self._timing_args.container_removal_time + self._timing_args.container_creation_time
            return max(self._timing_args.node_creation_time, ceil(self._time_limit / crca) * crca) +\
                   self._timing_args.container_creation_time + crca + self._timing_args.container_removal_time +\
                   self._timing_args.node_removal_time

    def _remove_allocate(self, cc: ContainerClass, replicas: int, node: Vmt,
                         command: Command, obsolete: bool=False) -> int:
        """
        Allocate the container replicas to the node, freeing up computational resources in the node
        coming from obsolete containers if necessary, while ensuring that application's minimum performance
        constraints are met. The node state does not change when no replicas are allocated.
        :param cc: Container class.
        :param replicas: The number of replicas to allocate.
        :param node: Node.
        :param command: Command with (obsolete) containers to be removed and containers to be allocated.
        :param obsolete: True if the containers to allocate are obsolete, coming from another node.
        :return: The number of actually allocated replicas.
        """

        # Required computational resources to allocate the replicas
        required_cores = replicas * cc.cores
        required_mem = replicas * cc.mem[0]

        # Obsolete replicas in the node to be removed in order to allocate the replicas
        obsolete_replicas = []
        if node in self._recycling.obsolete_containers:
            # A deep copy of application's performance surplus
            available_perf_surplus = dict(self._app_perf_surplus)
            for obsolete_cc in self._recycling.obsolete_containers[node]:
                # We do not remove the same replicas to allocate
                if obsolete_cc == cc:
                    continue
                # Number of required obsolete replicas of the container class to remove
                required_replicas = max(
                    ceil((required_cores - node.free_cores) / obsolete_cc.cores),
                    ceil((required_mem - node.free_mem) / obsolete_cc.mem[0])
                )
                if required_replicas == 0:
                    break

                """
                To be improved. It assumes a single family of container classes. 
                """

                # Get the number of replicas of the obsolete container that could be removed
                obsolete_replicas_count = self._get_removable_replicas(obsolete_cc, node, required_replicas,
                                                                       available_perf_surplus[obsolete_cc.app])
                if obsolete_replicas_count > 0:
                    available_perf_surplus[obsolete_cc.app] -= obsolete_cc.perf * obsolete_replicas_count
                    obsolete_replicas.append((obsolete_cc, obsolete_replicas_count))
                    required_cores -= obsolete_replicas_count * obsolete_cc.cores
                    required_mem -= obsolete_replicas_count * obsolete_cc.mem[0]

        # Calculate the number of cores and memory obtained from the removal of obsolete containers in the node
        removed_cores = replicas * cc.cores - required_cores
        removed_mem = replicas * cc.mem[0] - required_mem

        # Calculate how many new container replicas could be allocated after the removals
        allocatable_replicas = min(
            replicas,
            int((node.free_cores.magnitude + removed_cores.magnitude + Transition._DELTA) / cc.cores.magnitude),
            int((node.free_mem.magnitude + removed_mem.magnitude + Transition._DELTA) / cc.mem[0].magnitude)
        )

        # Allocate the replicas. At this point, the removals and allocations are actually performed
        if allocatable_replicas > 0:
            for obsolete_cc, obsolete_replicas_count in obsolete_replicas:
                self._remove_obsolete_replicas(obsolete_cc, obsolete_replicas_count, node, command)
            allocated_replicas = self._allocate(cc, allocatable_replicas, node, command, obsolete)
            assert allocatable_replicas == allocated_replicas, "The replicas must be allocatable"

        return allocatable_replicas

    def _allocate(self, cc: ContainerClass, replicas: int, node: Vmt, command: Command, obsolete: bool=False) -> int:
        """
        Allocate a number of replicas in the node.
        :param cc: Container class.
        :param replicas: The number of replicas to allocate.
        :param node: Node.
        :param command: Allocation command.
        :param obsolete: Set to True if the replicas to allocate will be obsolete once allocated.
        :return: The number of actually allocated replicas.
        """
        allocatable_replicas = min(
            replicas,
            int((node.free_cores.magnitude + Transition._DELTA) / cc.cores.magnitude),
            int((node.free_mem.magnitude + Transition._DELTA) / cc.mem[0].magnitude)
        )
        if allocatable_replicas > 0:
            node.free_cores = node.free_cores - allocatable_replicas * cc.cores
            assert node.free_cores.magnitude > - Transition._DELTA, "Node free cores cannot not be negative"
            node.free_mem = node.free_mem - allocatable_replicas * cc.mem[0]
            assert node.free_mem.magnitude > - Transition._DELTA, "Node free memory cannot not be negative"
            node.replicas[cc] += allocatable_replicas
            command.allocate_containers.append((node, cc, allocatable_replicas))
            if not obsolete:
                self._app_unalloc_perf[cc.app] -= allocatable_replicas * cc.perf
                assert self._app_unalloc_perf[cc.app].magnitude > -Transition._DELTA, "Invalid performance"
            else:
                if node not in self._recycling.obsolete_containers:
                    self._recycling.obsolete_containers[node] = {cc: allocatable_replicas}
                else:
                    if cc not in self._recycling.obsolete_containers[node]:
                        self._recycling.obsolete_containers[node][cc] = allocatable_replicas
                    else:
                        self._recycling.obsolete_containers[node][cc] += allocatable_replicas
            self._app_perf_increment[cc.app] += allocatable_replicas * cc.perf
        return allocatable_replicas

    def _remove_obsolete_replicas(self, cc: ContainerClass, replicas: int, node: Vmt, command: Command) -> int:
        """
        Remove obsolete replicas from the container class in the node.
        :param cc: Container class.
        :param replicas: Replicas to remove.
        :param node: Node.
        :param command: Command with container removals.
        :return: Number of replicas that are actually removed.
        """
        if cc not in node.replicas:
            return 0
        removed_replicas = min(node.replicas[cc], replicas)
        command.remove_containers.append((node, cc, removed_replicas))
        node.replicas[cc] -= removed_replicas
        if node.replicas[cc] == 0:
            del node.replicas[cc]
        else:
            assert node.replicas[cc] >= 0, "Invalid number of replicas"
        node.free_cores += cc.cores * removed_replicas
        assert (node.free_cores - node.ic.cores).magnitude < Transition._DELTA, "Invalid node free cores"
        node.free_mem += cc.mem[0] * removed_replicas
        assert (node.free_mem - node.ic.mem).magnitude < Transition._DELTA, "Invalid node free mem"
        self._app_perf_surplus[cc.app] -= cc.perf * removed_replicas
        assert self._app_perf_surplus[cc.app].magnitude > -Transition._DELTA, "Performance deficit"
        self._recycling.obsolete_containers[node][cc] -= removed_replicas
        if self._recycling.obsolete_containers[node][cc] == 0:
            del self._recycling.obsolete_containers[node][cc]
        else:
            assert self._recycling.obsolete_containers[node][cc] >= 0, "Invalid number of replicas"
        return removed_replicas

    def _get_removable_replicas(self, cc: ContainerClass, node: Vmt, replicas_to_remove: int,
                                available_perf_surplus: RequestsPerTime) -> int:
        """
        Get the number of replicas of a container class in a node that can be removed while ensuring
        the application's minimum performance constraint.
        :param cc: Container class.
        :param node: Node.
        :param replicas_to_remove: The number of replicas to remove.
        :param available_perf_surplus: The available performance surplus for the container application.
        :return: The actual number of removable replicas.
        """
        n_removable = min(
            node.replicas[cc],
            int(available_perf_surplus / cc.perf),
            replicas_to_remove,
            self._recycling.obsolete_containers[node][cc]
        )
        return n_removable

    def _get_sorted_nodes(self, nodes_to_sort: list[Vm] = None) -> list[Vmt]:
        """
        Sort nodes with allocated containers in descending order of maximum freeable computational capacity.
        This capacity is calculated considering not only free computational resources, but also
        obsolete containers that may be removed while fullfiling application's minimum performance contraints.
        Empty nodes, i.e., nodes that do not allocate containers, are placed at the end of the list sorted by
        increasing price.
        :param nodes_to_sort: Sort the nodes in this list or all the nodes in the allocation when
        this parameter is not set.
        :return: A list of sorted nodes.
        """
        if nodes_to_sort is None:
            nodes_to_sort = self._current_alloc
        free_cores_list = []
        free_mem_list = []
        empty_nodes_list = []
        allocated_nodes_list = []

        for node in nodes_to_sort:
            # Check if node is empty
            if (node.ic.cores - node.free_cores).magnitude < Transition._DELTA and \
                    (node.ic.mem - node.free_mem).magnitude < Transition._DELTA:
                empty_nodes_list.append(node)
                continue
            free_cores = node.free_cores
            free_mem = node.free_mem
            if free_cores.magnitude == 0 or free_mem.magnitude == 0:
                continue
            allocated_nodes_list.append(node)
            app_perf_surplus = dict(self._app_perf_surplus)
            for cc, replicas in node.replicas.items():
                if node in self._recycling.obsolete_containers and cc in self._recycling.obsolete_containers[node]:
                    app = cc.app
                    cc_perf_surplus = min(app_perf_surplus[app], cc.perf * replicas)
                    surplus_replicas = floor(cc_perf_surplus / cc.perf + Transition._DELTA)
                    free_cores += surplus_replicas * cc.cores
                    free_mem += surplus_replicas * cc.mem[0]
                    app_perf_surplus[app] -= cc_perf_surplus
            free_cores_list.append(free_cores)
            free_mem_list.append(free_mem)
        total_free_cores = sum(free_cores_list)
        total_free_mem = sum(free_mem_list)
        if len(allocated_nodes_list) == 0:
            return sorted(empty_nodes_list, key=lambda n: n.ic.price)
        free_capacities = [
            (free_cores_list[i]/total_free_cores).magnitude + (free_mem_list[i]/total_free_mem).magnitude
            for i in range(len(allocated_nodes_list))
        ]
        allocated_nodes_list = [
            node
            for node, size in sorted(zip(allocated_nodes_list, free_capacities), key=lambda x: x[1], reverse=True)
        ]
        empty_nodes_list.sort(key=lambda n: n.ic.price)

        return allocated_nodes_list + empty_nodes_list

    def remove_obsolete_containers(self, command: Command):
        """
        Check if there are obsolete containers that can be removed. The obsolete containers of an application
        can be removed when all its new containers have been allocated.
        :param command: A command with the removal of containers.
        """
        for node, obsolete_cc_replicas in self._recycling.obsolete_containers.items():
            for obsolete_cc, replicas in dict(obsolete_cc_replicas.items()).items():
                if self._app_unalloc_perf[obsolete_cc.app].magnitude < Transition._DELTA:
                    replicas_count = self._remove_obsolete_replicas(obsolete_cc, replicas, node, command)
                    assert replicas_count == replicas, "All the replicas must be removable"

    def remove_obsolete_nodes(self, command: Command):
        """
        Check if there are obsolete nodes that can be removed. Obsolete nodes can be removed when
        they do not allocate containers. A command with the node removal operation is generated, but the
        nodes are not actually removed from the allocation, since they may be useful during the transition.
        :param command: A command with the removal of nodes.
        """
        for node in self._recycling.obsolete_nodes[:]:
            if node.is_empty():
                assert (node.free_cores - node.ic.cores).magnitude < Transition._DELTA, "Can not remove the node"
                assert (node.free_mem - node.ic.mem).magnitude < Transition._DELTA, "Can not remove the node"
                command.remove_nodes.append(node)
                del self._recycling.obsolete_containers[node]
                self._recycling.obsolete_nodes.remove(node)

    def _remove_allocate_copy_init(self, min_perf: dict[App, RequestsPerTime]):
        """
        Initialize the remove-allocate-copy algorithm.
        :param min_perf: Minimum application performances, to be fulfilled during the transition.
        """

        # Calculate the total cores and memory of new containers in recycled and upgraded nodes.
        # They are necessary to calculate container sizes
        total_new_cpu = 0
        total_new_mem = 0
        for n, cc_replicas in self._recycling.new_containers.items():
            # We focus on new containers in recycled nodes
            if n in self._recycling.recycled_node_pairs | self._recycling.upgraded_node_pairs:
                total_new_cpu += sum(cc.cores.magnitude * replicas for cc, replicas in cc_replicas.items())
                total_new_mem += sum(cc.mem[0].magnitude * replicas for cc, replicas in cc_replicas.items())

        # Build a list of unallocated containers (new containers) in recycled and upgraded nodes,
        # sorted by decreasing size
        self._unalloc_node_cs = []
        container_sizes = []
        for n, cc_replicas in self._recycling.new_containers.items():
            if n in self._recycling.recycled_node_pairs | self._recycling.upgraded_node_pairs:
                for cc, replicas in cc_replicas.items():
                    new_replicas = (n, cc, replicas)
                    container_size = cc.cores.magnitude / total_new_cpu + cc.mem[0].magnitude / total_new_mem
                    container_sizes.append(container_size)
                    self._unalloc_node_cs.append(new_replicas)
        self._unalloc_node_cs = [
            new_replicas
            for size, new_replicas in sorted(zip(container_sizes, self._unalloc_node_cs), key=lambda x: x[0],
                                             reverse = True)
        ]

        # Get unallocated application performances. It is calculated from application containers not allocated yet
        self._app_unalloc_perf = defaultdict(lambda : RequestsPerTime("0 req/s"))
        for _, cc_replicas in self._recycling.new_containers.items():
            for cc, replicas in cc_replicas.items():
                self._app_unalloc_perf[cc.app] += replicas * cc.perf

        # Get initial application's performance surplus
        self._app_perf_surplus = get_app_perf_surplus(min_perf, self._current_alloc)
        assert min(self._app_perf_surplus.values()).magnitude >= 0, "Invalid performance surplus"

        # Performance increment to add at the end of the command
        self._app_perf_increment = defaultdict(lambda : RequestsPerTime("0 req/s"))


        # List of containers that will be allocatable in the next execution of the remove-allocate-copy algorithm
        self._allocatable_cs_next_step = []

    def _allocate_copying_obsolete_containers(self, node_cc_replicas: tuple[Vmt, ContainerClass, int],
                                              available_obsolete_containers: dict[ContainerClass, int],
                                              dest_nodes: list[Vmt]) -> tuple[int, Command]:
        """
        Allocate container replicas in a node copying obsolete replicas from the node to one of the destination nodes.
        :param node_cc_replicas: Container replicas to allocate in a given node
        :param available_obsolete_containers: Elegible obsolete containers in the node.
        :param dest_nodes: Destination nodes for the elegible obsolete containers.
        :return: A tuple with the number of allocatable containers and a command with the allocation.
        """

        # Source node, container class and number of replicas to allocate
        node, cc, replicas_to_allocate = node_cc_replicas

        # Required cores and memory to allocate all the replicas
        required_cores = replicas_to_allocate * cc.cores
        required_mem = replicas_to_allocate * cc.mem[0]

        # The state must be recovered when we fail to allocate at least one replica, so create backups
        dest_nodes_modified = []
        dest_nodes_backup = {
            node: (node.free_cores, node.free_mem, defaultdict(lambda: 0, node.replicas))
            for node in dest_nodes
        }
        zero_perf = RequestsPerTime("0 req/s")
        app_perf_surplus_backup = defaultdict(lambda: zero_perf, self._app_perf_surplus)
        app_perf_increment_backup = defaultdict(lambda: zero_perf, self._app_perf_increment)
        removed_containers_backup = {
            node: {cc: replicas for cc, replicas in cc_replicas.items()}
            for node, cc_replicas in self._recycling.obsolete_containers.items()
        }
        removed_containers_backup = defaultdict(lambda: 0, dict(removed_containers_backup))

        # Allocate obsolete container copies in destination nodes
        command2 = Command()
        allocated_obsolete_replicas = [] # In case of success, these obsolete replicas will no longer be elegible
        for removable_cc, removable_replicas in available_obsolete_containers.items():
            # Required obsolete containers to free up enough computational resurces to allocate
            # replicas_to_allocate replicas in the node, considering rounding errors and the maximum
            # number of available replicas
            required_obsolete_replicas = max(
                ((required_cores - node.free_cores) / removable_cc.cores).magnitude,
                ((required_mem - node.free_mem) / removable_cc.mem[0]).magnitude
            )
            required_obsolete_replicas = int(ceil(required_obsolete_replicas - Transition._DELTA))
            required_obsolete_replicas = min(required_obsolete_replicas, removable_replicas)

            if required_obsolete_replicas == 0:
                break

            # Try copying required_obsolete_replicas to other nodes
            for dest_node in dest_nodes:
                if dest_node == node:
                    # Cannot copy obsolete containers to the same node
                    continue
                removed_obsolete_replicas = self._remove_allocate(removable_cc, required_obsolete_replicas,
                                                                  dest_node, command2, obsolete=True)
                if removed_obsolete_replicas > 0:
                    allocated_obsolete_replicas.append((removable_cc, removed_obsolete_replicas))
                    dest_nodes_modified.append(dest_node)
                    required_cores -= removed_obsolete_replicas * removable_cc.cores
                    required_mem -= removed_obsolete_replicas * removable_cc.mem[0]
                    required_obsolete_replicas -= removed_obsolete_replicas
                    if required_obsolete_replicas == 0:
                        break

        # Calculate the number of replicas that would be allocatable in the next remove-allocate-copy step
        free_cores = node.free_cores + replicas_to_allocate * cc.cores - required_cores
        free_mem = node.free_mem + replicas_to_allocate * cc.mem[0] - required_mem
        allocatable_replicas = min(
            replicas_to_allocate,
            int((free_cores.magnitude + Transition._DELTA) / cc.cores.magnitude),
            int((free_mem.magnitude + Transition._DELTA) / cc.mem[0].magnitude),
        )
        if allocatable_replicas == 0:
            # Recover from backups
            for mod_node in dest_nodes_modified:
                mod_node.free_cores, mod_node.free_mem, mod_node.replicas = dest_nodes_backup[mod_node]
            self._app_perf_surplus = app_perf_surplus_backup
            self._app_perf_increment = app_perf_increment_backup
            self._recycling.obsolete_containers = removed_containers_backup
        else:
            for cc, replicas in allocated_obsolete_replicas:
                available_obsolete_containers[cc] -= replicas
                if available_obsolete_containers[cc] == 0:
                    del available_obsolete_containers[cc]

        return allocatable_replicas, command2

    def _remove_allocate_copy(self, copy_nodes: list[Vmt] = None) -> Command:
        """
        Perform one transition step by removing obsolete containers and nodes, allocating unallocated new
        containers, and copying obsolete containers to other nodes, in preparation for the next transition step.
        :param copy_nodes: A list of nodes where obsolete containers can be copied. All the nodes in the
        current allocation are used when this parameter is not set.
        :return: A command with node removals, containers removal and containers allocation.
        """
        command = Command()

        # Firstly, check if obsolete containers can be removed and update the command
        self.remove_obsolete_containers(command)

        # Allocate container replicas prepared in a previous copy phase of the algorithm. They must be allocatable
        for node, cc, allocatable_replicas in self._allocatable_cs_next_step:
            # Allocate using free computational resources on the same node and removing obsolete
            # containers from the same node if it were necessary
            allocated_replicas = self._remove_allocate(cc, allocatable_replicas, node, command)
            assert allocated_replicas == allocatable_replicas, "Containers must be allocatable"
        self._allocatable_cs_next_step.clear()

        # Firstly, allocate using free computational resources in the same node and removing obsolete
        # containers from the same node if it were necessary
        unalloc_node_cs = self._unalloc_node_cs[:]
        for node, cc, replicas_to_allocate in unalloc_node_cs:
            node_cc_replicas_index = self._unalloc_node_cs.index((node, cc, replicas_to_allocate))
            allocated_replicas = self._remove_allocate(cc, replicas_to_allocate, node, command)
            if allocated_replicas > 0:
                self._unalloc_node_cs.pop(node_cc_replicas_index)
                replicas_to_allocate -= allocated_replicas
                if replicas_to_allocate > 0:
                    self._unalloc_node_cs.insert(node_cc_replicas_index, (node, cc, replicas_to_allocate))
        if len(self._unalloc_node_cs) == 0:
            # Check if obsolete nodes can be removed and update the command
            self.remove_obsolete_nodes(command)
            return command

        # Nodes are sorted by decreasing freeable capacity, leaving empty nodes at the end, sorted by
        # increasing price
        if copy_nodes is None:
            copy_nodes = self._current_alloc
        copy_nodes = self._get_sorted_nodes(copy_nodes)

        # Next, try copying obsolete containers from the node to other nodes (destination nodes),
        # yielding enough application's performance surplus to allocate the replicas of unallocated
        # containers in the next transition step
        unalloc_node_cs = self._unalloc_node_cs[:]
        node_obsolete_containers = {
            node: {cc: replicas for cc, replicas in cc_replicas.items()}
            for node, cc_replicas in self._recycling.obsolete_containers.items()
        }
        for unalloc_node_cc in unalloc_node_cs:
            node, cc, replicas_to_allocate = unalloc_node_cc
            # Nodes before the upgrading may have unallocated containers and no obsolete containers at the same time
            if node in node_obsolete_containers:
                allocatable_replicas, command2 = \
                    self._allocate_copying_obsolete_containers(unalloc_node_cc, node_obsolete_containers[node],
                                                               copy_nodes)
                if allocatable_replicas > 0:
                    # Complete the list of containers allocatable in the next transisition step and remove them
                    # from the list of unallocated containers
                    self._allocatable_cs_next_step.append((node, cc, allocatable_replicas))
                    index = self._unalloc_node_cs.index((node, cc, replicas_to_allocate))
                    replicas_to_allocate -= allocatable_replicas
                    if replicas_to_allocate > 0:
                        self._unalloc_node_cs[index] = (node, cc, replicas_to_allocate)
                    else:
                        self._unalloc_node_cs.pop(index)
                    command.extend(command2)

        # Check if obsolete nodes can be removed and update the command
        self.remove_obsolete_nodes(command)

        return command

    def get_recycling_levels(self) -> tuple[float, float]:
        """
        Get node and container recycling levels for the las transition.
        :return: A tuple with node and container recycling levels.
        """
        return self._recycling.node_recycling_level, self._recycling.container_recycling_level

    def get_recycled_node_pairs(self):
        """
        Get the recycled node pairs.
        :return: Recycled node pairs.
        """
        return self._recycling_vm.recycled_node_pairs

    def get_allocation(self, app_performance: dict[App, RequestsPerTime]) -> list[Vm]:
        """
        Get an allocation to fulfill application performances.
        :param app_performance: Application performances.
        :return: An allocation.
        """
        zero_performances = True
        for _, perf in app_performance.items():
            if perf.magnitude > 0:
                zero_performances = False
                break
        if zero_performances:
            return []
        fcma_problem = Fcma(self._system, workloads=app_performance)
        solving_pars = SolvingPars(speed_level=3)
        solution = fcma_problem.solve(solving_pars)
        allocation = []
        for _, nodes in solution.allocation.items():
            allocation.extend(nodes)
        return allocation

    def _append_command(self, command: Command, append_null_command=False):
        """
        Append a command to the list of commands and updates application's performance surplus.
        :param command: The command to append.
        :append_null_command: Null commands are not appended if this option is not set.
        """
        for app in self._app_perf_increment:
            self._app_perf_surplus[app] += self._app_perf_increment[app]
            self._app_perf_increment[app] = 0
        if not command.is_null() or append_null_command:
            self._commands.append(command)
        allocation_in_upgraded_nodes = any(
            node1 in self._recycling.upgraded_node_pairs and node1.ic == self._recycling.upgraded_node_pairs[node1].ic
            for node1, _, _  in command.allocate_containers
        )
        if self._sync_on_next_alloc_upgraded_nodes and allocation_in_upgraded_nodes:
            command.sync_on_nodes_upgrade = True
            self._sync_on_next_alloc_upgraded_nodes = False

    def _post_process_commands(self):
        """
        Perform post-processing on the comand list:
        - Remove empty commands.
        - For each obsolete node, remove all except its last remove operation. Note that obsolete nodes are not
        actually removed from the allocation.
        - Replace Vmt nodes by Vm nodes.
        """

        # Remove empty commands
        for command in self._commands:
            if command.is_null():
                self._commands.remove(command)

        # Node removal commands are generated by the transition step when all the containers of an obsolete
        # node are removed. However, the nodes can be useful in future to help in the transition of recycled
        # nodes, so they could be used after a removal command. Therefore, only the last removal command
        # for a node must remain, and any previous removal command must be deleted from the command list
        last_node_removal_command = defaultdict(lambda: Command())
        for command in self._commands:
            if len(command.remove_nodes) > 0:
                for node_to_remove in command.remove_nodes:
                    previous_command = last_node_removal_command[node_to_remove]
                    if not previous_command.is_null():
                        previous_command.remove_nodes.remove(node_to_remove)
                        if previous_command.is_null():
                            self._commands.remove(previous_command)
                    last_node_removal_command[node_to_remove] = command
        for node_to_remove in last_node_removal_command:
            self._current_alloc.remove(node_to_remove)

        # Replace Vmt nodes by nodes Vm nodes in the commands
        for command in self._commands:
            command.vmt_to_vm()

    @staticmethod
    def get_transition_time(commands: list[Command], timing_args: TimedOps.TimingArgs) -> int:
        """
        Get the transition time from a list of commands.
        :param commands: A list of commands.
        :param timing_args: Creation/removal times of nodes and containers.
        :return: The transition time.
        """
        transition_time = 0
        last_node_removal_time = -1
        for command in commands:
            if len(command.create_nodes) > 0:
                assert commands.index(command) == 0, "Nodes must be created in the first command"
            if command.sync_on_nodes_creation and len(commands[0].create_nodes) > 0:
                assert commands.index(command) > 0, "Invalid sync on first command"
                transition_time = max(transition_time, timing_args.node_creation_time)
            if len(command.remove_containers) > 0:
                transition_time += timing_args.container_removal_time
            if len(command.remove_nodes) > 0:
                last_node_removal_time = transition_time + timing_args.node_removal_time
            if len(command.allocate_containers) > 0:
                transition_time += timing_args.container_creation_time
        return max(transition_time, last_node_removal_time)

    def allocation_loop(self, max_time: int) -> int:
        """
        Repeat a remove-allocate-copy operation adding the commands to the command list.
        The method returns when total time required to perform execute the commands is higher than or equal to
        a maximum time limit or no more allocation are possible.
        :param max_time:  Maximum time.
        :return: The time required to execute the added commands.
        """
        elapsed_time = 0
        while elapsed_time < max_time:
            command = self._remove_allocate_copy()
            self._append_command(command)
            elapsed_time += command.get_container_command_time(self._timing_args)
            if elapsed_time >= max_time or len(command.allocate_containers) == 0:
                break
        return elapsed_time

    def _complete_allocation_in_temporal_nodes(self, create_nodes_command: Command,
                                               allocate_new_nodes_command: Command):
        """
        Complete the allocation using temporal nodes.
        :param create_nodes_command: Command where temporal nodes creations are appended.
        :param allocate_new_nodes_command: Command where container allocations in temporal nodes are appended.
        """

        # Add a dummy node with enough capacity to allocate any number of containers
        dummy_node = Vmt(Vm(self._current_alloc[0].ic, ignore_ic_index=True))
        dummy_node.free_cores *= 10E12
        dummy_node.free_mem *= 10E12
        self._current_alloc.append(dummy_node)

        # Calculate application's performance provided by temporal nodes to allocate the
        # remaining containers
        copy_nodes = None if self._time_limit > 0 else [dummy_node]
        command = self._remove_allocate_copy(copy_nodes)
        zero_rps = RequestsPerTime("0 rps")
        tmp_app_perf = {app: zero_rps for app in self._app_perf_surplus}
        for node, cc, replicas in command.allocate_containers:
            if node == dummy_node:
                tmp_app_perf[cc.app] += cc.perf * replicas

        # Get an allocation for application's performance on temporal nodes
        tmp_nodes = [Vmt(node) for node in self.get_allocation(tmp_app_perf)]
        for tmp_node_index in range(len(tmp_nodes)):
            # Change the id of temporal nodes to negative values to be easily identified
            tmp_nodes[tmp_node_index].id = -(tmp_node_index + 1)
            tmp_nodes[tmp_node_index].vm.id = -(tmp_node_index + 1)
        create_nodes_command.create_nodes.extend(tmp_nodes)
        containers_in_tmp_nodes = [
            (node, cc, replicas)
            for node in tmp_nodes for cc, replicas in node.replicas.items()
        ]

        # Move allocations in the command from the dummy node to the temporal nodes
        command.allocate_containers = [
            (node, cc, replicas)
            for node, cc, replicas in command.allocate_containers[:]
            if node != dummy_node
        ]
        allocate_new_nodes_command.allocate_containers.extend(containers_in_tmp_nodes)

        # Remove obsolete containers and nodes from the recycling object
        if dummy_node in self._recycling.obsolete_containers:
            del self._recycling.obsolete_containers[dummy_node]
        for tmp_node in tmp_nodes:
            self._recycling.obsolete_containers[tmp_node] = dict(tmp_node.replicas)
            self._recycling.obsolete_nodes.append(tmp_node)

        # Replace the dummy node with temporal nodes
        self._current_alloc.remove(dummy_node)
        self._current_alloc.extend(tmp_nodes)

        self._append_command(command)

    def _transition_init(self, initial_alloc: Allocation, final_alloc: Allocation) -> bool:
        """
        Initialize the trasition between two allocations.
        :param initial_alloc: Initial allocation.
        :param final_alloc: Final allocation.
        :return: True when a transition is necessary to go from the initial allocation to the final allocation.
        """
        self._commands = []

        # Now it is time to transition from the initial allocation to the final allocation, so
        # start with the initial allocation. All the nodes are changed to the Vmt format.
        # Transition is performed on a copy of initial_alloc nodes in Vmt format. Transition modifies
        # Vmt format nodes, while preserving their initial state, stored in initial_alloc.
        self._current_alloc = [Vmt(node) for node in initial_alloc]
        final_alloc_vmt = [Vmt(node) for node in final_alloc]
        vm_to_vmt = dict(zip(initial_alloc, self._current_alloc)) | \
                    dict(zip(final_alloc, final_alloc_vmt))

        # Calculate recycled node pairs, new nodes, nodes to remove, recycled containers, new containers
        # and containers to remove when transitioning from the initial allocation to the final allocation
        self._recycling_vm = Recycling(initial_alloc, final_alloc, self._hot_node_scale_up)
        self._recycling = RecyclingVmt(self._recycling_vm, vm_to_vmt)

        # Check whether the initial allocation is identical to the final allocation
        if get_allocation_signature(self._current_alloc) == get_allocation_signature(final_alloc_vmt):
            return False

        # Calculate the minimum application performance during the transition
        min_perf, _ = get_min_max_perf(initial_alloc, final_alloc)

        # List of containers in new nodes that have not been allocated yet
        self._unallocated_containers_in_new_nodes = [
            (n, cc, replicas)
            for n, cc_replicas in self._recycling.new_containers.items()
            if n in self._recycling.new_nodes
            for cc, replicas in cc_replicas.items()
        ]

        # Initialize the remove-allocate-copy algorithm, which will be executed iteratively during the transition
        self._remove_allocate_copy_init(min_perf)

        # The first command allocating containers in upgraded nodes must be synchronized to
        # the upgrading of nodes
        self._sync_on_next_alloc_upgraded_nodes = True

        return True

    def calculate_sync(self, initial_alloc: Allocation, final_alloc: Allocation) -> tuple[list[Command], int]:
        """
        Calculate a synchronous transition from the initial allocation to the final allocation, while fulfilling the
        application's minimum performance requirement. It is based on the remove-allocate-copy algorithm.

        :param initial_alloc: Initial allocation.
        :param final_alloc: Final allocation.
        :return: A list of commands and the time to perform the transition.
        """

        # Initialize the transition and check if a transition is necessary
        if not self._transition_init(initial_alloc, final_alloc):
            return [], 0

        # Node creation and node upgrade operations are the most time-consuming. Therefore, they are the first
        # operations to be performed. This command may be empty and extended later to include the creation of
        # temporal nodes
        upgrade_node_info = [(n1, n2.ic) for n1, n2 in self._recycling.upgraded_node_pairs.items()]
        create_upgrade_nodes_command = Command(create_nodes=self._recycling.new_nodes, upgrade_nodes=upgrade_node_info)
        self._append_command(create_upgrade_nodes_command, append_null_command=True)

        # Allocation loop until node upgrading completes
        elapsed_time = 0
        if self._time_limit > 0:
            elapsed_time = self.allocation_loop(self._timing_args.hot_node_scale_up_time)

        # Update the elapsed time and current allocation after completing the node upgrading
        elapsed_time = max(elapsed_time, self._timing_args.hot_node_scale_up_time)
        for initial_node, final_node in self._recycling.upgraded_node_pairs.items():
            initial_node.upgrade(final_node)

        # Allocation loop until node creation completes
        if self._time_limit > 0:
            max_time = self._timing_args.node_creation_time - self._timing_args.hot_node_scale_up_time
            elapsed_time += self.allocation_loop(max_time)

        # Update the elapsed time and current allocation after completing the node creation
        elapsed_time = max(elapsed_time, self._timing_args.node_creation_time)
        self._current_alloc.extend(self._recycling.new_nodes)

        # Allocate new containers in new nodes. The corresponding command may be empty if there
        # are no new nodes in the final allocation. This command can be extended later to include
        # containers in temporal nodes.
        allocate_in_new_nodes_command = Command(allocate_containers=self._unallocated_containers_in_new_nodes[:])
        allocate_in_new_nodes_command.sync_on_nodes_creation = True
        for _, cc, replicas in self._unallocated_containers_in_new_nodes:
            self._app_perf_increment[cc.app] += replicas * cc.perf
            self._app_unalloc_perf[cc.app] -= replicas * cc.perf
            assert self._app_unalloc_perf[cc.app].magnitude >= - Transition._DELTA, "Invalid performance"
        self._unallocated_containers_in_new_nodes.clear()
        self._append_command(allocate_in_new_nodes_command, append_null_command=True)

        # Allocation loop until the time limit
        if elapsed_time < self._time_limit:
            self.allocation_loop(self._time_limit - elapsed_time)

        # If there are still unallocated containers in recycled nodes
        if len(self._allocatable_cs_next_step) > 0 or len(self._unalloc_node_cs) > 0:
            if len(self._unalloc_node_cs) == 0 and self._time_limit > 0:
                # Extend a little the transition time instead of creating temporal nodes
                self._append_command(self._remove_allocate_copy())
            # If new nodes are not enough to complete the transition of recycled nodes, temporal nodes are required
            else:
                self._complete_allocation_in_temporal_nodes(create_upgrade_nodes_command, allocate_in_new_nodes_command)

        # Two remove-allocate-copy steps may be necessary to complete the transition. The first
        # command to remove obsolete containers and next allocate the remaining new containers in recycled nodes.
        # The second command to remove containers from the temporal nodes and next the temporal nodes
        first_command = self._remove_allocate_copy()
        if not first_command.is_null():
            self._append_command(first_command)
        second_command = self._remove_allocate_copy()
        if not second_command.is_null():
            self._append_command(second_command)

        # Post-processing operations to obtain the final command list
        self._post_process_commands()

        # Check whether the commands implement a valid transition between the initial and the final allocations
        assert Transition.check_transition(initial_alloc, final_alloc, self._commands), "Invalid transition"

        return self._commands, Transition.get_transition_time(self._commands, self._timing_args)

    def calculate_async(self, initial_alloc: Allocation, final_alloc: Allocation) -> tuple[list[Command], int]:
        """
        Calculate an asynchronous transition from the initial allocation to the final allocation,
        while fulfilling the application's minimum performance requirement.

        :param initial_alloc: Initial allocation.
        :param final_alloc: Final allocation.
        :return: A list of commands and the worst-case time to perform the transition.
        """
        sync_commands, worst_case_time = self.calculate_sync(initial_alloc, final_alloc)[:]
        for command in sync_commands[:]:
            if command.sync_on_nodes_creation:
                sync_commands[0].allocate_containers.extend(command.allocate_containers)
                command.allocate_containers.clear()
                command.sync_on_nodes_creation = False
                if command.is_null():
                    sync_commands.remove(command)
                break
        return sync_commands, worst_case_time

    def _debug_perf_surplus_balance(self) -> dict[Vmt, RequestsPerTime]:
        """
        Get the performance balance. The performance balance must be constant during the transition.
        This function is used for debugging purposes, which can be called in the debugger command line.
        """
        balance = dict(self._app_perf_surplus)
        for app, perf in self._app_perf_increment.items():
            balance[app] += perf
        for node, cc, replicas in self._unalloc_node_cs + self._allocatable_cs_next_step + \
                                  self._unallocated_containers_in_new_nodes:
            balance[cc.app] += cc.perf * replicas
        for node, cc_replicas in self._recycling.obsolete_containers.items():
            for cc, replicas in cc_replicas.items():
                balance[cc.app] -= cc.perf * replicas
        return balance

    @staticmethod
    def check_transition(initial_alloc: Allocation, final_alloc: Allocation, commands: list[Command]) -> bool:
        """
        Check the transition between the initial and final allocations.
        :param initial_alloc: Initial allocation
        :param final_alloc: Final allocatio
        :param commands: List of commands to transition.
        :return: True if commands perform the required transition.
        :raises  ValueError: If some command is invalid.
        """
        min_perf, _ = get_min_max_perf(initial_alloc, final_alloc)
        initial_alloc_vmt = [Vmt(node) for node in initial_alloc]
        final_alloc_vmt = [Vmt(node) for node in final_alloc]
        vm_to_vmt = dict(zip(initial_alloc + final_alloc, initial_alloc_vmt + final_alloc_vmt))
        app_perf_surplus = get_app_perf_surplus(min_perf, initial_alloc_vmt)

        command_index = 0
        for command in commands:
            command_index += 1
            app_perf_increment = defaultdict(lambda: 0)

            # Remove container commands
            obsolete_cc_replicas = []
            for node, cc, replicas in command.remove_containers:
                obsolete_cc_replicas.append((node, cc))
                op_str = f'Command #{command_index}. Remove containers ({node}, {cc}, {replicas})'
                if node not in vm_to_vmt:
                    raise ValueError(f'{op_str} -> Invalid node: {node}')
                node_vmt = vm_to_vmt[node]
                node_vmt.replicas[cc] -= replicas
                if node_vmt.replicas[cc] == 0:
                    del node_vmt.replicas[cc]
                elif node_vmt.replicas[cc] < 0:
                    raise ValueError(f'{op_str} -> Invalid container removal. Replicas < 0')
                app_perf_surplus[cc.app] -= cc.perf * replicas
                if app_perf_surplus[cc.app].magnitude < -Transition._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container removal. app surplus < 0')
                node_vmt.free_cores += cc.cores * replicas
                if (node_vmt.free_cores - node_vmt.ic.cores).magnitude > Transition._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container removal. Too many cores')
                node_vmt.free_mem += cc.mem[0] * replicas
                if (node_vmt.free_mem - node_vmt.ic.mem).magnitude > Transition._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container removal. Too many mem')

            # Add node commands
            for node in command.create_nodes:
                node_vmt = Vmt(node)
                vm_to_vmt[node] = node_vmt
                op_str = f'Command #{command_index}. Add node ({node})'
                if node_vmt in initial_alloc_vmt:
                    raise ValueError(f'{op_str} -> Duplicated node')
                initial_alloc_vmt.append(node_vmt.clear())

            # Add container commands
            for node, cc, replicas in command.allocate_containers:
                op_str = f'Command #{command_index}. Allocate containers ({node}, {cc}, {replicas})'
                if (node, cc) in obsolete_cc_replicas:
                    raise ValueError(f'{op_str} -> Removing and adding identical containers in same command')
                if node not in vm_to_vmt:
                    raise ValueError(f'{op_str} -> Invalid node: {node}')
                node_vmt = vm_to_vmt[node]
                node_vmt.replicas[cc] += replicas
                app_perf_increment[cc.app] += cc.perf * replicas
                node_vmt.free_cores -= cc.cores * replicas
                if node_vmt.free_cores.magnitude < -Transition._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container addition. Not enough cores are available')
                node_vmt.free_mem -= cc.mem[0] * replicas
                if node_vmt.free_mem.magnitude < -Transition._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container removal. Not enough memory is available')

            # Remove node commands
            for node in command.remove_nodes:
                node_vmt = vm_to_vmt[node]
                op_str = f'Command #{command_index}. Remove node ({node})'
                if node_vmt not in initial_alloc_vmt:
                    raise ValueError(f'{op_str} -> Invalid node')
                for _ in node_vmt.replicas:
                        raise ValueError(f'{op_str} -> Node allocates containers')
                initial_alloc_vmt.remove(node_vmt)

            # Upgrade node commands
            for initial_node, final_node_ic in command.upgrade_nodes:
                initial_node_vmt = vm_to_vmt[initial_node]
                op_str = f'Command #{command_index}. Upgrade node ({initial_node} to instance class {final_node_ic})'
                if initial_node_vmt not in initial_alloc_vmt:
                    raise ValueError(f'{op_str} -> Invalid node')
                initial_node_vmt.free_cores = initial_node_vmt.free_cores + \
                                              final_node_ic.cores - initial_node_vmt.ic.cores
                initial_node_vmt.free_mem = initial_node_vmt.free_mem + \
                                            final_node_ic.mem - initial_node_vmt.ic.mem
                initial_node_vmt.ic = final_node_ic

            # Update application's performance surplus
            for app in app_perf_increment:
                app_perf_surplus[app] += app_perf_increment[app]
                app_perf_increment[app] = RequestsPerTime("0 req/s")

        # Compare initial and final allocations
        initial_alloc_signature = get_allocation_signature(initial_alloc_vmt)
        final_alloc_signature = get_allocation_signature(final_alloc_vmt)
        if initial_alloc_signature != final_alloc_signature:
            only_in_initial_alloc = initial_alloc_signature - final_alloc_signature
            only_in_final_alloc = final_alloc_signature - initial_alloc_signature
            if len(only_in_initial_alloc) > 0:
                print("* Only in initial allocation:")
                for item in only_in_initial_alloc.elements():
                    print(f'- {loads(item)}')
            if len(only_in_final_alloc) > 0:
                print("* Only in final allocation:")
                for item in only_in_final_alloc.elements():
                    print(f'- {loads(item)}')
            return False

        return True
