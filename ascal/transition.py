"""
Implement synchronous transitions between two allocations as a list of commands.
A synchronous transition defines predefined times to start node/container creations/removals, assuming
fixed durations to perform these operations. A command starts when the previous command completes
and the node creation/upgrade time has elapsed if the command flag 
"sync_on_nodes_creation"/"sync_on_nodes_upgrade" is set. 
A command completes after finishing all its container removals, allocations and scales. 
Node creations and removals execute in background.
Within a command operations are executed following these restrictions:

        - Node creations and upgrades. Start inmediately with the command. Only the first command can include them.
        - Container removals and container scale-downs. Start inmediately with the command.
        - Node removals. Start after completing all the container removals in the command.
        - Container allocations and container scale-ups. Start after completing all the container removals in the 
          command.

While useful for theoretical analysis, synchronous transitions are too restrictive in practice,
as creation and removal times are not fixed, and assuming their worst-case durations can be overly pessimistic.
Nevertheless, an asynchronous transition can be derived from a synchronous transition with small changes:

- Move container allocations in created/upgraded nodes to the first command, i.e, to the node creation/upgrade 
command. Although these container allocations and node creations/upgrades are triggered at the same time, 
allocations are suspended until completing the node creations/upgrades. These containers can be easily 
identified by a set command flag "sync_on_nodes_creation"/"sync_on_nodes_upgrade".

- Container removals, container scale-downs, node removals and container allocations in a command are triggered 
with the command. However, any individual node removal is suspended until its containers have been removed. 
Similarly, any individual container allocation or scale-up is suspended until the destination node has enough 
computational resources.

Three transition algorithms are implemented:
- Baseline transition. It performs the transition in 4 steps: 
    1) Create all the nodes in the final allocation. 
    2) Allocate all the containers in these created nodes, 
    3) Remove all the containers in the initial allocation 
    4) Remove all the nodes in the initial allocation.
- RBT2. It is the simplest variant based on recycling. It does not perform remove-alocate-copy steps while new 
nodes are created, or recycled nodes are upgraded.
- RBT1. The same as RBT2, but it performs remove-allocate-copy steps while new nodes are created, or recycled 
nodes are upgraded. 

Limitations of the current implementation:
- All the nodes belong to instance classes of the same family.
- All the containers of a given application are configured with the same amount of memory. Nevertheless, CPU 
and performance parameters can be different for different container classes of the same application.

"""

from enum import Enum
from math import ceil, floor
from json import loads
from collections import defaultdict
from dataclasses import dataclass, field, replace
from abc import ABC, abstractmethod
from fcma import Fcma, SolvingPars, System, Allocation, App, Vm, ContainerClass, RequestsPerTime, InstanceClass
from ascal.timedops import TimedOps
from ascal.recycling import Recycling
from ascal.helper import (
    get_min_max_perf,
    Vmt,
    RecyclingVmt,
    get_vmt_allocation_signature,
    get_app_perf_surplus
)


class TransitionAlgorithm(Enum):
    """
    Different algorithms for calculating the transition between two allocations.
    """
    RBT1 = 1          # Recycling-based Transition algorithm 1
    RBT2 = 2          # Recycling-based Transition algorithm 2
    RBT  = RBT2       # Default RBT algorithm
    BASELINE = 3      # Baseline transition algorithm


@dataclass
class Command:
    """
    The transition between an initial allocation and a final allocation is a sequence of commands.
    Each command can include several types of operations with nodes and containers.
    """
    allocate_containers: list[tuple[Vm|Vmt, ContainerClass, int]] = field(default_factory=list)
    remove_containers: list[tuple[Vm|Vmt, ContainerClass, int]] = field(default_factory=list)
    scale_containers: list[tuple[Vm|Vmt, ContainerClass, int, float]] = field(default_factory=list)
    remove_nodes: list[Vm|Vmt] = field(default_factory=list)
    # Only the first command can create/upgrade nodes
    create_nodes: list[Vm|Vmt] = field(default_factory=list)
    upgrade_nodes: list[tuple[Vm|Vmt, InstanceClass]] = field(default_factory=list)
    # Some commands must delay execution until completing the node creations/upgrades in the first command    
    sync_on_nodes_creation: bool = False
    sync_on_nodes_upgrade: bool = False

    def __str__(self) -> str:
        """
        String representation of the command
        """
        alloc_c_str = 'allocate->[' + \
            ', '.join(f'({str(v)}, {str(c)}): {r}' for v, c, r in self.allocate_containers) + ']'
        if len(self.allocate_containers) == 0:
            alloc_c_str = ''
        remove_c_str = 'remove->[' + \
            ', '.join(f'({str(v)}, {str(c)}): {r}' for v, c, r in self.remove_containers) + ']'
        if len(self.remove_containers) == 0:
            remove_c_str = ''
        scale_c_str = 'scale->[' + \
            str([f'({str(v)}, {str(c)}, {m:1.2f}): {r}' for v, c, r, m in self.scale_containers]) + ']'
        if len(self.scale_containers) == 0:
            scale_c_str = ''
        create_n_str = 'create->[' + ', '.join([str(v) for v in self.create_nodes]) + ']'
        if len(self.create_nodes) == 0:
            create_n_str = ''
        remove_n_str = 'remove->[' + str([str(v) for v in self.remove_nodes]) + ']'
        if len(self.remove_nodes) == 0:
            remove_n_str = ''
        upgrade_n_str = 'upgrade->[' + str([f'({str(v)}), {str(i)})' for v, i in self.upgrade_nodes]) + ']'
        if len(self.upgrade_nodes) == 0:
            upgrade_n_str = ''
        sync_n_c = 'sync_on_n_creation' if self.sync_on_nodes_creation else ''         
        sync_n_u = 'sync_on_n_upgrade' if self.sync_on_nodes_upgrade else ''
        items = [alloc_c_str, remove_c_str, scale_c_str, create_n_str, remove_n_str, upgrade_n_str, sync_n_c, sync_n_u]
        return ' || '.join(s for s in items if s)
    
    def extend(self, command: 'Command'):
        """
        Extend the command operations with a new command, appending all the container and node operations.
        :param command: Command with the extensions.
        """
        self.allocate_containers.extend(command.allocate_containers)
        self.remove_containers.extend(command.remove_containers)
        self.scale_containers.extend(command.scale_containers)
        self.create_nodes.extend(command.create_nodes)
        self.remove_nodes.extend(command.remove_nodes)
        self.upgrade_nodes.extend(command.upgrade_nodes)

    def vmt_to_vm(self):
        """
        Replace Vmt formatted nodes in the command by Vm nodes.
        """
        self.allocate_containers = [(node.vm, cc, replicas) for node, cc, replicas in self.allocate_containers]
        self.remove_containers = [(node.vm, cc, replicas) for node, cc, replicas in self.remove_containers]
        self.scale_containers = [
            (node.vm, cc, replicas, multiplier) 
            for node, cc, replicas, multiplier in self.scale_containers
        ]
        self.create_nodes = [node.vm for node in self.create_nodes]
        self.remove_nodes = [node.vm for node in self.remove_nodes]
        self.upgrade_nodes = [(initial_node.vm, final_ic) for initial_node, final_ic in self.upgrade_nodes]

    def is_null(self) -> bool:
        """
        Check if the command is null.
        :return: True when the command is null.
        """
        return (len(self.allocate_containers) == 0 and len(self.remove_containers) == 0 and
                len(self.scale_containers) == 0 and len(self.create_nodes) == 0 and 
                len(self.remove_nodes) == 0 and len(self.upgrade_nodes) == 0)

    def get_container_command_time(self, timing_args: TimedOps.TimingArgs) -> int:
        """
        Get the time required to complete a command ignoring node creation/removal operations. It assumes that
        container scale-up and scale-down operations take the same or less time than container creation and removal, 
        respectively.
        :param timing_args: Times for creation/removal of nodes/containers.
        :return: The time to complete the command.
        """
        # Check if the are scale-up and scale-down container operations in the command
        container_scale_up = False
        container_scale_down = False
        for _, _, _, multiplier in self.scale_containers:
            if multiplier > 1:
                container_scale_up = True
            if multiplier < 1:
                container_scale_down = True
        container_command_time = 0

        # Sum up the time required to perform container operations
        if len(self.remove_containers) > 0:
            container_command_time += timing_args.container_removal_time
        elif container_scale_down:
            container_command_time += timing_args.hot_container_scale_down_time
        if len(self.allocate_containers) > 0:
            container_command_time += timing_args.container_creation_time
        elif container_scale_up:
            container_command_time += timing_args.hot_container_scale_up_time
        return container_command_time
    
    def simplification(self):
        """
        Labels in container classes are removed, and after that, performs the following simplifications:
        - Translate container creations/removals into container scale-ups/scale-downs when there are
        containers with the same id and hot scale of containers is enabled.
        - Sum up replicas of the same container class in the same node.
        - Remove allocations and removals of the same containers.
        """
        # Remove labels in container classes
        self.allocate_containers = [
            (node, replace(cc, label=""), replicas) 
            for node, cc, replicas in self.allocate_containers
        ] 
        self.remove_containers = [
            (node, replace(cc, label=""), replicas) 
            for node, cc, replicas in self.remove_containers
        ]

        # Sum up replicas of the same container class in the same node. The same for removals
        allocs = defaultdict(int)
        for node, cc, replicas in self.allocate_containers:
            allocs[(node, cc)] += replicas
        removals = defaultdict(int)
        for node, cc, replicas in self.remove_containers:
            removals[(node, cc)] += replicas

        # Remove common allocations and removals       
        for node, cc in allocs:
            if (node, cc) in removals:
                common_replicas = min(allocs[(node, cc)], removals[(node, cc)])
                allocs[(node, cc)] -= common_replicas
                removals[(node, cc)] -= common_replicas
        self.allocate_containers = [
            (node, cc, allocs[(node, cc)]) 
            for (node, cc) in allocs 
            if allocs[(node, cc)] > 0
        ]
        self.remove_containers = [
            (node, cc, removals[(node, cc)]) 
            for (node, cc) in removals 
            if removals[(node, cc)] > 0
        ]

    def replace_nodes(self, node_pairs: dict[Vm, Vm]) -> 'Command':
        """
        Replace the nodes in the command by their counterparts in the node pairs.
        :param node_pairs: A dictionary with the final node for each initial node.
        :return: A new command with the replacement.
        """
        new_command = Command()
        new_command.sync_on_nodes_creation = self.sync_on_nodes_creation
        new_command.sync_on_nodes_upgrade = self.sync_on_nodes_upgrade
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
        for node, cc, replicas, multiplier in self.scale_containers:
            if node in node_pairs:
                new_command.scale_containers.append((node_pairs[node], cc, replicas, multiplier))
            else:
                new_command.scale_up_containers.append((node, cc, replicas, multiplier))
        for node in self.create_nodes:
            if node in node_pairs:
                new_command.create_nodes.append(node_pairs[node])
            else:
                new_command.create_nodes.append(node)
        for node1, new_ic in self.upgrade_nodes:
            if node1 in node_pairs:
                new_command.upgrade_nodes.append((node_pairs[node1], new_ic))
            else:
                new_command.upgrade_nodes.append((node1, new_ic))
        for node in self.remove_nodes:
            if node in node_pairs:
                new_command.remove_nodes.append(node_pairs[node])
            else:
                new_command.remove_nodes.append(node)
        return new_command

class Transition(ABC):
    """
    Abstract class for transition algorithms between initial and final allocations for a given system.
    There are two types of transitions: synchronous and asynchronous.
    """
    # Constant used to deal with numerical approximations
    _DELTA = 0.000001

    @abstractmethod
    def get_worst_case_transition_time(self) -> int:
        """
        Get the worst-case transition time.
        :return: The worst-case transition time.
        """
        pass
        
    @abstractmethod
    def get_recycling_levels(self) -> tuple[float, float]:
        """
        Get node and container recycling levels for the last transition.
        :return: A tuple with node and container recycling levels.
        """
        return self._recycling.node_recycling_level, self._recycling.container_recycling_level

    @abstractmethod
    def get_recycled_node_pairs(self) -> dict[Vm, Vm]|None:
        """
        Get the recycled node pairs.
        :return: Recycled node pairs or None if the transition algorithm does not use recycling.
        """
        pass

    @abstractmethod    
    def calculate_sync(self, initial_alloc: Allocation, final_alloc: Allocation) -> tuple[list[Command], int]:
        """
        Calculate a synchronous transition from the initial allocation to the final allocation.

        :param initial_alloc: Initial allocation.
        :param final_alloc: Final allocation.
        :return: A list of commands and the time to perform the transition.
        """
        pass

    def calculate_async(self, initial_alloc: Allocation, final_alloc: Allocation) -> tuple[list[Command], int]:
        """
        Calculate an asynchronous transition from the initial allocation to the final allocation,
        while fulfilling the application's minimum performance requirement.

        :param initial_alloc: Initial allocation.
        :param final_alloc: Final allocation.
        :return: A tuple with thelist of commands and the worst-case time to perform the transition.
        """
        sync_commands, worst_case_time = self.calculate_sync(initial_alloc, final_alloc)[:]
        for command in sync_commands[:]:
            if command.sync_on_nodes_creation or command.sync_on_nodes_upgrade:
                sync_commands[0].allocate_containers.extend(command.allocate_containers)
                command.allocate_containers.clear()
                command.sync_on_nodes_creation = False
                command.sync_on_nodes_upgrade = False
                if command.is_null():
                    sync_commands.remove(command)
        return sync_commands, worst_case_time

    def check_transition(self, initial_alloc: Allocation, final_alloc: Allocation, commands: list[Command]) -> bool:
        """
        Check the transition between the initial and final allocations.
        :param initial_alloc: Initial allocation.
        :param final_alloc: Final allocation.
        :param commands: List of commands to transition.
        :return: True if commands perform the required transition.
        :raises ValueError: If some command is invalid.
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
                if app_perf_surplus[cc.app].magnitude < -TransitionRBT._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container removal. app surplus < 0')
                node_vmt.free_cores += cc.cores * replicas
                if (node_vmt.free_cores - node_vmt.ic.cores).magnitude > TransitionRBT._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container removal. Too many cores')
                node_vmt.free_mem += cc.memv * replicas
                if (node_vmt.free_mem - node_vmt.ic.mem).magnitude > TransitionRBT._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container removal. Too many mem')
                
            # Container scale-up and scale-down commands
            for node, cc, replicas, multiplier in command.scale_containers:
                op_str = f'Command #{command_index}. Scale containers ({node}, {cc}, {replicas}, {multiplier})'
                if node not in vm_to_vmt:
                    raise ValueError(f'{op_str} -> Invalid node: {node}')
                node_vmt = vm_to_vmt[node]
                if node_vmt.replicas[cc] < replicas:
                    raise ValueError(f'{op_str} -> Invalid container scale. Replicas to scale > allocated replicas')
                # Remove scaled replicas
                node_vmt.replicas[cc] -= replicas
                if node_vmt.replicas[cc] == 0:
                    del node_vmt.replicas[cc]
                # Add the replicas after the scaling. Note that other replicas of the same container class
                # may be allocated
                scaled_cc = cc * multiplier
                close_cc_found = False # Dealing with floats may introduce runding errors
                for other_cc in node_vmt.replicas:
                    if scaled_cc.almost_equal(other_cc):
                        node_vmt.replicas[other_cc] += replicas
                        close_cc_found = True
                        break
                if not close_cc_found:
                    node_vmt.replicas[scaled_cc] = replicas
                # Update the node free cores
                node_vmt.free_cores += cc.cores * replicas * (1 - multiplier)
                if multiplier > 1:
                    # The performance increment is delayed until the command termination
                    app_perf_increment[cc.app] += cc.perf * replicas * (multiplier - 1)
                    if node_vmt.free_cores.magnitude < -TransitionRBT._DELTA:
                        raise ValueError(f'{op_str} -> Invalid container scale-up. Not enough cores are available')
                else:
                    # The performance surplus is inmediately reduced
                    app_perf_surplus[cc.app] -= cc.perf * replicas * (1 - multiplier)
                    if app_perf_surplus[cc.app].magnitude < -Transition._DELTA:
                        raise ValueError(f'{op_str} -> Invalid container scale down. app surplus < 0')
                    if (node_vmt.free_cores - node_vmt.ic.cores).magnitude > TransitionRBT._DELTA:
                        raise ValueError(f'{op_str} -> Invalid container scale-down. Too many cores')
                # Memory is no checked as it does not change in scale operations

            # Add node commands
            for node in command.create_nodes:
                node_vmt = Vmt(node)
                vm_to_vmt[node] = node_vmt
                op_str = f'Command #{command_index}. Add node ({node})'
                if node_vmt in initial_alloc_vmt:
                    raise ValueError(f'{op_str} -> Duplicated node')
                initial_alloc_vmt.append(node_vmt.clear())

            # Allocate container commands
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
                if node_vmt.free_cores.magnitude < -TransitionRBT._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container addition. Not enough cores are available')
                node_vmt.free_mem -= cc.memv * replicas
                if node_vmt.free_mem.magnitude < -TransitionRBT._DELTA:
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

            # Update application's performance surplus at the command termination
            for app in app_perf_increment:
                app_perf_surplus[app] += app_perf_increment[app]
                app_perf_increment[app] = RequestsPerTime("0 req/s")

        # Compare initial and final allocations
        initial_alloc_signature = get_vmt_allocation_signature(initial_alloc_vmt)
        final_alloc_signature = get_vmt_allocation_signature(final_alloc_vmt)
        if initial_alloc_signature != final_alloc_signature:
            only_in_initial_alloc = initial_alloc_signature - final_alloc_signature
            only_in_final_alloc = final_alloc_signature - initial_alloc_signature
            if len(only_in_initial_alloc) > 0:
                print("* Only after transitioning the initial allocation:")
                for item in only_in_initial_alloc.elements():
                    print(f'- {loads(item)}')
            if len(only_in_final_alloc) > 0:
                print("* Only in final allocation:")
                for item in only_in_final_alloc.elements():
                    print(f'- {loads(item)}')
            return False

        return True

    @staticmethod
    def get_transition_time(commands: list[Command], timing_args: TimedOps.TimingArgs) -> int:
        """
        Get the transition time from a list of commands. Node upgrade times are assumed to be lower 
        than or equal to node creation times, and container scale up/down times are assumed to be lower 
        than or equal to container creation/removal times.
        :param commands: A list of commands.
        :param timing_args: Creation/removal/scale times of nodes and containers.
        :return: The transition time.
        """

        transition_time = 0
        last_node_removal_time = -1 
        for command in commands:
            # Check if the are scale-up and scale-down container operations in the command
            container_scale_up = False
            container_scale_down = False
            for _, _, _, multiplier in command.scale_containers:
                if multiplier > 1:
                    container_scale_up = True
                if multiplier < 1:
                    container_scale_down = True
            if len(command.create_nodes) > 0:
                assert commands.index(command) == 0, "Nodes must be created in the first command"
            if len(command.upgrade_nodes) > 0:
                assert commands.index(command) == 0, "Nodes must be upgraded in the first command"
            if command.sync_on_nodes_creation:
                assert len(commands[0].create_nodes) > 0, "Invalid sync without nodes creation in the first command"
                assert commands.index(command) > 0, "Invalid sync on first command"
                # Wait until the nodes are created
                transition_time = max(transition_time, timing_args.node_creation_time)
            if command.sync_on_nodes_upgrade:
                assert len(commands[0].upgrade_nodes) > 0, "Invalid sync without nodes upgrade in the first command"
                assert commands.index(command) > 0, "Invalid sync on first command"
                # Wait until nodes are upgraded
                transition_time = max(transition_time, timing_args.hot_node_scale_up_time)
            # Add the maximum time required to perform container removals or container scale-downs
            incremental_time = 0
            if container_scale_down:
                incremental_time = timing_args.hot_container_scale_down_time
            if len(command.remove_containers) > 0:
                incremental_time = timing_args.container_removal_time
            transition_time += incremental_time
            if len(command.remove_nodes) > 0:
                # Nodes can be removed in background. Calculate the latest time when a node removal finishes
                last_node_removal_time = transition_time + timing_args.node_removal_time
            # Add the maximum time required to perform container creations or container scale-ups
            incremental_time = 0
            if container_scale_up:
                incremental_time = timing_args.hot_container_scale_up_time
            if len(command.allocate_containers) > 0:
                incremental_time = timing_args.container_creation_time
            transition_time += incremental_time

        return max(transition_time, last_node_removal_time)


class TransitionBaseline(Transition):
    """
    Class to implement the baseline transition algorithm. It calculates a transition with 4 commands:
    1) Create all the nodes in the final allocation. 
    2) Allocate the containers in these created nodes.
    3) Remove all the containers in the initial allocation.
    4) Remove all the nodes in the initial allocation. 
    """
    def __init__(self, timing_args: TimedOps.TimingArgs, system: System):
        """
        Creates an object for transition between two allocations.
        :param timing_args: Creation and removal times for containers and nodes.
        :param system: System performance and computational requirements.
        """
        self._timing_args = timing_args
        self._system = system

    def get_worst_case_transition_time(self) -> int:
        """
        Get the worst-case transition time.
        :return: The worst-case transition time.
        """
        return self._timing_args.node_creation_time + self._timing_args.container_creation_time + \
            self._timing_args.container_removal_time + self._timing_args.node_removal_time
    
    def get_creation_in_transition_time(self) -> int:
        """
        Get the total time required to create nodes and next allocate containers in the transition.
        :return: The node+container creation time.
        """
        return self._timing_args.node_creation_time + self._timing_args.container_creation_time
    
    def get_recycling_levels(self) -> tuple[float, float]:
        """
        Get node and container recycling levels for the transition.
        :return: A tuple with node and container recycling levels.
        """
        return Recycling.INVALID_RECYCLING, Recycling.INVALID_RECYCLING

    def get_recycled_node_pairs(self) -> None:
        """
        Get the recycled node pairs.
        :return: Recycled node pairs.
        """
        # In the baseline transition there is no node recycling, so None is returned.
        return None

    @staticmethod
    def compare_vm_nodes(node1: Vm, node2: Vm) -> bool:
        """
        Check if two Vm nodes come from the same instance class and allocate the same containers.
        :param node1: Node 1.
        :param node2: Node 2.
        :return: True if the nodes come from the same instance class and allocate the same containers.
        """
        if node1.ic != node2.ic or len(node1.cgs) != len(node2.cgs):
            return False
        for cg1 in node1.cgs:
            found_equal_cg = False
            for cg2 in node2.cgs:
                if cg1.cc == cg2.cc and cg1.replicas == cg2.replicas:
                    found_equal_cg = True
                    break
            if not found_equal_cg:
                return False
        return True

    def calculate_sync(self, initial_alloc: Allocation, final_alloc: Allocation) -> tuple[list[Command], int]:
        """
        Calculate a synchronous transition from the initial allocation to the final allocation for
        the baseline algorithm.
        :param initial_alloc: Initial allocation.
        :param final_alloc: Final allocation.
        :return: A list of commands and the time to perform the transition.
        """
        self._commands = []
        initial_nodes = initial_alloc[:]
        final_nodes = final_alloc[:]

        # A transition is necessary when some nodes or allocated containers change
        for initial_node in initial_nodes[:]:
            for final_node in final_nodes:
                if TransitionBaseline.compare_vm_nodes(initial_node, final_node):
                    initial_nodes.remove(initial_node)
                    final_nodes.remove(final_node)
                    break
        if len(initial_nodes) == 0 and len(final_nodes) == 0:
            return [], 0

        if len(final_nodes) > 0:
            # Node creation is the most time-consuming. Therefore, it is the first operation to be performed
            create_nodes_command = Command(create_nodes=final_nodes[:])
            self._commands.append(create_nodes_command)
            # Allocate containers in the created nodes
            allocate_containers_command = \
                Command(allocate_containers=[
                    (node, cg.cc, cg.replicas) 
                    for node in final_nodes 
                    for cg in node.cgs
                ])
            allocate_containers_command.sync_on_nodes_creation = True
            self._commands.append(allocate_containers_command)

        if len(initial_nodes) > 0:
            # Remove containers in the initial nodes
            remove_containers_command = \
                Command(remove_containers=[
                    (node, cg.cc, cg.replicas) 
                    for node in initial_nodes
                    for cg in node.cgs
                ])
            self._commands.append(remove_containers_command)
            # Remove nodes in the initial nodes
            remove_nodes_command = Command(remove_nodes=initial_nodes[:])        
            self._commands.append(remove_nodes_command)

        return self._commands, self.get_transition_time(self._commands, self._timing_args)


class TransitionRBT(Transition):
    """
    Class to implement the recycling based transition algorithm. There are
    two variants: RBT1 and RBT2.
    """

    # A negative ID for unscalable containers
    UNSCALABLE_CONTAINER_ID = -1

    def __init__(self, timing_args: TimedOps.TimingArgs, system: System, 
                 transition_algorithm: TransitionAlgorithm.RBT,
                 hot_node_scale_up: bool = False, hot_container_scale: bool = False):
        """
        Creates an object for transition between two allocations.
        :param timing_args: Creation/removal/scaling times for containers and nodes.
        :param system: Applications's performance on different container classes and available instance classes.
        :param transition_algorithm: The specific RBT variant to use (RBT1 or RBT2).
        :param hot_node_scale_up: Set to enable hot node scaling-up.
        :param hot_replicas_scale: Set to enable hot scaling of container computational parameters.
        """
        self._timing_args: TimedOps.TimingArgs = timing_args
        self._system: System = system
        self._recycling: RecyclingVmt = None
        self._recycling_vm: Recycling = None
        self._id_scalable_containers: dict[Vmt, dict[int, tuple[int, int]]] = {}
        self._current_alloc: list[Vmt] = []
        self._unalloc_node_cs: list[tuple[Vmt, ContainerClass, int]]  = []
        self._app_unalloc_perf: defaultdict[App, RequestsPerTime]  = None
        self._app_perf_surplus:  defaultdict[App, RequestsPerTime]  = None
        self._app_perf_increment: defaultdict[App, RequestsPerTime]  = None
        self._allocatable_cs_next_step: list[tuple[Vm, ContainerClass, int]] = []
        self._unallocated_containers_in_new_nodes: list[tuple[Vm, ContainerClass, int]] = []
        self._hot_node_scale_up = hot_node_scale_up
        self._hot_replicas_scale = hot_container_scale
        self._commands: list[Command] = None
        self._sync_on_next_alloc_upgraded_nodes = True
        self._rbt1 = transition_algorithm == TransitionAlgorithm.RBT1 
        self._rbt2 = transition_algorithm == TransitionAlgorithm.RBT2 

    def _remove_allocate(self, cc: ContainerClass, replicas: int, node: Vmt, command: Command, 
                         obsolete: bool=False) -> tuple[int, list[tuple[ContainerClass, int, Vmt]]]:
        """
        Allocate the container replicas to the node, freeing up computational resources in the node
        coming from obsolete containers if necessary, while ensuring the fullfilment of the application's 
        minimum performance constraints. The node state remains unchanged when no replicas are allocated.
        :param cc: Container class.
        :param replicas: The number of replicas to allocate.
        :param node: Node.
        :param command: Command with (obsolete) containers to be removed and containers to be allocated.
        :param obsolete: True if the containers to allocate are copies of obsolete containers.
        :return: The number of actually allocated replicas and a list removed replicas.
        """

        # Remove any label from the container class
        if cc.label != "":
            cc = replace(cc, label="")

        # List of tuples with removed replicas (container class, number of replicas, node)
        removed_replicas = []

        # Required computational resources to allocate the replicas
        required_cores = replicas * cc.cores
        required_mem = replicas * cc.memv

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
                    ceil((required_mem - node.free_mem) / obsolete_cc.memv)
                )
                if required_replicas == 0:
                    break

                # Get the number of replicas of the obsolete container that could be removed
                obsolete_replicas_count = self._get_obsolete_removable_replicas(obsolete_cc, node, required_replicas,
                                                                       available_perf_surplus[obsolete_cc.app])
                if obsolete_replicas_count > 0:
                    available_perf_surplus[obsolete_cc.app] -= obsolete_cc.perf * obsolete_replicas_count
                    obsolete_replicas.append((obsolete_cc, obsolete_replicas_count))
                    required_cores -= obsolete_replicas_count * obsolete_cc.cores
                    required_mem -= obsolete_replicas_count * obsolete_cc.memv

        # Calculate the number of cores and memory obtained from the removal of obsolete containers in the node
        removed_cores = replicas * cc.cores - required_cores
        removed_mem = replicas * cc.memv - required_mem

        # Calculate how many new container replicas could be allocated after the removals
        allocatable_replicas = min(
            replicas,
            int((node.free_cores.magnitude + removed_cores.magnitude + TransitionRBT._DELTA) / cc.cores.magnitude),
            int((node.free_mem.magnitude + removed_mem.magnitude + TransitionRBT._DELTA) / cc.memv.magnitude)
        )

        # Allocate the replicas. At this point, the removals and allocations are actually performed
        if allocatable_replicas > 0:
            for obsolete_cc, obsolete_replicas_count in obsolete_replicas:
                self._remove_obsolete_replicas(obsolete_cc, obsolete_replicas_count, node, command)
                removed_replicas.append((obsolete_cc, obsolete_replicas_count, node))
            allocated_replicas = self._allocate(cc, allocatable_replicas, node, command, obsolete)
            assert allocatable_replicas == allocated_replicas, "The replicas must be allocatable"

        return allocatable_replicas, removed_replicas

    def _allocate(self, cc: ContainerClass, replicas: int, node: Vmt, command: Command, obsolete: bool=False) -> int:
        """
        Allocate a number of replicas in the node.
        :param cc: Container class.
        :param replicas: The number of replicas to allocate.
        :param node: Node.
        :param command: Allocation command.
        :param obsolete: Set to True if the replicas to allocate are obsolete replicas.
        :return: The number of actually allocated replicas.
        """
        allocatable_replicas = min(
            replicas,
            int((node.free_cores.magnitude + TransitionRBT._DELTA) / cc.cores.magnitude),
            int((node.free_mem.magnitude + TransitionRBT._DELTA) / cc.memv.magnitude)
        )
        if allocatable_replicas > 0:
            node.free_cores = node.free_cores - allocatable_replicas * cc.cores
            assert node.free_cores.magnitude > - TransitionRBT._DELTA, "Node free cores cannot not be negative"
            node.free_mem = node.free_mem - allocatable_replicas * cc.memv
            assert node.free_mem.magnitude > - TransitionRBT._DELTA, "Node free memory cannot not be negative"
            node.replicas[cc] += allocatable_replicas
            command.allocate_containers.append((node, cc, allocatable_replicas))
            if not obsolete:
                # The application's unallocated performance accounts for new containers, not obsolete ones
                self._app_unalloc_perf[cc.app] -= allocatable_replicas * cc.perf
                assert self._app_unalloc_perf[cc.app].magnitude > -TransitionRBT._DELTA, "Invalid performance"
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

    def _remove_obsolete_replicas(self, cc: ContainerClass, replicas: int, node: Vmt, command: Command,
                                  relaxed_removal: bool = False) -> int:
        """
        Remove obsolete replicas for the container class in the node. The number of replicas actually 
        removed is limited by the number of replicas in the node and the minimum application's 
        performance constraint.
        :param cc: Container class.
        :param replicas: Replicas to remove.
        :param node: Node.
        :param command: Command with container removals.
        :param relaxed_removal: Set to True to use the application's performance increment instead of
        the application's performance surplus to perform the removal. This relaxed performance contraint
        applies only to the removal of containers previously created in the same command.
        :return: Number of replicas that are actually removed.        
        """
        # Remove any label from the obsolete container class
        if cc.label != "":
            cc = replace(cc, label="")
        if cc not in node.replicas:
            return 0
        # The relaxed performance surplus is the sum of the application's performance surplus and the performance 
        # increment from the same command. It is enabled when removing containers that have been allocated in the 
        # same command. The command simplification method deletes both, the allocation and removal of the same 
        # containers within the same command. 
        relaxed_perf_surplus = {
            app: self._app_perf_surplus[app] + self._app_perf_increment[app] 
            for app in self._app_perf_surplus
        }
        performance_surplus = relaxed_perf_surplus if relaxed_removal else self._app_perf_surplus
        removed_replicas = min(
            node.replicas[cc], 
            replicas, 
            int(performance_surplus[cc.app]/cc.perf)
        )
        command.remove_containers.append((node, cc, removed_replicas))
        node.replicas[cc] -= removed_replicas
        if node.replicas[cc] == 0:
            del node.replicas[cc]
        node.free_cores += cc.cores * removed_replicas
        assert (node.free_cores - node.ic.cores).magnitude < TransitionRBT._DELTA, "Invalid node free cores"
        node.free_mem += cc.memv * removed_replicas
        assert (node.free_mem - node.ic.mem).magnitude < TransitionRBT._DELTA, "Invalid node free mem"
        performance_surplus[cc.app] -= cc.perf * removed_replicas
        self._recycling.obsolete_containers[node][cc] -= removed_replicas
        if self._recycling.obsolete_containers[node][cc] == 0:
            del self._recycling.obsolete_containers[node][cc]
        return removed_replicas

    def _get_obsolete_removable_replicas(self, cc: ContainerClass, node: Vmt, replicas_to_remove: int,
                                         available_perf_surplus: RequestsPerTime) -> int:
        """
        Get the number of obsolete replicas of a container class in a node that can be removed while ensuring
        the application's minimum performance constraint.
        :param cc: Container class.
        :param node: Node.
        :param replicas_to_remove: The number of replicas to remove.
        :param available_perf_surplus: The available performance surplus for the container application.
        :return: The actual number of removable replicas.
        """
        n_removable = min(
            int(available_perf_surplus / cc.perf),
            replicas_to_remove,
            self._recycling.obsolete_containers[node][cc]
        )
        return n_removable
  
    def _get_free_capacity_nodes(self, nodes: list[Vm] = None) -> list[Vmt]:
        """
        Get nodes with free or freeable capacity. 
        :param nodes: Elegible nodes or all the nodes in the allocation when this parameter is no set.
        :return: A list of nodes with free or freeable capacity. Nodes that do not allocate containers are returned at 
        the end of the list sorted by increasing price.
        """
        if nodes is None:
            nodes = self._current_alloc
        empty_nodes = [] # Nodes without allocated containers
        allocated_nodes = [] # Nodes with allocated containers

        for node in nodes:
            # Check if node is empty
            if (node.ic.cores - node.free_cores).magnitude < Transition._DELTA and \
                    (node.ic.mem - node.free_mem).magnitude < Transition._DELTA:
                empty_nodes.append(node)
                continue
            free_cores = node.free_cores
            free_mem = node.free_mem
            app_perf_surplus = dict(self._app_perf_surplus) # Copy of application's performance surplus
            for cc, replicas in node.replicas.items():
                if node in self._recycling.obsolete_containers and cc in self._recycling.obsolete_containers[node]:
                    app = cc.app
                    cc_perf_surplus = min(app_perf_surplus[app], cc.perf * replicas)
                    surplus_replicas = floor(cc_perf_surplus / cc.perf + Transition._DELTA)
                    free_cores += surplus_replicas * cc.cores
                    free_mem += surplus_replicas * cc.memv
                    app_perf_surplus[app] -= cc_perf_surplus
            if free_cores.magnitude > Transition._DELTA and free_mem.magnitude > Transition._DELTA:
                allocated_nodes.append(node)
        sorted_empty_nodes = sorted(empty_nodes, key=lambda n: n.ic.price)
        return allocated_nodes + sorted_empty_nodes
        
    def _remove_last_obsolete_containers(self, command: Command):
        """
        Remove the last obsolete containers of applications with all its new containers allocated.
        :param command: A command with the removal of containers.
        """
        for node, obsolete_cc_replicas in self._recycling.obsolete_containers.items():
            for obsolete_cc, replicas in dict(obsolete_cc_replicas).items():
                if self._app_unalloc_perf[obsolete_cc.app].magnitude < TransitionRBT._DELTA:
                    replicas_count = self._remove_obsolete_replicas(obsolete_cc, replicas, node, command)
                    assert replicas_count == replicas, "All the replicas must be removable"

    def _remove_obsolete_nodes(self, command: Command):
        """
        Check if there are obsolete nodes that can be removed. Obsolete nodes can be removed when
        they do not allocate containers. A command with the node removal operation is generated, but the
        nodes are not actually removed from the allocation, since they may be useful during the transition.
        :param command: A command with the removal of nodes.
        """
        for node in self._recycling.obsolete_nodes[:]:
            if node.is_empty():
                assert (node.free_cores - node.ic.cores).magnitude < TransitionRBT._DELTA, "Can not remove the node"
                assert (node.free_mem - node.ic.mem).magnitude < TransitionRBT._DELTA, "Can not remove the node"
                command.remove_nodes.append(node)
                if node in self._recycling.obsolete_containers:
                    del self._recycling.obsolete_containers[node]
                self._recycling.obsolete_nodes.remove(node)

    def _remove_allocate_copy_init(self, min_perf: dict[App, RequestsPerTime]):
        """
        Initialize the remove-allocate-copy algorithm.
        :param min_perf: Minimum application performances to be fulfilled during the transition.
        """
        # Calculate the total cores and memory of new containers in recycled and upgraded nodes.
        # They are necessary to calculate relative container sizes
        total_new_cpu = 0
        total_new_mem = 0
        for n, cc_replicas in self._recycling.new_containers.items():
            # We focus on new containers in recycled and upgraded nodes
            if n in self._recycling.recycled_node_pairs | self._recycling.upgraded_node_pairs:
                total_new_cpu += sum(cc.cores.magnitude * replicas for cc, replicas in cc_replicas.items())
                total_new_mem += sum(cc.memv.magnitude * replicas for cc, replicas in cc_replicas.items())

        # Build a list of unallocated containers (new containers) in recycled and upgraded nodes,
        # sorted by decreasing size
        self._unalloc_node_cs = []
        container_sizes = []
        for n, cc_replicas in self._recycling.new_containers.items():
            if n in self._recycling.recycled_node_pairs | self._recycling.upgraded_node_pairs:
                for cc, replicas in cc_replicas.items():
                    new_replicas = (n, cc, replicas)
                    container_size = cc.cores.magnitude / total_new_cpu + cc.memv.magnitude / total_new_mem
                    container_sizes.append(container_size)
                    self._unalloc_node_cs.append(new_replicas)
        self._unalloc_node_cs = [
            new_replicas
            for _, new_replicas in sorted(zip(container_sizes, self._unalloc_node_cs), key=lambda x: x[0],
                                             reverse = True)
        ]

        # Unallocated application performances. It is calculated from application containers not allocated yet
        self._app_unalloc_perf = defaultdict(lambda : RequestsPerTime("0 req/s"))
        for _, cc_replicas in self._recycling.new_containers.items():
            for cc, replicas in cc_replicas.items():
                self._app_unalloc_perf[cc.app] += replicas * cc.perf

        # Initial application's performance surplus
        self._app_perf_surplus = get_app_perf_surplus(min_perf, self._current_alloc)
        assert min(self._app_perf_surplus.values()).magnitude >= 0, "Invalid performance surplus"

        # Performance increment to update the application performance surplus at the end of the command
        self._app_perf_increment = defaultdict(lambda : RequestsPerTime("0 req/s"))

        # List of containers that will be allocatable in the next execution of the remove-allocate-copy algorithm
        self._allocatable_cs_next_step = []

    def _update_copy_state(self, allocatable_replicas:int, cc: ContainerClass, src_node: Vmt, 
                           copied_obsolete_replicas: list[tuple[ContainerClass, int, Vmt]], 
                           removed_obsolete_replicas: list[tuple[ContainerClass, int, Vmt]],
                           available_obsolete_replicas: dict[Vmt, dict[ContainerClass, int]], 
                           available_node_free_resources: list[Vmt, tuple[float, float]]) -> Command:
        """
        Update the state for the next copy of obsolete containers within the remove-allocate-copy algorithm.
        The state is defined by: 
        - Available obsolete replicas in all the nodes.
        - Available free resources in all the nodes.
        :param allocatable_replicas: Number of allocatable replicas after the last copy operation.
        :param cc: Container class for the allocatable replicas after the last copy operation.
        :param src_node: Source node that will allocate the replicas after the last copy operation.
        :param copied_obsolete_replicas: List of obsolete replicas in the source node copied to allocate 
        new replicas.
        :param removed_obsolete_replicas: List of obsolete replicas removed in destination nodes to free up
        computational resources for copies of obsolete replicas coming from source nodes.
        :param available_obsolete_containers: Available obsolete containers after the copy 
        and subsequent allocation in the source node.
        :param available_node_free_resources: Available free computational resources after after the copy 
        and subsequent allocation in the source node.
        :return: A command with the removal of copies of obsolete containers in the source node.
        """
        command = Command()

        # Removed Obsolete replicas in the destination nodes will free up resources for copies of obsolete 
        # replicas coming from source nodes, increasing the available free cores and memory in those nodes.
        # The copied obsolete replicas will reduce free cores and memory in those nodes
        for removed_cc, removed_replicas, dest_node in removed_obsolete_replicas:
            free_dst_cores, free_dst_mem = available_node_free_resources[dest_node]
            free_dst_cores += removed_replicas * removed_cc.cores
            free_dst_mem += removed_replicas * removed_cc.memv
            available_node_free_resources[dest_node] = (free_dst_cores, free_dst_mem)
        for copied_cc, copied_replicas, dest_node in copied_obsolete_replicas:
            free_dst_cores, free_dst_mem = available_node_free_resources[dest_node]
            free_dst_cores -= copied_replicas * copied_cc.cores
            free_dst_mem -= copied_replicas * copied_cc.memv
            available_node_free_resources[dest_node] = (free_dst_cores, free_dst_mem)

        # Copied obsolete replicas in the source node will increase the available free cores 
        # and memory in the source nodes after being removed. Next, they will be reduced after 
        # the allocation of container replicas in the source node 
        free_src_cores, free_src_mem = available_node_free_resources[src_node]
        for copied_cc, copied_replicas, _ in copied_obsolete_replicas:
            free_src_cores += copied_replicas * copied_cc.cores
            free_src_mem += copied_replicas * copied_cc.memv
        free_src_cores -= allocatable_replicas * cc.cores
        free_src_mem -= allocatable_replicas * cc.memv
        available_node_free_resources[src_node] = (free_src_cores, free_src_mem)

        # The available obsolete replicas change after the copy. On one hand, the original replicas will be removed, 
        # so they will not be available for future copies. On the other hand, the destination copies will be available 
        # for future copies. 
        for copied_cc, copied_replicas, dest_node in copied_obsolete_replicas:
            available_obsolete_replicas[src_node][copied_cc] -= copied_replicas
            if available_obsolete_replicas[src_node][copied_cc] == 0:
                del available_obsolete_replicas[src_node][copied_cc]
            if copied_cc.label == "c": 
                # If the original replicas are copies, they can be removed with relaxed performance constraints
                self._remove_obsolete_replicas(copied_cc, copied_replicas, src_node, command, relaxed_removal=True)
            # Label the copied replicas
            labelled_copied_cc = replace(copied_cc, label="c")
            # The copied replicas are available for future copies 
            if dest_node not in available_obsolete_replicas:
                available_obsolete_replicas[dest_node] = {labelled_copied_cc: copied_replicas}
            else:
                if labelled_copied_cc not in available_obsolete_replicas[dest_node]:
                    available_obsolete_replicas[dest_node][labelled_copied_cc] = copied_replicas
                else:
                    available_obsolete_replicas[dest_node][labelled_copied_cc] += copied_replicas

        # Removed obsolete replicas in the destination nodes will not be longer available
        for removed_cc, removed_replicas, dest_node in removed_obsolete_replicas:
            available_obsolete_replicas[dest_node][removed_cc] -= removed_replicas
            if available_obsolete_replicas[dest_node][removed_cc] == 0:
                del available_obsolete_replicas[dest_node][removed_cc]

        return command
    
    def _copy_obsolete_containers(self, node_cc_replicas: tuple[Vmt, ContainerClass, int],
                                  available_obsolete_containers: dict[Vmt, dict[ContainerClass, int]],
                                  available_node_free_resources: dict[Vmt, tuple[int, int]],
                                  dest_nodes: list[Vmt]) -> tuple[int, Command]:
        """
        Copy obsolete replicas from a source node to destination nodes, removing obsolete containers from
        the destination nodes if necessary. This operation sets up the allocation of new containers at the
        beginning of the next remove-allocate-copy execution.
        The copy process depends of previous copies in the same call of this method that modify the available 
        obsolete containers and free computational resources.
        :param node_cc_replicas: Source node, container class and replicas for the new containers to allocate. 
        at the beginning of the next call to the remove-allocate-copy algorithm.
        :param available_obsolete_containers: Elegible obsolete containers in all the nodes. They are updated
        when at least a new container replica will be allocatable in the next remove-allocate-copy execution.
        :param available_node_free_resources: Free computational resources in the node. They are updated when
        at least a new container replica will be allocatable in the next remove-allocate-copy execution.
        :param dest_nodes: Elegible destination nodes to allocate copies of obsolete containers.
        :return: A tuple with the number of allocatable new containers and a command with the removals and 
        allocations of containers.
        """
        # Source node, container class and number of replicas to allocate
        src_node, cc, replicas_to_allocate = node_cc_replicas

        # Available obsolete containers in the source node. A given obsolete container frees up computational 
        # resources in the source node, but it can be used only once
        available_src_node_obsolete_containers = available_obsolete_containers[src_node]

        # Current free computational resources in the source node. It is updated after copying obsolete containers
        free_src_cores, free_src_mem = available_node_free_resources[src_node]

        # Required cores and memory to allocate all the replicas
        required_cores = replicas_to_allocate * cc.cores
        required_mem = replicas_to_allocate * cc.memv

        # The state must be recovered when we fail to allocate at least one replica, so create backups.
        # It should be noted that the state is changed by the execution of the remove-allocate algorithm
        modified_dest_nodes = [] # List of modified nodes to recover their state when the copy fails
        dest_nodes_backup = {
            node: (node.free_cores, node.free_mem, defaultdict(lambda: 0, node.replicas))
            for node in dest_nodes
        } # Backup of all the destination nodes, including free resources and allocated containers
        zero_perf = RequestsPerTime("0 req/s")
        app_perf_surplus_backup = defaultdict(lambda: zero_perf, self._app_perf_surplus)
        app_perf_increment_backup = defaultdict(lambda: zero_perf, self._app_perf_increment)
        obsolete_containers_backup = {
            node: {cc: replicas for cc, replicas in cc_replicas.items()}
            for node, cc_replicas in self._recycling.obsolete_containers.items()
        } # backup of obselete containers in all the nodes

        command = Command()

        # In case of success, these obsolete replicas in the source node will be no longer elegible in next copies
        copied_obsolete_replicas = []

        # In case of success, these obsolete replicas will be removed from destination nodes and will no longer 
        # elegible in next copies
        removed_obsolete_replicas = []

        # Copy enough obsolete replicas to free up computational resources in the source node to allocate
        # the new replicas
        for removable_cc, available_replicas in available_src_node_obsolete_containers.items():
            # Calculate the number of obselete replicas of the container to remove from the source node
            required_obsolete_replicas = max(
                ((required_cores - free_src_cores) / removable_cc.cores).magnitude,
                ((required_mem - free_src_mem) / removable_cc.memv).magnitude
            )
            required_obsolete_replicas = int(ceil(required_obsolete_replicas - TransitionRBT._DELTA))
            replicas_to_remove = min(required_obsolete_replicas, available_replicas)

            # Go out if no more replicas are required to free up enough computational resources
            if replicas_to_remove <= 0:
                break

            # Try copying the obsolete replicas from the source node to destination nodes
            for dest_node in dest_nodes:
                if dest_node == src_node:
                    # Cannot copy obsolete containers to the same node
                    continue
                obsolete_replicas, removed_replicas = self._remove_allocate(removable_cc, replicas_to_remove,
                                                                            dest_node, command, obsolete=True)
                # if some obsolete replicas have been copied, update for the next copy of obsolete containers
                if obsolete_replicas > 0:
                    copied_obsolete_replicas.append((removable_cc, obsolete_replicas, dest_node))
                    removed_obsolete_replicas.extend(removed_replicas)
                    modified_dest_nodes.append(dest_node)
                    required_cores -= obsolete_replicas * removable_cc.cores
                    required_mem -= obsolete_replicas * removable_cc.memv
                    replicas_to_remove -= obsolete_replicas
                    if replicas_to_remove == 0:
                        # Go out if we have copied all the required obsolete replicas to free up enough resources
                        break

        # Calculate the number of replicas that will be allocatable in the source node at the beginning the next 
        # remove-allocate-copy execution
        free_cores = free_src_cores + replicas_to_allocate * cc.cores - required_cores
        free_mem = free_src_mem + replicas_to_allocate * cc.memv - required_mem
        allocatable_replicas = min(
            replicas_to_allocate,
            int((free_cores.magnitude + TransitionRBT._DELTA) / cc.cores.magnitude),
            int((free_mem.magnitude + TransitionRBT._DELTA) / cc.memv.magnitude),
        )

        # If no replicas of the new container will be allocatable
        if allocatable_replicas == 0:
            # Recover from backups
            for mod_node in modified_dest_nodes:
                mod_node.free_cores, mod_node.free_mem, mod_node.replicas = dest_nodes_backup[mod_node]
            self._app_perf_surplus = app_perf_surplus_backup
            self._app_perf_increment = app_perf_increment_backup
            self._recycling.obsolete_containers = obsolete_containers_backup
 
        # If some replicas of the new container will be allocatable
        else:
            # The allocation is preserved and the state is updated for the next new containers to allocate
            command2 = self._update_copy_state(allocatable_replicas, cc, src_node,
                                               copied_obsolete_replicas, removed_obsolete_replicas,
                                               available_obsolete_containers, available_node_free_resources)            
            # Extend the comand with the removal of copies of obsolete containers in the source node
            command.extend(command2)

        if allocatable_replicas == 0:
            command = Command() # Return an empty command if no replicas will be allocatable

        return allocatable_replicas, command

    def _allocate_with_free_obsolete(self, command: Command):
        """
        Allocate unallocated new containers using free computational resources in the same node and removing    
        obsolete containers from the same node if it were necessary.
        :param command: A command with container allocations and removals.
        """
        # Allocate new container replicas prepared in a previous copy phase of the algorithm. They must be allocatable
        for src_node, cc, allocatable_replicas in self._allocatable_cs_next_step:
            # Allocate using free computational resources on the same node and removing obsolete
            # containers from the same node if it were necessary
            allocated_replicas, _ = self._remove_allocate(cc, allocatable_replicas, src_node, command)
            assert allocated_replicas == allocatable_replicas, "Containers must be allocatable"
        self._allocatable_cs_next_step.clear()

        # Allocate the rest of unallocated new container replicas
        unalloc_node_cs = self._unalloc_node_cs[:]
        for src_node, cc, replicas_to_allocate in unalloc_node_cs:
            node_cc_replicas_index = self._unalloc_node_cs.index((src_node, cc, replicas_to_allocate))
            allocated_replicas, _ = self._remove_allocate(cc, replicas_to_allocate, src_node, command)
            if allocated_replicas > 0:
                self._unalloc_node_cs.pop(node_cc_replicas_index)
                replicas_to_allocate -= allocated_replicas
                if replicas_to_allocate > 0:
                    self._unalloc_node_cs.insert(node_cc_replicas_index, (src_node, cc, replicas_to_allocate))
        
    def _remove_allocate_copy(self, copy_nodes: list[Vmt] = None) -> Command:
        """
        Perform one transition step by removing obsolete containers and nodes, allocating unallocated new
        containers, and copying obsolete containers to other nodes, in preparation for the next transition step.
        :param copy_nodes: A list of nodes where obsolete containers can be copied. All the nodes in the
        current allocation are used when this parameter is not set.
        :return: One command with container removals, containers allocations and node removals.
        """
        # Check that obsolete containers have not the copy label, 'c'. It can be commented in production
        assert self._debug_check_copy_label_obsolete_containers(), "Obsolete containers are not properly labelled"
        
        command = Command()

        # Remove the last obsolete containers of applications with all its new containers allocated.
        # The corresponding computational resources can be freed up for the allocation of new containers
        self._remove_last_obsolete_containers(command)

        # Allocate using free computational resources in the same node and removing obsolete
        # containers from the same node if it were necessary
        self._allocate_with_free_obsolete(command)

        # If all the new containers were allocated, it is time to remove the obsolete nodes
        # Nodes are not actually removed from the allocation until the end of the transition, 
        # since they may be useful during the transition. They appear as removed onlyin the command
        if len(self._unalloc_node_cs) == 0:
            # Check if obsolete nodes can be removed and update the command
            self._remove_obsolete_nodes(command)
            command.simplification()  
            return command
        
        # Set up the allocation of new container replicas for the next remove-allocate-copy step.
        # Try copying obsolete containers from the node to other nodes (destination nodes),
        # yielding enough application's performance surplus to allocate the replicas of unallocated
        # containers in the next transition step

        # Get nodes with free or freeable capacity, leaving empty nodes at the end, sorted by
        # increasing price. Thus, empty nodes will be used as a last option to copy obsolete containers.
        # copy_nodes include all the nodes when it is None (RBT1) and only the temporal nodes with RBT2
        if copy_nodes is None:
            copy_nodes = self._current_alloc
        elegible_nodes = self._get_free_capacity_nodes(copy_nodes)

        # Available obsolete containers for copy in the node for the next transition step
        available_obsolete_containers = {
            node: {cc: replicas for cc, replicas in cc_replicas.items()}
            for node, cc_replicas in self._recycling.obsolete_containers.items()
        }

        # Available free computational resources for copy in the node for the next transition step 
        available_node_free_resources = {
            node: (node.free_cores, node.free_mem) for node in self._current_alloc
        }

        # Copy of obsolete containers from source nodes to elegible destination nodes, starting with the nodes
        # with larger unallocated containers 
        for unalloc_node_cc in self._unalloc_node_cs[:]:
            src_node, cc, replicas_to_allocate = unalloc_node_cc

            # If the node has obsolete containers
            if src_node in available_obsolete_containers:
                # Copy obsolete containers from the source node to elegible nodes. This process updates the
                # available obsolete containers and the available free computational resources for the next copy. 
                # It returns the number of allocatable replicas in the source node at the next transition step and 
                # one command with the removal of obsolete containers in the source node and the allocation of copies
                # of obsolete containers in elegible nodes 
                allocatable_replicas, command2 = \
                    self._copy_obsolete_containers(unalloc_node_cc, available_obsolete_containers, 
                                                   available_node_free_resources, elegible_nodes)
                command.extend(command2)
                
                if allocatable_replicas > 0:
                    # Remove the node from the elegible nodes
                    if src_node in elegible_nodes:
                        elegible_nodes.remove(src_node)
                    if self._rbt1:
                        # Complete the list of containers allocatable in the next transisition step and remove them
                        # from the list of unallocated containers
                        self._allocatable_cs_next_step.append((src_node, cc, allocatable_replicas))
                        index = self._unalloc_node_cs.index((src_node, cc, replicas_to_allocate))
                        replicas_to_allocate -= allocatable_replicas
                        if replicas_to_allocate > 0:
                            self._unalloc_node_cs[index] = (src_node, cc, replicas_to_allocate)
                        else:
                            self._unalloc_node_cs.pop(index)

        # Check if obsolete nodes can be removed and update the command
        self._remove_obsolete_nodes(command)

        command.simplification()  
        return command

    def _get_allocation(self, app_performance: dict[App, RequestsPerTime]) -> list[Vm]:
        """
        Get an allocation to fulfill application performances using FCMA with speed level 3.
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
        Append a command to the list of commands and update application's performance surplus from the 
        performance incrementscoming from command's container allocations. In addition, it combines
        the fragments to scale the containers.
        :param command: The command to append.
        :append_null_command: Null commands are not appended if this option is not set.
        """
        for app in self._app_perf_increment:
            self._app_perf_surplus[app] += self._app_perf_increment[app]
            self._app_perf_increment[app] = 0
        if not command.is_null() or append_null_command:
            self._commands.append(command)
        # The first command that allocates containers in upgraded nodes is set with the synchronization on 
        # node upgrade
        allocation_in_upgraded_nodes = any(
            node1 in self._recycling.upgraded_node_pairs and node1.ic == self._recycling.upgraded_node_pairs[node1].ic
            for node1, _, _  in command.allocate_containers
        )
        if self._sync_on_next_alloc_upgraded_nodes and allocation_in_upgraded_nodes:
            command.sync_on_nodes_upgrade = True
            self._sync_on_next_alloc_upgraded_nodes = False
        
        # Combine fragments to scale-up containers
        for node, fcc, fr in command.allocate_containers[:]:
            # If the container is a fragment
            if fcc.id > 0:
                scale_up_command = self.combine_replica_fragments(node, fcc.cc, fr, up_down=1)
                for cc in dict(self._current_alloc[node].keys()):
                    if cc.id == fcc.id:
                        del self._current_alloc[node][cc]
                command.scale_containers.append(scale_up_command.scale_containers)
                command.allocate_containers.remove((node, fcc, fr))

        # Combine fragments to scale-down containers
        for node, fcc, fr in command.remove_containers[:]:
            # If the container is a fragment
            if fcc.id > 0:
                scale_down_command = self.combine_replica_fragments(node, fcc, fr, up_down=-1)
                command.scale_containers.extend(scale_down_command.scale_containers)
                command.remove_containers.remove((node, fcc, fr))

    def combine_replica_fragments(self, node: Vmt, fcc: ContainerClass, fr: int, up_down:int = 1) -> Command:
        """
        Combine fragments to scale replicas updating the dictionary with the replicas and current fragments 
        in self._id_scalable_containers.
        :param node: Node where the fragments are allocated or removed.
        :param cc: Container class for the fragments.
        :param fr: Number of fragments.
        :param up_down: 1 for scale-up using the fragments and -1 for scale-down. 
        :return: A Command with the scales.
        """
        # The number of replicas to scale is given in the horizontal axis whereas the number of fragments is
        # given in the vertical axis. Bottom plots depict the replicas before adding new fragments and top
        # plots after de addition (the situation for fragment removals would be the reverse). The quotient
        # between the top and bottom plots would be another plot with container multipliers. Each plot includes
        # replicas with the minimum number of fragments and replicas with the maximum (with one additional 
        # fragment). Left and right plots depict two possible scenarios.
        # The goal is to distribute the fragments equally among all replicas with multipliers as close to 1.0
        # as possible. 
        #
        #    |                       |
        #    |         _________     |           ______
        #    | ________|             | __________|
        #    |           _______     |       __________
        #    | __________|           | ______|
        #    |___________________    |____________________
        #

        # Check params
        assert fcc.id > 0, "The fragments can not be combined"
        assert up_down == 1 or up_down == -1, "Invalid combination of fragments"

        command = Command()

        # Replicas and fragments for the container before the combination with the new fragments
        replicas, prev_fragments = self._id_scalable_containers[node][fcc.id]

        # Minimum and maximum number of fragments per replica before the combination 
        prev_min_fragments_per_replica = prev_fragments // replicas
        prev_max_fragments_per_replica = prev_min_fragments_per_replica + 1
        # Number of replicas with the maximum and the minimum number of replicas before the combination
        prev_n_max_replicas = prev_fragments - prev_min_fragments_per_replica * replicas
        prev_n_min_replicas = replicas - prev_n_max_replicas
        # Container class for the replicas with minimum and maximum fragments before the combination
        prev_min_cc = fcc * prev_min_fragments_per_replica
        if prev_n_max_replicas > 0:
            prev_max_cc = fcc * prev_max_fragments_per_replica
        # Total number of fragments after the combination  
        new_fragments = prev_fragments + fr * up_down
        # Minimum and maximum number of fragments per replica after the combination 
        new_min_fragments_per_replica = new_fragments // replicas
        new_max_fragments_per_replica = new_min_fragments_per_replica + 1
        # Number of replicas with the maximum and the minimum number of replicas after the combination
        new_n_max_replicas = new_fragments - new_min_fragments_per_replica * replicas
        new_n_min_replicas = replicas - new_n_max_replicas
        # Multipliers for escales between the minimum and between the maximum container classes
        min_multiplier = new_min_fragments_per_replica / prev_min_fragments_per_replica
        max_multiplier = new_max_fragments_per_replica / prev_max_fragments_per_replica
        # Number of replicas with scales between the minimums and between the maximums container classes
        n_min_scaled_replicas = min(prev_n_min_replicas, new_n_min_replicas)
        n_max_scaled_replicas = min(prev_n_max_replicas, new_n_max_replicas)
        # Scales between the minimums and between the maximums
        command.scale_containers.append((node, prev_min_cc, n_min_scaled_replicas, min_multiplier))
        if prev_n_max_replicas > 0:
            command.scale_containers.append((node, prev_max_cc, n_max_scaled_replicas, max_multiplier))
        # Intermediate scales between minimum and maximum container classes (leftmost figure)
        if prev_n_min_replicas > new_n_min_replicas:
            medium_multiplier = new_max_fragments_per_replica / prev_min_fragments_per_replica
            scaled_replicas = prev_n_min_replicas - new_n_min_replicas
            command.scale_containers.append((node, prev_min_cc, scaled_replicas, medium_multiplier))
        # Intermediate scales between minimum and maximum container classes (rightmost figure)
        elif prev_n_min_replicas < new_n_min_replicas and prev_n_max_replicas > 0:
            medium_multiplier = new_min_fragments_per_replica / prev_max_fragments_per_replica
            scaled_replicas = new_n_min_replicas - prev_n_min_replicas
            command.scale_containers.append((node, prev_max_cc, scaled_replicas, medium_multiplier))

        return command

    def _post_process_commands(self):
        """
        Perform the following post-processing on the comand list:
        - Delete a node removal if the node is used in a later command for container allocations.
        Note that obsolete nodes are not actually removed from the allocation, since they can be used
        as temporary nodes in later commands.
        - For each obsolete node, ignore all except its first remove operation, optimizing the cost. 
        - Remove empty commands.
        - Remove obsolete nodes from the current allocation.
        - Check the labels of allocated and removed containers to transform these operations into
        container scales when required.
        - Replace Vmt nodes by Vm nodes in the commands.
        """
        # Node removal commands are generated when all the containers of an obsolete node are removed.
        # However, the nodes could be useful in future to help in the transition of recycled nodes, so
        # they could be used after a removal command. A node removal command is invalid when there is
        # a later allocation in the node, so the removal must be deleted
        null_command = Command() 
        node_removal_command = defaultdict(lambda: null_command) # Start with no removals for all nodes
        for command in self._commands:
            for node, _, _ in command.allocate_containers:
                # If the containers are allocated in a node included in a previous removal
                if not node_removal_command[node].is_null():
                    # The node removal in the previous comand is invalid, so delete it
                    removal_command = node_removal_command[node]
                    removal_command.remove_nodes.remove(node)
            for node_to_remove in command.remove_nodes:
                # When a node is removed in several commands, only the first removal is preserved
                if node_removal_command[node_to_remove].is_null():
                    # This is the first removal command for the node (or a previous removal command was deleted)
                    node_removal_command[node_to_remove] = command
                else:
                    # A previous node removal command exists, so remove the node from the current command
                    command.remove_nodes.remove(node_to_remove)

        # Remove empty commands
        for command in self._commands[:]:
            if command.is_null():
                self._commands.remove(command)

        # Remove nodes from the current allocation. Now, nodes are actually removed
        for node, command in node_removal_command.items():
            if not command.is_null():
                # The node is removed in the first removal command
                self._current_alloc.remove(node)

        # Replace Vmt nodes by nodes Vm nodes in the commands
        for command in self._commands:
            command.vmt_to_vm()

    def _remove_allocate_copy_loop(self, max_time: int) -> int:
        """
        Repeat remove-allocate-copy operations adding the commands to the command list.
        The method returns when the last remove-allocation operation can not advance, or the total time required 
        to execute the commands (excluding node removals) is higher than or equal to maximum time limit. In that
        later case, the last remove-allocation operation is not performed.
        :param max_time: Maximum time to peform remove-allocate-copy operations.
        :return: The time required to execute the added commands.
        """
        elapsed_time = 0
        while elapsed_time < max_time:
            # Maximum time for a remove-allocate-copy operation, excludingnode removals
            command_max_container_time = self._timing_args.container_removal_time + \
                self._timing_args.container_creation_time
            # Backup the current allocation if the container operations may extend the time limit
            if elapsed_time + command_max_container_time > max_time:
                self._current_alloc_backup = [
                    (node.free_cores, node.free_mem, {cc:replicas for cc, replicas in node.replicas.items()}) 
                    for node in self._current_alloc
                ]
            command = self._remove_allocate_copy()
            container_command_time = command.get_container_command_time(self._timing_args)
            elapsed_time += container_command_time
            if elapsed_time > max_time:
                # Recover the allocation when the time limit is exceeded.
                # It should be noted that the command is not added to the command list, 
                # so its operations are not performed and the current allocation is not modified.
                for node in self._current_alloc:
                    node.free_cores, node.free_mem, node.replicas = self._current_alloc_backup[node]
                break
            else:
                self._append_command(command)
            if container_command_time == 0:
                break
        return elapsed_time

    def _complete_allocation_in_temporary_nodes(self, create_nodes_command: Command,
                                                allocate_in_new_nodes_command: Command):
        """
        Complete the allocation using temporary nodes.
        :param create_nodes_command: Command where temporary nodes creations are appended.
        :param allocate_new_nodes_command: Command where container allocations in temporary nodes are appended.
        """
        # Add a dummy node with enough capacity to allocate any number of containers
        dummy_node = Vmt(Vm(self._current_alloc[0].ic, ignore_ic_index=True))
        dummy_node.free_cores *= 10E12
        dummy_node.free_mem *= 10E12
        self._current_alloc.append(dummy_node)

        # Calculate application's performance provided by temporary nodes to allocate the
        # remaining containers. The following approaches are valid in the calculation of the temporary nodes:
        # - Execute the remove_allocate method allowing negative values for application's performance surplus. 
        # Next, they are compensated by container replicas in temporary nodes.
        # - Add a dummy node to the system with unlimited computational capacity, so all the remaining containers 
        # can be allocated executing the remove_allocate_copy method. Next, the allocations in the dummy node 
        # are compensated with container replicas in temporary nodes.
        # The second approach has been followed, since it does not require to modify the remove_allocate method 
        # to allow negative performance surplus values
        copy_nodes = None if self._rbt1 else [dummy_node]
        command = self._remove_allocate_copy(copy_nodes)
        zero_rps = RequestsPerTime("0 rps")
        tmp_app_perf = {app: zero_rps for app in self._app_perf_surplus}
        for node, cc, replicas in command.allocate_containers:
            if node == dummy_node:
                # The allocations in the dummy node will be performed in a previous command, so
                # a performance increment translates into a performance surplus increment
                self._app_perf_increment[cc.app] -= replicas * cc.perf
                self._app_perf_surplus[cc.app] += replicas * cc.perf
                # Update the application performance provided by temporary nodes
                tmp_app_perf[cc.app] += cc.perf * replicas

        # Get an allocation for application's performance on temporary nodes
        tmp_nodes = [Vmt(node) for node in self._get_allocation(tmp_app_perf)]
        for tmp_node_index in range(len(tmp_nodes)):
            # Change the id of temporary nodes to negative values to be easily identified
            tmp_nodes[tmp_node_index].id = -(tmp_node_index + 1)
            tmp_nodes[tmp_node_index].vm.id = -(tmp_node_index + 1)

        # The temporary nodes are added to the command with the creation of new nodes
        create_nodes_command.create_nodes.extend(tmp_nodes)

        # Move allocations from the dummy node to the temporary nodes
        containers_in_tmp_nodes = [
            (node, cc, replicas)
            for node in tmp_nodes for cc, replicas in node.replicas.items()
        ]
        allocate_in_new_nodes_command.allocate_containers.extend(containers_in_tmp_nodes)
        command.allocate_containers = [
            (node, cc, replicas)
            for node, cc, replicas in command.allocate_containers[:]
            if node != dummy_node
        ]

        # Remove obsolete containers and nodes from the recycling object
        if dummy_node in self._recycling.obsolete_containers:
            del self._recycling.obsolete_containers[dummy_node]

        # Temporary nodes and their containers are obsolete 
        for tmp_node in tmp_nodes:
            self._recycling.obsolete_nodes.append(tmp_node)
            self._recycling.obsolete_containers[tmp_node] = dict(tmp_node.replicas)

        # Replace the dummy node with temporary nodes in the current allocation
        self._current_alloc.remove(dummy_node)
        self._current_alloc.extend(tmp_nodes)

        # Some replicas can be allocated once temporary nodes are added, since some computational
        # resources may remain free in these nodes after the allocation of containers_in_tmp_nodes
        if self._rbt2:
            self._allocate_with_free_obsolete(command)

        self._append_command(command)

    def fragment_scaled_containers(self):
        """
        Fragment containers to emulate container's scale-ups and scale-downs through allocations of new
        containers (fragments) and removals of obsolete containers (fragments).
        """
        # Scaled-up containers are replaced by the initial container, as a recycled container, plus a set of
        # container fragments, as new containers. Scaled-down containers are replaced by the initial container, 
        # as a recycled container, plus a set of container fragments, as obsolete containers
        new_id = 1
        for node, cc1_cc2_replicas in dict(self._recycling.scaled_containers).items():
            self._id_scalable_containers[node] = {}
            for cc1_cc2, replicas in cc1_cc2_replicas.items():
                cc1, cc2 = cc1_cc2
                # One fragment corresponds to a minimum-size container
                cc1_cc2_fragment = replace(cc1 * (1/cc1.agg_level), id=new_id)
                if cc1.cores > cc2.cores:
                    # Fragment the scaled replicas in the allocation
                    node.replicas[cc1] -= replicas
                    if node.replicas[cc1] == 0:
                        del node.replicas[cc1]
                    node.replicas[cc2] = replicas
                    node.replicas[cc1_cc2_fragment] = replicas * (cc1.agg_level - cc2.agg_level) 
                # Set the same possitive container ID for all the scaled containers. All the containers get 
                # a negative container iD by default 
                    cc1 = replace(cc1, id=new_id)
                cc2 = replace(cc2, id=new_id)
                if cc1.cores > cc2.cores:
                    self._recycling.recycled_containers[node][cc2] = replicas
                    self._recycling.obsolete_containers[node][cc1_cc2_fragment] = \
                        replicas * (cc1.agg_level - cc2.agg_level)
                else:
                    self._recycling.recycled_containers[node][cc1] = replicas
                    self._recycling.new_containers[node][cc1_cc2_fragment] = \
                        replicas * (cc2.agg_level - cc1.agg_level)
                # Set the number of replicas and the total number of fragments for each escalable container
                self._id_scalable_containers[node][cc1.id] = (replicas, replicas*cc1.agg_level)                    
                new_id += 1

    def _transition_init(self, initial_alloc: Allocation, final_alloc: Allocation) -> bool:
        """
        Initialize the trasition between two allocations.
        :param initial_alloc: Initial allocation.
        :param final_alloc: Final allocation.
        :return: True when a transition is necessary to go from the initial allocation to the final allocation.
        """
        self._commands = []

        # Start with the initial allocation to transition from the initial allocation to the final allocation.
        # Transition is performed on a copy of initial_alloc nodes in Vmt format. Transition modifies
        # Vmt nodes, while preserving their initial state in Vm format, stored in initial_alloc.
        self._current_alloc = [Vmt(node) for node in initial_alloc]
        final_alloc_vmt = [Vmt(node) for node in final_alloc]
        vm_to_vmt = dict(zip(initial_alloc, self._current_alloc)) | \
                    dict(zip(final_alloc, final_alloc_vmt))

        # Calculate recycled node pairs, new nodes, nodes to remove, recycled containers, new containers
        # and containers to remove when transitioning from the initial allocation to the final allocation
        self._recycling_vm = Recycling(initial_alloc, final_alloc, 
                                       self._hot_node_scale_up, self._hot_replicas_scale)
        
        self._recycling = RecyclingVmt(self._recycling_vm, vm_to_vmt)

        if len(self._recycling.scaled_containers) > 0:
            # Frament scaled-up and scaled-down containers
            self.fragment_scaled_containers()

        # Check whether the initial allocation is identical to the final allocation
        if get_vmt_allocation_signature(self._current_alloc) == get_vmt_allocation_signature(final_alloc_vmt):
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
        # temporary nodes
        upgrade_node_info = [(n1, n2.ic) for n1, n2 in self._recycling.upgraded_node_pairs.items()]
        create_upgrade_nodes_command = Command(create_nodes=self._recycling.new_nodes, upgrade_nodes=upgrade_node_info)
        self._append_command(create_upgrade_nodes_command, append_null_command=True)

        # Allocation loop until node upgrading completes for RBT1 variant. 
        # Node upgrading time is less time-consuming than node creation time, so it completes before node creation.
        # The elapsed time can not be higher than the hot node scale up time
        elapsed_time = 0
        if self._rbt1:
            max_time = self._timing_args.hot_node_scale_up_time
            elapsed_time = self._remove_allocate_copy_loop(max_time)

        # Update the current allocation after completing the node upgrading
        for initial_node, final_node in self._recycling.upgraded_node_pairs.items():
            initial_node.upgrade(final_node)

        # Allocation loop until node creation completes for RBT1 variant.
        # Upgraded nodes can be used in the allocation of new containers meanwhile
        if self._rbt1 and elapsed_time < self._timing_args.node_creation_time:
            max_time = self._timing_args.node_creation_time - elapsed_time
            elapsed_time += self._remove_allocate_copy_loop(max_time)

        # Update the elapsed time and current allocation after completing the node creation
        elapsed_time = self._timing_args.node_creation_time
        self._current_alloc.extend(self._recycling.new_nodes)

        # Allocate new containers in new nodes. The corresponding command may be empty if there
        # are no new nodes in the final allocation. This command can be extended later to include
        # containers in temporary nodes
        allocate_in_new_nodes_command = Command(allocate_containers=self._unallocated_containers_in_new_nodes[:])
        allocate_in_new_nodes_command.sync_on_nodes_creation = True
        for _, cc, replicas in self._unallocated_containers_in_new_nodes:
            self._app_perf_increment[cc.app] += replicas * cc.perf
            self._app_unalloc_perf[cc.app] -= replicas * cc.perf
            assert self._app_unalloc_perf[cc.app].magnitude >= - TransitionRBT._DELTA, "Invalid performance"
        self._unallocated_containers_in_new_nodes.clear()
        self._append_command(allocate_in_new_nodes_command, append_null_command=True)

        # If there are still unallocated containers in recycled nodes
        if len(self._allocatable_cs_next_step) > 0 or len(self._unalloc_node_cs) > 0:
            # Temporary nodes are added if it were necessary
            self._complete_allocation_in_temporary_nodes(create_upgrade_nodes_command, allocate_in_new_nodes_command)

        # Two remove-allocate-copy steps may be necessary to complete the transition. The first
        # command to remove obsolete containers and next allocate the remaining new containers in recycled nodes.
        # The second command to remove containers from the nodes and next the obsolete nodes
        first_command = self._remove_allocate_copy()
        if not first_command.is_null():
            self._append_command(first_command)
        second_command = self._remove_allocate_copy()
        if not second_command.is_null():
            self._append_command(second_command)

        # Post-processing operations to obtain the final command list
        self._post_process_commands()

        # Check whether the commands implement a valid transition between the initial and the final allocations
        # Four commands must be enough for the RBT2 version
        assert not self._rbt2 or len(self._commands) <= 4, "Too many commands"
        assert self.check_transition(initial_alloc, final_alloc, self._commands), "Invalid transition"

        return self._commands, self.get_transition_time(self._commands, self._timing_args)

    def get_worst_case_transition_time(self) -> int:
        """
        Get the worst-case transition time, including node removals
        :return: The worst-case transition time.
        """
        worst_case_time = self._timing_args.node_creation_time + self._timing_args.container_creation_time + \
            (self._timing_args.container_removal_time + self._timing_args.container_creation_time) + \
             self._timing_args.container_removal_time + self._timing_args.node_removal_time 
        return worst_case_time 

    def get_recycling_levels(self) -> tuple[float, float]:
        """
        Get node and container recycling levels for the last transition.
        :return: A tuple with node and container recycling levels.
        """
        return self._recycling.node_recycling_level, self._recycling.container_recycling_level

    def get_recycled_node_pairs(self) -> dict[Vm, Vm]:
        """
        Get the recycled node pairs.
        :return: Recycled node pairs.
        """
        return self._recycling_vm.recycled_node_pairs

    def _debug_check_copy_label_obsolete_containers(self) -> bool:            
        """ 
        Check if at least one obsolete replica has the copy label, 'c'.
        :return: True if no obsolete replicas have label 'c'.
        """
        for node in self._recycling.obsolete_containers:
            for cc in list(self._recycling.obsolete_containers[node].keys()):
                if cc.label == "c":
                    return False
        return True

    def _debug_perf_surplus_balance(self) -> dict[Vmt, RequestsPerTime]:

        """
        Get the performance balance. The performance balance must be constant during the transition.
        This function is used for debugging purposes, which can be called in the debugger command line.
        """
        balance = dict(self._app_perf_surplus)
        for app, perf in self._app_perf_increment.items():
            balance[app] += perf
        for _, cc, replicas in self._unalloc_node_cs + self._allocatable_cs_next_step + \
                                  self._unallocated_containers_in_new_nodes:
            balance[cc.app] += cc.perf * replicas
        for _, cc_replicas in self._recycling.obsolete_containers.items():
            for cc, replicas in cc_replicas.items():
                balance[cc.app] -= cc.perf * replicas
        return balance
    
