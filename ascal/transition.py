"""
Implement synchronous transitions between two allocations as a list of commands.
A synchronous transition defines predefined times to start node/container creations/removals, assuming
fixed durations to perform these operations. A command starts when the previous command completes
and the node creation/upgrade time has elapsed if the command flag "sync_on_nodes_creation"/
"sync_on_nodes_upgrade" is set. 
A command ends and the next can be processed after completing its container removals and allocations, 
so the time required to execute a command is the sum of one container removal time and one container creation time.
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
container allocation is suspended until the destination node has enough computational resources.
"""

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


@dataclass
class Command:
    """
    The transition between an initial allocation and a final allocation is a sequence of commands.
    Each command can include several types of operations with nodes and containers.
    """
    allocate_containers: list[tuple[Vm|Vmt, ContainerClass, int]] = field(default_factory=list)
    remove_containers: list[tuple[Vm|Vmt, ContainerClass, int]] = field(default_factory=list)
    scale_up_containers: list[tuple[Vm|Vmt, ContainerClass, int, int]] = field(default_factory=list)
    scale_down_containers: list[tuple[Vm|Vmt, ContainerClass, int, int]] = field(default_factory=list)
    remove_nodes: list[Vm|Vmt] = field(default_factory=list)
    # Only the first command can create or upgrade nodes. Some commands must delay command execution until
    # completing node creations or node upgrades in the first command
    create_nodes: list[Vm|Vmt] = field(default_factory=list)
    upgrade_nodes: list[tuple[Vm|Vmt, InstanceClass]] = field(default_factory=list)
    sync_on_nodes_creation: bool = False
    sync_on_nodes_upgrade: bool = False

    def extend(self, command: 'Command'):
        """
        Extend with a new command, appending all the container and node operations.
        :param command: Command to extend.
        """
        self.allocate_containers.extend(command.allocate_containers)
        self.remove_containers.extend(command.remove_containers)
        self.scale_up_containers.extend(command.scale_up_containers)
        self.scale_down_containers.extend(command.scale_down_containers)
        self.create_nodes.extend(command.create_nodes)
        self.remove_nodes.extend(command.remove_nodes)
        self.upgrade_nodes.extend(command.upgrade_nodes)

    def vmt_to_vm(self):
        """
        Replaces the Vmt nodes in the command by Vm nodes.
        """
        self.allocate_containers = [(node.vm, cc, replicas) for node, cc, replicas in self.allocate_containers]
        self.remove_containers = [(node.vm, cc, replicas) for node, cc, replicas in self.remove_containers]
        self.scale_up_containers = [
            (node.vm, cc, multiplier, replicas) 
            for node, cc, multiplier, replicas in self.scale_up_containers
        ]
        self.scale_down_containers = [
            (node.vm, cc, divider, replicas) 
            for node, cc, divider, replicas in self.scale_up_containers
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
                len(self.scale_up_containers) == 0 and len(self.scale_down_containers) == 0 and
                len(self.create_nodes) == 0 and len(self.remove_nodes) == 0 and len(self.upgrade_nodes) == 0)

    def get_container_command_time(self, timing_args: TimedOps.TimingArgs) -> int:
        """
        Get the time required to complete a command ignoring node creation/removal operations. It assumes that
        container scale-up and scale-down operations take the same or less time as container creation and removal, 
        respectively.
        :param timing_args: Times for creation/removal of nodes/containers.
        :return: The time to complete the command.
        """
        container_command_time = 0
        if len(self.remove_containers) > 0:
            container_command_time += timing_args.container_removal_time
        elif len(self.scale_down_containers) > 0:
            container_command_time += timing_args.hot_container_scale_time
        if len(self.allocate_containers) > 0:
            container_command_time += timing_args.container_creation_time
        elif len(self.scale_up_containers) > 0:
            container_command_time += timing_args.hot_container_scale_time
        return container_command_time
    
    def simplification(self):
        """
        Labels in container classes are removed, and after that, performs the following simplifications:
        - Summing up replicas of the same container class in the same node.
        - Removing allocations and removals of the same containers.
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

        # Sum up replicas of the same container class in the same node and remove common allocations and removals
        allocs = defaultdict(int)
        for node, cc, replicas in self.allocate_containers:
            allocs[(node, cc)] += replicas
        removes = defaultdict(int)
        for node, cc, replicas in self.remove_containers:
            removes[(node, cc)] += replicas       
        for node, cc in allocs:
            if (node, cc) in removes:
                common_replicas = min(allocs[(node, cc)], removes[(node, cc)])
                allocs[(node, cc)] -= common_replicas
                removes[(node, cc)] -= common_replicas
        self.allocate_containers = [
            (node, cc, allocs[(node, cc)]) 
            for (node, cc) in allocs 
            if allocs[(node, cc)] > 0
        ]
        self.remove_containers = [
            (node, cc, removes[(node, cc)]) 
            for (node, cc) in removes 
            if removes[(node, cc)] > 0
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
        for node, cc, multiplier, replicas in self.scale_up_containers:
            if node in node_pairs:
                new_command.scale_up_containers.append((node_pairs[node], cc, multiplier, replicas))
            else:
                new_command.scale_up_containers.append((node, cc, multiplier, replicas))
        for node, cc, divider, replicas in self.scale_down_containers:
            if node in node_pairs:
                new_command.scale_down_containers.append((node_pairs[node], cc, divider, replicas))
            else:
                new_command.scale_down_containers.append((node, cc, divider, replicas))
        for node in self.create_nodes:
            if node in node_pairs:
                new_command.create_nodes.append(node_pairs[node])
            else:
                new_command.create_nodes.append(node)
        for node1, node2 in self.upgrade_nodes:
            if node1 in node_pairs:
                new_command.upgrade_nodes.append((node_pairs[node1], node2))
            else:
                new_command.upgrade_nodes.append((node1, node2))
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
    def get_recycled_node_pairs(self):
        """
        Get the recycled node pairs.
        :return: Recycled node pairs.
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

    def check_transition(self, initial_alloc: Allocation, final_alloc: Allocation, commands: list[Command]) -> bool:
        """
        Check the transition between the initial and final allocations.
        :param initial_alloc: Initial allocation.
        :param final_alloc: Final allocation.
        :param commands: List of commands to transition.
        :return: True if commands perform the required transition.
        :raises  ValueError: If some command is invalid.
        """
        min_perf, _ = get_min_max_perf(initial_alloc, final_alloc)
        initial_alloc_vmt = [Vmt(node) for node in initial_alloc]
        final_alloc_vmt = [Vmt(node) for node in final_alloc]
        vm_to_vmt = dict(zip(initial_alloc + final_alloc, initial_alloc_vmt + final_alloc_vmt))
        app_perf_surplus = get_app_perf_surplus(min_perf, initial_alloc_vmt)

        # Four commands must be enough for time_limit = 0
        assert self._time_limit > 0 or len(commands) <= 4, "Too many commands"

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
                node_vmt.free_mem += cc.mem[0] * replicas
                if (node_vmt.free_mem - node_vmt.ic.mem).magnitude > TransitionRBT._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container removal. Too many mem')
                
            # Container scaled-down commands
            for node, cc, divider, replicas in command.scale_down_containers:
                op_str = f'Command #{command_index}. Scale down containers ({node}, {cc}, {divider}, {replicas})'
                if node not in vm_to_vmt:
                    raise ValueError(f'{op_str} -> Invalid node: {node}')
                node_vmt = vm_to_vmt[node]
                if node_vmt.replicas[cc] < replicas:
                    raise ValueError(f'{op_str} -> Invalid container scale down. Replicas to scale down > allocated replicas')
                node_vmt.replicas[cc] -= replicas
                if node_vmt.replicas[cc] == 0:
                    del node_vmt.replicas[cc]
                scaled_down_cc = replace(
                    cc, 
                    cores=cc.cores / divider,
                    perf=cc.perf / divider,
                    agg_level=cc.agg_level / divider
                )
                node_vmt.replicas[scaled_down_cc] += replicas
                app_perf_surplus[cc.app] -= cc.perf * replicas * (divider - 1) / divider
                if app_perf_surplus[cc.app].magnitude < -Transition._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container scale down. app surplus < 0')
                node_vmt.free_cores += cc.cores * replicas * (divider - 1) / divider
                if (node_vmt.free_cores - node_vmt.ic.cores).magnitude > Transition._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container scale down. Too many cores')
                # Memory is no checked as it is not changed in scale down operations

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
                if node_vmt.free_cores.magnitude < -TransitionRBT._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container addition. Not enough cores are available')
                node_vmt.free_mem -= cc.mem[0] * replicas
                if node_vmt.free_mem.magnitude < -TransitionRBT._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container removal. Not enough memory is available')
            
            # Container scaled-down commands
            for node, cc, multiplier, replicas in command.scale_up_containers:
                op_str = f'Command #{command_index}. Scale up containers ({node}, {cc}, {multiplier}, {replicas})'
                if node not in vm_to_vmt:
                    raise ValueError(f'{op_str} -> Invalid node: {node}')
                node_vmt = vm_to_vmt[node]
                if node_vmt.replicas[cc] < replicas:
                    raise ValueError(f'{op_str} -> Invalid container scale up. Replicas to scale up > allocated replicas')
                node_vmt.replicas[cc] -= replicas
                if node_vmt.replicas[cc] == 0:
                    del node_vmt.replicas[cc]
                scaled_up_cc = replace(
                    cc, 
                    cores=cc.cores * multiplier,
                    perf=cc.perf * multiplier,
                    agg_level=cc.agg_level * multiplier
                )
                node_vmt.replicas[scaled_up_cc] += replicas
                app_perf_increment[cc.app] += cc.perf * replicas * (multiplier - 1)
                if app_perf_increment[cc.app].magnitude < -Transition._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container scale up. app surplus < 0')
                node_vmt.free_cores -= cc.cores * replicas * (multiplier - 1)
                if node_vmt.free_cores.magnitude < -Transition._DELTA:
                    raise ValueError(f'{op_str} -> Invalid container scale up. Not enough cores are available')
                # Memory is no checked as it is not changed in scale up operations 

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
        initial_alloc_signature = get_vmt_allocation_signature(initial_alloc_vmt)
        final_alloc_signature = get_vmt_allocation_signature(final_alloc_vmt)
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

    @staticmethod
    def get_transition_time(commands: list[Command], timing_args: TimedOps.TimingArgs) -> int:
        """
        Get the transition time from a list of commands. Node upgrade times are assumed to be lower 
        than node creation times, and container scale up/down times are assumed to be lower than 
        container creation/removal times.
        :param commands: A list of commands.
        :param timing_args: Creation/removal times of nodes and containers.
        :return: The transition time.
        """
        transition_time = 0
        last_node_removal_time = -1
        for command in commands:
            if len(command.create_nodes) > 0:
                assert commands.index(command) == 0, "Nodes must be created in the first command"
            if len(command.upgrade_nodes) > 0:
                assert commands.index(command) == 0, "Nodes must be upgraded in the first command"
            if command.sync_on_nodes_creation and len(commands[0].create_nodes) > 0:
                assert commands.index(command) > 0, "Invalid sync on first command"
                # Nodes are created in the first command. Nodes creation occurs in background
                transition_time = max(transition_time, timing_args.node_creation_time)
            if command.sync_on_nodes_upgrade and len(commands[0].upgrade_nodes) > 0:
                assert commands.index(command) > 0, "Invalid sync on first command"
                # Nodes are upgraded in the first command. Nodes upgrade occurs in background
                transition_time = max(transition_time, timing_args.hot_node_scale_up_time)
            # Add the maximum time required to perform container removals or container scale-downs
            incremental_time = 0
            if len(command.scale_down_containers) > 0:
                incremental_time = timing_args.hot_container_scale_time
            if len(command.remove_containers) > 0:
                incremental_time = timing_args.container_removal_time
            transition_time += incremental_time
            if len(command.remove_nodes) > 0:
                # Nodes can be removed in background. Calculate the latest time when a node removal finishes
                last_node_removal_time = transition_time + timing_args.node_removal_time
            # Add the maximum time required to perform container creations or container scale-ups
            incremental_time = 0
            if len(command.scale_up_containers) > 0:
                incremental_time = timing_args.hot_container_scale_time
            if len(command.allocate_containers) > 0:
                incremental_time = timing_args.container_creation_time
            transition_time += incremental_time

        return max(transition_time, last_node_removal_time)


class TransitionBaseline(Transition):
    """
    Class to implement the baseline transition algorithm. It calculates a transition with a single command that removes all the containers in the initial allocation and then creates all the containers in the final allocation, while creating and removing nodes at the beginning and at the end of the transition, respectively.
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
        :return: The creation total time.
        """
        return self._timing_args.node_creation_time + self._timing_args.container_creation_time
    
    def get_recycling_levels(self) -> tuple[float, float]:
        """
        Get node and container recycling levels for the last transition.
        :return: A tuple with node and container recycling levels.
        """
        return Recycling.INVALID_RECYCLING, Recycling.INVALID_RECYCLING

    def get_recycled_node_pairs(self):
        """
        Get the recycled node pairs.
        :return: Recycled node pairs.
        """
        return None

    @staticmethod
    def _check_node_equality(node1: Vm, node2: Vm) -> bool:
        """
        Check if two nodes are equal, i.e., they have the same instance class and the same allocated containers.
        :param node1: Node 1.
        :param node2: Node 2.
        :return: True if the nodes are equal.
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
        Calculate a synchronous transition from the initial allocation to the final allocation. It is a baseline algorithm.
        :param initial_alloc: Initial allocation.
        :param final_alloc: Final allocation.
        :return: A list of commands and the time to perform the transition.
        """
        self._commands = []
        initial_nodes = initial_alloc[:]
        final_nodes = final_alloc[:]

        # A transition is necessary whan some node or allocated containers change
        for initial_node in initial_nodes[:]:
            for final_node in final_nodes:
                if TransitionBaseline._check_node_equality(initial_node, final_node):
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
    two variants: 
    - time_limit = 0. Simplified version of the algorithm that focus on reducing the
    transition time. 
    - time_limit > 0. Complete version that focus on reducing the transition cost.
    """
    def __init__(self, timing_args: TimedOps.TimingArgs, system: System, time_limit: int = None,
                 hot_node_scale_up: bool = False, hot_replicas_scale: bool = False):
        """
        Creates an object for transition between two allocations.
        :param timing_args: Creation and removal times for containers and nodes.
        :param system: System performance and computational requirements.
        :param time_limit: Maximum time to carry out transitions. By default, this is set to the node creation time.
        :param hot_node_scale_up: Set to enable hot node scaling-up.
        :param hot_replicas_scale: Set to enable hot replicas scaling.
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
        self._hot_replicas_scale = hot_replicas_scale
        self._commands: list[Command] = None
        self._sync_on_next_alloc_upgraded_nodes = True

    def _remove_allocate(self, cc: ContainerClass, replicas: int, node: Vmt, command: Command, 
                         obsolete: bool=False) -> tuple[int, list[tuple[ContainerClass, int, Vmt]]]:
        """
        Allocate the container replicas to the node, freeing up computational resources in the node
        coming from obsolete containers if necessary, while ensuring that application's minimum performance
        constraints are met. The node state does not change when no replicas are allocated.
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

        # List of removed replicas (container class, number of replicas, node)
        removed_replicas = []

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
            int((node.free_cores.magnitude + removed_cores.magnitude + TransitionRBT._DELTA) / cc.cores.magnitude),
            int((node.free_mem.magnitude + removed_mem.magnitude + TransitionRBT._DELTA) / cc.mem[0].magnitude)
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
        :param obsolete: Set to True if the replicas to allocate are copies of obsolete replicas.
        :return: The number of actually allocated replicas.
        """
        allocatable_replicas = min(
            replicas,
            int((node.free_cores.magnitude + TransitionRBT._DELTA) / cc.cores.magnitude),
            int((node.free_mem.magnitude + TransitionRBT._DELTA) / cc.mem[0].magnitude)
        )
        if allocatable_replicas > 0:
            node.free_cores = node.free_cores - allocatable_replicas * cc.cores
            assert node.free_cores.magnitude > - TransitionRBT._DELTA, "Node free cores cannot not be negative"
            node.free_mem = node.free_mem - allocatable_replicas * cc.mem[0]
            assert node.free_mem.magnitude > - TransitionRBT._DELTA, "Node free memory cannot not be negative"
            node.replicas[cc] += allocatable_replicas
            command.allocate_containers.append((node, cc, allocatable_replicas))
            if not obsolete:
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
        Remove obsolete replicas from the container class in the node. The number of replicas actually 
        removed is limited by the number of replicas in the node and the application's 
        minimum performance constraint.
        :param cc: Container class.
        :param replicas: Replicas to remove.
        :param node: Node.
        :param command: Command with container removals.
        :param relaxed_removal: Set to True to use the application's performance increment instead of
        the application's performance surplus to perform the removal. This is useful when removing copies
        of obsolete containers that are copied again to other nodes.
        :return: Number of replicas that are actually removed.
        """
        # Remove any label from the obsolete container class
        if cc.label != "":
            cc = replace(cc, label="")
        if cc not in node.replicas:
            return 0
        performance_surplus = self._app_perf_increment if relaxed_removal else self._app_perf_surplus
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
        node.free_mem += cc.mem[0] * removed_replicas
        assert (node.free_mem - node.ic.mem).magnitude < TransitionRBT._DELTA, "Invalid node free mem"
        performance_surplus[cc.app] -= cc.perf * removed_replicas
        self._recycling.obsolete_containers[node][cc] -= removed_replicas
        if self._recycling.obsolete_containers[node][cc] == 0:
            del self._recycling.obsolete_containers[node][cc]
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
            if (node.ic.cores - node.free_cores).magnitude < TransitionRBT._DELTA and \
                    (node.ic.mem - node.free_mem).magnitude < TransitionRBT._DELTA:
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
                    surplus_replicas = floor(cc_perf_surplus / cc.perf + TransitionRBT._DELTA)
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
            for node, _ in sorted(zip(allocated_nodes_list, free_capacities), key=lambda x: x[1], reverse=True)
        ]
        empty_nodes_list.sort(key=lambda n: n.ic.price)

        return allocated_nodes_list + empty_nodes_list

    def _remove_last_obsolete_containers(self, command: Command):
        """
        Remove the las obsolete containers of applications with all its new containers allocated.
        :param command: A command with the removal of containers.
        """
        for node, obsolete_cc_replicas in self._recycling.obsolete_containers.items():
            for obsolete_cc, replicas in dict(obsolete_cc_replicas.items()).items():
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

    def _update_next_copy_state(self, allocatable_replicas:int, cc: ContainerClass, src_node: Vmt, 
                               copied_obsolete_replicas: list[tuple[ContainerClass, int, Vmt]], 
                               removed_obsolete_replicas: list[tuple[ContainerClass, int, Vmt]],
                               available_obsolete_containers: dict[Vmt, dict[ContainerClass, int]], 
                               available_node_free_resources: list[Vmt, tuple[float, float]]) \
                                -> Command:
        """
        Update the state for the copy of obsolete containers, associated to the allocation of the following 
        new container, in the next remove-allocate-copy execution.
        :param allocatable_replicas: Number of allocatable replicas of the new container.
        :param cc: Container class for the new container.
        :param src_node: Source node for the new_container.
        :param copied_obsolete_replicas: List of obsolete replicas copied to allocate the new container.
        :param removed_obsolete_replicas: List of obsolete replicas removed in the destination nodes 
        to allocate the new container.
        :param available_obsolete_containers: Available obsolete containers after the allocation of the new container.
        :param available_node_free_resources: Available free computational resources after the allocation of
        the new container.
        :return: A command with the removal of copies of obsolete containers in the source node.
        """
        command = Command()

        # The allocation of allocatable_replicas of the new container will reduce the free cores and memory 
        # of the source node, whereas the copied obsolete replicas will increase the free cores and memory
        free_src_cores, free_src_mem = available_node_free_resources[src_node]
        for copied_cc, copied_replicas, _ in copied_obsolete_replicas:
            free_src_cores += copied_replicas * copied_cc.cores
            free_src_mem += copied_replicas * copied_cc.mem[0]
        free_src_cores -= allocatable_replicas * cc.cores
        free_src_mem -= allocatable_replicas * cc.mem[0]
        available_node_free_resources[src_node] = (free_src_cores, free_src_mem)

        # Removed obsolete replicas in destination nodes will increase free cores and memory 
        # whereas the copied obsolete replicas will reduce free cores and memory
        for removed_cc, removed_replicas, dest_node in removed_obsolete_replicas:
            free_dst_cores, free_dst_mem = available_node_free_resources[dest_node]
            free_dst_cores += removed_replicas * removed_cc.cores
            free_dst_mem += removed_replicas * removed_cc.mem[0]
            available_node_free_resources[dest_node] = (free_dst_cores, free_dst_mem)
        for copied_cc, copied_replicas, dest_node in copied_obsolete_replicas:
            free_dst_cores, free_dst_mem = available_node_free_resources[dest_node]
            free_dst_cores -= copied_replicas * copied_cc.cores
            free_dst_mem -= copied_replicas * copied_cc.mem[0]
            available_node_free_resources[dest_node] = (free_dst_cores, free_dst_mem)

        # Copied obsolete replicas will not be longer available in the source node, but can be used
        # to free up space in the destination node
        for copied_cc, copied_replicas, dest_node in copied_obsolete_replicas:
            available_obsolete_containers[src_node][copied_cc] -= copied_replicas
            if available_obsolete_containers[src_node][copied_cc] == 0:
                del available_obsolete_containers[src_node][copied_cc]
            if copied_cc.label == "c": # It is a copy of an obsolete container
                labelled_copied_cc = copied_cc
                self._remove_obsolete_replicas(labelled_copied_cc, copied_replicas, src_node, 
                                               command, relaxed_removal=True) 
            else:
                labelled_copied_cc = replace(copied_cc, label="c")
            if dest_node not in available_obsolete_containers:
                available_obsolete_containers[dest_node] = {labelled_copied_cc: copied_replicas}
            else:
                if labelled_copied_cc not in available_obsolete_containers[dest_node]:
                    available_obsolete_containers[dest_node][labelled_copied_cc] = copied_replicas
                else:
                    available_obsolete_containers[dest_node][labelled_copied_cc] += copied_replicas

        # Removed obsolete replicas will not be longer available in the destination nodes
        for removed_cc, removed_replicas, node in removed_obsolete_replicas:
            available_obsolete_containers[node][removed_cc] -= removed_replicas
            if available_obsolete_containers[node][removed_cc] == 0:
                del available_obsolete_containers[node][removed_cc]

        return command
    
    def _copy_obsolete_containers(self, node_cc_replicas: tuple[Vmt, ContainerClass, int],
                                  available_obsolete_containers: dict[Vmt, dict[ContainerClass, int]],
                                  available_node_free_resources: dict[Vmt, tuple[int, int]],
                                  dest_nodes: list[Vmt]) -> tuple[int, Command]:
        """
        Copy obsolete replicas from a source node to destination nodes, removing obsolete containers from
        the destination nodes if necessary. This operation sets up the allocation of new containers for the next
        remove-allocate-copy execution.
        The copy process depends of previous copies in the same call of remove-execute-copy, as it modifies 
        the free computational resources and obsolete containers in the nodes.
        :param node_cc_replicas: Source node, container class and replicas for the new containers to allocate 
        in the next call to the remove-allocate-copy algorithm.
        :param available_obsolete_containers: Elegible obsolete containers in all the nodes. They are updated
        when at least a new container replica will be allocatable in the next remove-allocate-copy execution.
        :param available_node_free_resources: Free computational resources in the node. They are updated when
        at least a new container replica will be allocatable in the next remove-allocate-copy execution.
        :param dest_nodes: Elegible destination node to allocate copies of obsolete containers.
        :return: A tuple with the number of allocatable new containers and a command with the operations.
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
        required_mem = replicas_to_allocate * cc.mem[0]

        # The state must be recovered when we fail to allocate at least one replica, so create backups.
        # The method that modifies the state is _remove_allocate(), which is called for each destination node.
        # This method modifies: destination nodes states, application performance surplus and increment,
        # and obsolete containers in nodes
        dest_nodes_modified = []
        dest_nodes_backup = {
            node: (node.free_cores, node.free_mem, defaultdict(lambda: 0, node.replicas))
            for node in dest_nodes
        }
        zero_perf = RequestsPerTime("0 req/s")
        app_perf_surplus_backup = defaultdict(lambda: zero_perf, self._app_perf_surplus)
        app_perf_increment_backup = defaultdict(lambda: zero_perf, self._app_perf_increment)
        obsolete_containers_backup = {
            node: {cc: replicas for cc, replicas in cc_replicas.items()}
            for node, cc_replicas in self._recycling.obsolete_containers.items()
        }
        obsolete_containers_backup = defaultdict(lambda: 0, dict(obsolete_containers_backup))

        # Allocate obsolete container copies in destination nodes

        command = Command()

        # In case of success, these obsolete replicas will be no longer elegible. Available obsolete containers
        # and free computational resources in nodes are updated with the copies        
        copied_obsolete_replicas = []

        # In case of success, these obsolete replicas will be removed from destination nodes. Available obsolete
        # containers and free computational resources in nodes are updated with the removals
        removed_obsolete_replicas = []

        for removable_cc, available_replicas in available_src_node_obsolete_containers.items():
            # Required obsolete containers to free up enough computational resurces to allocate
            # replicas_to_allocate replicas in the node, considering rounding errors and the maximum
            # number of available replicas
            required_obsolete_replicas = max(
                ((required_cores - free_src_cores) / removable_cc.cores).magnitude,
                ((required_mem - free_src_mem) / removable_cc.mem[0]).magnitude
            )
            required_obsolete_replicas = int(ceil(required_obsolete_replicas - TransitionRBT._DELTA))
            replicas_to_remove = min(required_obsolete_replicas, available_replicas)

            # If no more replicas are required to free up enough computational resources
            if replicas_to_remove <= 0:
                break

            # Try copying required_obsolete_replicas to other nodes
            for dest_node in dest_nodes:
                if dest_node == src_node:
                    # Cannot copy obsolete containers to the same node
                    continue
                obsolete_replicas, removed_replicas = self._remove_allocate(removable_cc, replicas_to_remove,
                                                                            dest_node, command, obsolete=True)
                if obsolete_replicas > 0:
                    copied_obsolete_replicas.append((removable_cc, obsolete_replicas, dest_node))
                    removed_obsolete_replicas.extend(removed_replicas)
                    dest_nodes_modified.append(dest_node)
                    required_cores -= obsolete_replicas * removable_cc.cores
                    required_mem -= obsolete_replicas * removable_cc.mem[0]
                    replicas_to_remove -= obsolete_replicas
                    if replicas_to_remove == 0:
                        # We have copied all the required obsolete replicas to free up enough resources
                        break

        # Calculate the number of replicas that would be allocatable in the next remove-allocate-copy step
        free_cores = free_src_cores + replicas_to_allocate * cc.cores - required_cores
        free_mem = free_src_mem + replicas_to_allocate * cc.mem[0] - required_mem
        allocatable_replicas = min(
            replicas_to_allocate,
            int((free_cores.magnitude + TransitionRBT._DELTA) / cc.cores.magnitude),
            int((free_mem.magnitude + TransitionRBT._DELTA) / cc.mem[0].magnitude),
        )
        if allocatable_replicas == 0:
            # Recover from backups
            for mod_node in dest_nodes_modified:
                mod_node.free_cores, mod_node.free_mem, mod_node.replicas = dest_nodes_backup[mod_node]
            self._app_perf_surplus = app_perf_surplus_backup
            self._app_perf_increment = app_perf_increment_backup
            self._recycling.obsolete_containers = obsolete_containers_backup
        else:
            # The allocation is preserved and the state is updated for the next new containers to allocate
            command2 = self._update_next_copy_state(allocatable_replicas, cc, src_node,
                                                    copied_obsolete_replicas, removed_obsolete_replicas,
                                                    available_obsolete_containers, available_node_free_resources)            
            # Extend the comand with the removal of copies of obsolete containers in the source node
            command.extend(command2)

        return allocatable_replicas, command

    def _allocate_with_free_obsolete(self, command: Command):
        """
        Allocate unallocated new containers using free computational resources in the same node and removing    
        obsolete containers from the same node if it were necessary.
        :param command: A command with container allocations and removals.
        """
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
        :return: A command with node removals, containers removal and containers allocation.
        """

        # Check that obsolete containers have not the copy label 'c'. It can be commented in production
        assert self._debug_check_label_obsolete_containers(), "Obsolete containers are not properly labelled"
        
        command = Command()

        # Remove the last obsolete containers of applications with all its new containers allocated
        self._remove_last_obsolete_containers(command)

        # Allocate container replicas prepared in a previous copy phase of the algorithm. They must be allocatable
        for src_node, cc, allocatable_replicas in self._allocatable_cs_next_step:
            # Allocate using free computational resources on the same node and removing obsolete
            # containers from the same node if it were necessary
            allocated_replicas, _ = self._remove_allocate(cc, allocatable_replicas, src_node, command)
            assert allocated_replicas == allocatable_replicas, "Containers must be allocatable"
        self._allocatable_cs_next_step.clear()

        # Allocate using free computational resources in the same node and removing obsolete
        # containers from the same node if it were necessary
        self._allocate_with_free_obsolete(command)

        if len(self._unalloc_node_cs) == 0:
            # Check if obsolete nodes can be removed and update the command
            self._remove_obsolete_nodes(command)
            # Remove common allocations and removals. One container can be allocated and removed in the same command
            # when an obsolete container is copied from node 1 to node 2, later from node 2 to node 3, and so on
            command.simplification()  

            return command

        # Nodes are sorted by decreasing freeable capacity, leaving empty nodes at the end, sorted by
        # increasing price
        if copy_nodes is None:
            copy_nodes = self._current_alloc
        copy_nodes = self._get_sorted_nodes(copy_nodes)

        # Next, try copying obsolete containers from the node to other nodes (destination nodes),
        # yielding enough application's performance surplus to allocate the replicas of unallocated
        # containers in the next transition step

        # Nodes elegible to allocate copies of obsolete containers        
        elegible_nodes = copy_nodes[:] 

        # Available obsolete containers for copy in each node
        available_obsolete_containers = {
            node: {cc: replicas for cc, replicas in cc_replicas.items()}
            for node, cc_replicas in self._recycling.obsolete_containers.items()
        }

        # Available free computational resources in the node in the next transtion step 
        available_node_free_resources = {
            node: (node.free_cores, node.free_mem) for node in self._current_alloc
        }

        for unalloc_node_cc in self._unalloc_node_cs[:]:
            src_node, cc, replicas_to_allocate = unalloc_node_cc

            # If the node has obsolete containers
            if src_node in available_obsolete_containers:
                # Copy obsolete containers from the source node to elegible nodes, setting up the allocation
                # of allocatable_replicas in the next remove-allocate-copy. This process updates the
                # available obsolete containers and the available free computational resources for the next copy. 
                # It returns the number of allocatable replicas in the source node at the next transition step and 
                # a command with the allocation of obsolete containers in elegible nodes 
                allocatable_replicas, command2 = \
                    self._copy_obsolete_containers(unalloc_node_cc, available_obsolete_containers, 
                                                   available_node_free_resources, elegible_nodes)
                if allocatable_replicas > 0:
                    # Remove the node from the elegible nodes, since it will allocate new containers in the next step
                    if src_node in elegible_nodes:
                        elegible_nodes.remove(src_node)
                    if self._time_limit > 0:
                        # Complete the list of containers allocatable in the next transisition step and remove them
                        # from the list of unallocated containers
                        self._allocatable_cs_next_step.append((src_node, cc, allocatable_replicas))
                        index = self._unalloc_node_cs.index((src_node, cc, replicas_to_allocate))
                        replicas_to_allocate -= allocatable_replicas
                        if replicas_to_allocate > 0:
                            self._unalloc_node_cs[index] = (src_node, cc, replicas_to_allocate)
                        else:
                            self._unalloc_node_cs.pop(index)
                    command.extend(command2)

        # Remove common allocations and removals. One container can be allocated and removed in the same command
        # when an obsolete container is copied from node 1 to node 2, later from node 2 to node 3, and so on
        command.simplification()  

        # Check if obsolete nodes can be removed and update the command
        self._remove_obsolete_nodes(command)

        return command

    def _get_allocation(self, app_performance: dict[App, RequestsPerTime]) -> list[Vm]:
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
        - For each obsolete node, remove all except its last remove operation. Note that obsolete nodes are not
        actually removed from the allocation.
        - Remove empty commands.
        - Replace Vmt nodes by Vm nodes.
        - Remove obsolete nodes from the current allocation.
        """
        # Node removal commands are generated when all the containers of an obsolete node are removed.
        # However, the nodes could be useful in future to help in the transition of recycled nodes, so
        # they could be used after a removal command. A node removal command is removed when there is
        # a later allocation in the node.
        null_command = Command() 
        node_removal_command = defaultdict(lambda: null_command) # Start with no removals for any node
        nodes_removed_once = set()
        for command in self._commands:
            for node, _, _ in command.allocate_containers:
                if not node_removal_command[node].is_null():
                    removal_command = node_removal_command[node]
                    removal_command.remove_nodes.remove(node)
                    removal_command = null_command
            for node_to_remove in command.remove_nodes:
                nodes_removed_once.add(node_to_remove)
                if node_removal_command[node_to_remove].is_null():
                    # First removal command for the node or the previous removal command has been removed
                    node_removal_command[node_to_remove] = command

        # Check the removals
        for node in nodes_removed_once:
            assert node in node_removal_command, "Node removal command not found"
            assert not node_removal_command[node].is_null(), "Node removal command is null"

        # Remove empty commands
        for command in self._commands:
            if command.is_null():
                self._commands.remove(command)

        # Remove the nodes from the current allocation
        for node, command in node_removal_command.items():
            if not command.is_null():
                # The node is removed in the last removal command
                self._current_alloc.remove(node)

        # Replace Vmt nodes by nodes Vm nodes in the commands
        for command in self._commands:
            command.vmt_to_vm()

    def _allocation_loop(self, max_time: int) -> int:
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

    def _complete_allocation_in_temporary_nodes(self, create_nodes_command: Command,
                                               allocate_new_nodes_command: Command):
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
        # remaining containers
        copy_nodes = None if self._time_limit > 0 else [dummy_node]
        command = self._remove_allocate_copy(copy_nodes)
        zero_rps = RequestsPerTime("0 rps")
        tmp_app_perf = {app: zero_rps for app in self._app_perf_surplus}
        for node, cc, replicas in command.allocate_containers:
            if node == dummy_node:
                # The allocations in the dummy node will be performed in a previous command, so
                # the performance surplus can be inmediately updated. It is required when
                # time_limit == to avoid and additional command. It is related to the
                # self._allocate_with_free_obsolete(command) call at the end of this method
                self._app_perf_increment[cc.app] -= replicas * cc.perf
                self._app_perf_surplus[cc.app] += replicas * cc.perf
                tmp_app_perf[cc.app] += cc.perf * replicas

        # Get an allocation for application's performance on temporary nodes
        tmp_nodes = [Vmt(node) for node in self._get_allocation(tmp_app_perf)]
        for tmp_node_index in range(len(tmp_nodes)):
            # Change the id of temporary nodes to negative values to be easily identified
            tmp_nodes[tmp_node_index].id = -(tmp_node_index + 1)
            tmp_nodes[tmp_node_index].vm.id = -(tmp_node_index + 1)
        create_nodes_command.create_nodes.extend(tmp_nodes)
        containers_in_tmp_nodes = [
            (node, cc, replicas)
            for node in tmp_nodes for cc, replicas in node.replicas.items()
        ]

        # Move allocations in the command from the dummy node to the temporary nodes
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

        # Replace the dummy node with temporary nodes
        self._current_alloc.remove(dummy_node)
        self._current_alloc.extend(tmp_nodes)

        # Some replicas can be allocated after the copies to temporary nodes
        if self._time_limit == 0:
            self._allocate_with_free_obsolete(command)

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

    def _debug_check_label_obsolete_containers(self) -> bool:            
        """ Check if at least one obsolete replica in self._recycling.obsolete_containers has label 'c'
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

        # Allocation loop until node upgrading completes
        elapsed_time = 0
        if self._time_limit > 0:
            elapsed_time = self._allocation_loop(self._timing_args.hot_node_scale_up_time)

        # Update the elapsed time and current allocation after completing the node upgrading
        elapsed_time = max(elapsed_time, self._timing_args.hot_node_scale_up_time)
        for initial_node, final_node in self._recycling.upgraded_node_pairs.items():
            initial_node.upgrade(final_node)

        # Allocation loop until node creation completes
        if self._time_limit > 0:
            max_time = self._timing_args.node_creation_time - self._timing_args.hot_node_scale_up_time
            elapsed_time += self._allocation_loop(max_time)

        # Update the elapsed time and current allocation after completing the node creation
        elapsed_time = max(elapsed_time, self._timing_args.node_creation_time)
        self._current_alloc.extend(self._recycling.new_nodes)

        # Allocate new containers in new nodes. The corresponding command may be empty if there
        # are no new nodes in the final allocation. This command can be extended later to include
        # containers in temporary nodes.
        allocate_in_new_nodes_command = Command(allocate_containers=self._unallocated_containers_in_new_nodes[:])
        allocate_in_new_nodes_command.sync_on_nodes_creation = True
        for _, cc, replicas in self._unallocated_containers_in_new_nodes:
            self._app_perf_increment[cc.app] += replicas * cc.perf
            self._app_unalloc_perf[cc.app] -= replicas * cc.perf
            assert self._app_unalloc_perf[cc.app].magnitude >= - TransitionRBT._DELTA, "Invalid performance"
        self._unallocated_containers_in_new_nodes.clear()
        self._append_command(allocate_in_new_nodes_command, append_null_command=True)

        # Allocation loop until the time limit. It does not execute with time limit = 0
        if elapsed_time < self._time_limit:
            self._allocation_loop(self._time_limit - elapsed_time)

        # If there are still unallocated containers in recycled nodes
        if len(self._allocatable_cs_next_step) > 0 or len(self._unalloc_node_cs) > 0:
            if len(self._unalloc_node_cs) == 0 and self._time_limit > 0:
                # Extend a little the transition time instead of creating temporary nodes
                self._append_command(self._remove_allocate_copy())
            # If new nodes are not enough to complete the transition of recycled nodes, temporary nodes are required
            else:
                self._complete_allocation_in_temporary_nodes(create_upgrade_nodes_command, allocate_in_new_nodes_command)

        # Two remove-allocate-copy steps may be necessary to complete the transition. The first
        # command to remove obsolete containers and next allocate the remaining new containers in recycled nodes.
        # The second command to remove containers from the nodes and next the nodes
        first_command = self._remove_allocate_copy()
        if not first_command.is_null():
            self._append_command(first_command)
        second_command = self._remove_allocate_copy()
        if not second_command.is_null():
            self._append_command(second_command)

        # Post-processing operations to obtain the final command list
        self._post_process_commands()

        # Check whether the commands implement a valid transition between the initial and the final allocations
        assert self.check_transition(initial_alloc, final_alloc, self._commands), "Invalid transition"

        return self._commands, self.get_transition_time(self._commands, self._timing_args)

    def get_worst_case_transition_time(self) -> int:
        """
        Get the worst-case transition time.
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

    def get_recycling_levels(self) -> tuple[float, float]:
        """
        Get node and container recycling levels for the last transition.
        :return: A tuple with node and container recycling levels.
        """
        return self._recycling.node_recycling_level, self._recycling.container_recycling_level

    def get_recycled_node_pairs(self):
        """
        Get the recycled node pairs.
        :return: Recycled node pairs.
        """
        return self._recycling_vm.recycled_node_pairs



class TransitionRBT2(Transition):
    """
    Class to perform transitions between initial allocation and final allocations for a given system.
    There are two types of transitions: synchronous and asynchronous. In addition, there are
    two variants of each type: one for time_limit = 0 (that tries to minimize the transition time)
    and other for time_limit > 0.
    """
    # Constant used to deal with numerical approximations
    _DELTA = 0.000001

    def __init__(self, timing_args: TimedOps.TimingArgs, system: System, time_limit: int = None,
                 hot_node_scale_up: bool = False, hot_replicas_scale: bool = False):
        """
        Creates an object for transition between two allocations.
        :param timing_args: Times for operations on containers and nodes.
        :param system: Applications's performance on different container classes and available node types.
        :param time_limit: Maximum time to carry out transitions. By default, this is set to the node creation time. 
        It is a best-effort limit.
        :param hot_node_scale_up: Set to enable hot node scaling-up.
        :param hot_replicas_scale: Set to enable hot scaling of container computational parameters.
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
        self._hot_replicas_scale = hot_replicas_scale
        self._commands: list[Command] = None
        self._sync_on_next_alloc_upgraded_nodes = True
        self._minimize_transition_commands = time_limit == 0
    
    def _remove_allocate(self, cc: ContainerClass, replicas: int, node: Vmt, command: Command, 
                         obsolete: bool=False) -> tuple[int, list[tuple[ContainerClass, int, Vmt]]]:
        """
        Allocate the container replicas to the node, freeing up computational resources in the node
        coming from obsolete containers if necessary, while ensuring that application's minimum performance
        constraints are met. The node state does not change when no replicas are allocated.
        Current implementation assumes a single family of container classes.
        :param cc: Container class.
        :param replicas: The number of replicas to allocate.
        :param node: Node.
        :param command: Command with (obsolete) containers to be removed and containers to be allocated.
        :param obsolete: True if the containers to allocate are copies of obsolete containers.
        :return: The number of actually allocated replicas and a list removed replicas.
        """

        # Remove any label from the container class
        #----REVIEW
        if cc.label != "":
            cc = replace(cc, label="")

        # List of removed replicas (container class, number of replicas, node)
        removed_replicas = []

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

                # Get the maximum number of replicas of the obsolete container that could be removed
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
            int((node.free_cores.magnitude + removed_cores.magnitude + TransitionRBT._DELTA) / cc.cores.magnitude),
            int((node.free_mem.magnitude + removed_mem.magnitude + TransitionRBT._DELTA) / cc.mem[0].magnitude)
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
        :param obsolete: Set to True if the replicas to allocate are copies of obsolete replicas.
        :return: The number of actually allocated replicas.
        """
        allocatable_replicas = min(
            replicas,
            int((node.free_cores.magnitude + TransitionRBT._DELTA) / cc.cores.magnitude),
            int((node.free_mem.magnitude + TransitionRBT._DELTA) / cc.mem[0].magnitude)
        )
        if allocatable_replicas > 0:
            node.free_cores = node.free_cores - allocatable_replicas * cc.cores
            assert node.free_cores.magnitude > - TransitionRBT._DELTA, "Node free cores cannot not be negative"
            node.free_mem = node.free_mem - allocatable_replicas * cc.mem[0]
            assert node.free_mem.magnitude > - TransitionRBT._DELTA, "Node free memory cannot not be negative"
            node.replicas[cc] += allocatable_replicas
            command.allocate_containers.append((node, cc, allocatable_replicas))
            if not obsolete:
                # Decrement the application performance coming from its unallocated containers
                self._app_unalloc_perf[cc.app] -= allocatable_replicas * cc.perf
                assert self._app_unalloc_perf[cc.app].magnitude > -TransitionRBT._DELTA, "Invalid performance"
            else:
                if node not in self._recycling.obsolete_containers:
                    self._recycling.obsolete_containers[node] = {cc: allocatable_replicas}
                else:
                    # Add the allocated replicas to the obsolete containers in the node
                    if cc not in self._recycling.obsolete_containers[node]:
                        self._recycling.obsolete_containers[node][cc] = allocatable_replicas
                    else:
                        self._recycling.obsolete_containers[node][cc] += allocatable_replicas
            # Extra application application performance surplus to be added
            # at the end of the current command
            self._app_perf_increment[cc.app] += allocatable_replicas * cc.perf
        return allocatable_replicas

    def _remove_obsolete_replicas(self, cc: ContainerClass, replicas: int, node: Vmt, command: Command,
                                  relaxed_removal: bool = False) -> int:
        """
        Remove obsolete replicas from the container class in the node. The number of replicas actually 
        removed is limited by the number of replicas in the node and a minimum application's 
        performance constraint.
        :param cc: Container class.
        :param replicas: Replicas to remove.
        :param node: Node.
        :param command: Command with container removals.
        :param relaxed_removal: Set to True to use the application's performance increment instead of
        the application's performance surplus to perform the removal. This relaxed performance contraint
        applies to obsolete containers created in the same command.
        :return: Number of replicas that are actually removed.
        """
        # Remove any label from the obsolete container class
        if cc.label != "":
            cc = replace(cc, label="")
        if cc not in node.replicas:
            return 0
        performance_surplus = self._app_perf_increment if relaxed_removal else self._app_perf_surplus
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
        node.free_mem += cc.mem[0] * removed_replicas
        assert (node.free_mem - node.ic.mem).magnitude < TransitionRBT._DELTA, "Invalid node free mem"
        performance_surplus[cc.app] -= cc.perf * removed_replicas
        self._recycling.obsolete_containers[node][cc] -= removed_replicas
        if self._recycling.obsolete_containers[node][cc] == 0:
            del self._recycling.obsolete_containers[node][cc]
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

    def _get_free_capacity_nodes(self, nodes: list[Vm] = None) -> list[Vmt]:
        """
        Get nodes with free or freeable capacity. 
        :param nodes: Nodes to sort or all the nodes in the allocation when this parameter is no set.
        :return: A list of nodes with free or freeable capacity. Nodes that do not allocate containers are returned at 
        the end of the list sorted by increasing price.
        """
        if nodes is None:
            nodes = self._current_alloc
        empty_nodes = []
        allocated_nodes = []

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
                    free_mem += surplus_replicas * cc.mem[0]
                    app_perf_surplus[app] -= cc_perf_surplus
            if free_cores.magnitude > Transition._DELTA and free_mem.magnitude > Transition._DELTA:
                allocated_nodes.append(node)
        sorted_empty_nodes = sorted(empty_nodes, key=lambda n: n.ic.price)
        return allocated_nodes + sorted_empty_nodes
        
    def remove_last_obsolete_containers(self, command: Command):
        """
        Remove the last obsolete containers of applications with all its new containers allocated.
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
                assert (node.free_cores - node.ic.cores).magnitude < TransitionRBT._DELTA, "Can not remove the node"
                assert (node.free_mem - node.ic.mem).magnitude < TransitionRBT._DELTA, "Can not remove the node"
                command.remove_nodes.append(node)
                if node in self._recycling.obsolete_containers:
                    del self._recycling.obsolete_containers[node] # Obsolete containers in the node with zero replicas
                self._recycling.obsolete_nodes.remove(node)

    def _remove_allocate_copy_init(self, min_perf: dict[App, RequestsPerTime]):
        """
        Initialize the remove-allocate-copy algorithm.
        :param min_perf: Minimum application performances, to be fulfilled during the transition.
        """
        # Calculate the total cores and memory of new containers in recycled and upgraded nodes.
        # They are necessary to calculate relative container sizes
        total_new_cpu = 0
        total_new_mem = 0
        for n, cc_replicas in self._recycling.new_containers.items():
            # We focus on new containers in recycled or upgraded nodes
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
            for _, new_replicas in sorted(zip(container_sizes, self._unalloc_node_cs), key=lambda x: x[0],
                                             reverse = True)
        ]

        # Unallocated application performances. It is calculated from new containers not allocated yet
        self._app_unalloc_perf = defaultdict(lambda : RequestsPerTime("0 req/s"))
        for _, cc_replicas in self._recycling.new_containers.items():
            for cc, replicas in cc_replicas.items():
                self._app_unalloc_perf[cc.app] += replicas * cc.perf

        # Initial application's performance surplus
        self._app_perf_surplus = get_app_perf_surplus(min_perf, self._current_alloc)
        assert min(self._app_perf_surplus.values()).magnitude >= 0, "Invalid performance surplus"

        # Performance increment of application performance surplus at the end of the command, once
        # all the allocations in the command have been performed
        self._app_perf_increment = defaultdict(lambda : RequestsPerTime("0 req/s"))

        # List of containers that will be allocatable in the next execution of the remove-allocate-copy algorithm
        self._allocatable_cs_next_step = []

    def _update_next_copy_state(self, allocatable_replicas:int, cc: ContainerClass, src_node: Vmt, 
                               copied_obsolete_replicas: list[tuple[ContainerClass, int, Vmt]], 
                               removed_obsolete_replicas: list[tuple[ContainerClass, int, Vmt]],
                               available_obsolete_containers: dict[Vmt, dict[ContainerClass, int]], 
                               available_node_free_resources: list[Vmt, tuple[float, float]]) \
                                -> Command:
        """
        Update the state for the copy of obsolete containers, associated to the allocation of the following 
        new container, in the next remove-allocate-copy execution.
        :param allocatable_replicas: Number of allocatable replicas of the new container.
        :param cc: Container class for the new container.
        :param src_node: Source node for the new_container.
        :param copied_obsolete_replicas: List of obsolete replicas copied to allocate the new container.
        :param removed_obsolete_replicas: List of obsolete replicas removed in the destination nodes 
        to allocate the new container.
        :param available_obsolete_containers: Available obsolete containers after the allocation of the new container.
        :param available_node_free_resources: Available free computational resources after the allocation of
        the new container.
        :return: A command with the removal of copies of obsolete containers in the source node.
        """
        command = Command()

        # The allocation of allocatable_replicas of the new container will reduce the free cores and memory 
        # of the source node, whereas the copied obsolete replicas will increase the free cores and memory
        free_src_cores, free_src_mem = available_node_free_resources[src_node]
        for copied_cc, copied_replicas, _ in copied_obsolete_replicas:
            free_src_cores += copied_replicas * copied_cc.cores
            free_src_mem += copied_replicas * copied_cc.mem[0]
        free_src_cores -= allocatable_replicas * cc.cores
        free_src_mem -= allocatable_replicas * cc.mem[0]
        available_node_free_resources[src_node] = (free_src_cores, free_src_mem)

        # Removed obsolete replicas in destination nodes will increase free cores and memory 
        # whereas the copied obsolete replicas will reduce free cores and memory
        for removed_cc, removed_replicas, dest_node in removed_obsolete_replicas:
            free_dst_cores, free_dst_mem = available_node_free_resources[dest_node]
            free_dst_cores += removed_replicas * removed_cc.cores
            free_dst_mem += removed_replicas * removed_cc.mem[0]
            available_node_free_resources[dest_node] = (free_dst_cores, free_dst_mem)
        for copied_cc, copied_replicas, dest_node in copied_obsolete_replicas:
            free_dst_cores, free_dst_mem = available_node_free_resources[dest_node]
            free_dst_cores -= copied_replicas * copied_cc.cores
            free_dst_mem -= copied_replicas * copied_cc.mem[0]
            available_node_free_resources[dest_node] = (free_dst_cores, free_dst_mem)

        # Copied obsolete replicas will not be longer available in the source node, but can be used
        # to free up space in the destination node
        for copied_cc, copied_replicas, dest_node in copied_obsolete_replicas:
            available_obsolete_containers[src_node][copied_cc] -= copied_replicas
            if available_obsolete_containers[src_node][copied_cc] == 0:
                del available_obsolete_containers[src_node][copied_cc]
            if copied_cc.label == "c": # It is a copy of an obsolete container
                labelled_copied_cc = copied_cc
                self._remove_obsolete_replicas(labelled_copied_cc, copied_replicas, src_node, 
                                               command, relaxed_removal=True) 
            else:
                labelled_copied_cc = replace(copied_cc, label="c")
            if dest_node not in available_obsolete_containers:
                available_obsolete_containers[dest_node] = {labelled_copied_cc: copied_replicas}
            else:
                if labelled_copied_cc not in available_obsolete_containers[dest_node]:
                    available_obsolete_containers[dest_node][labelled_copied_cc] = copied_replicas
                else:
                    available_obsolete_containers[dest_node][labelled_copied_cc] += copied_replicas

        # Removed obsolete replicas will not be longer available in the destination nodes
        for removed_cc, removed_replicas, node in removed_obsolete_replicas:
            available_obsolete_containers[node][removed_cc] -= removed_replicas
            if available_obsolete_containers[node][removed_cc] == 0:
                del available_obsolete_containers[node][removed_cc]

        return command
    
    def _copy_obsolete_containers(self, node_cc_replicas: tuple[Vmt, ContainerClass, int],
                                  available_obsolete_containers: dict[Vmt, dict[ContainerClass, int]],
                                  available_node_free_resources: dict[Vmt, tuple[int, int]],
                                  dest_nodes: list[Vmt]) -> tuple[int, Command]:
        """
        Copy obsolete replicas from a source node to destination nodes, removing obsolete containers from
        the destination nodes if necessary. This operation sets up the allocation of new containers for the next
        remove-allocate-copy execution.
        The copy process depends of previous copies in the same call of the remove-execute-copy algorithm, 
        as it modifies the free computational resources and obsolete containers in the nodes.
        :param node_cc_replicas: Source node, container class and replicas for the new containers to allocate 
        in the next call to the remove-allocate-copy algorithm.
        :param available_obsolete_containers: Elegible obsolete containers in all the nodes. They are updated
        when at least a new container replica will be allocatable in the next remove-allocate-copy execution.
        :param available_node_free_resources: Free computational resources in the node. They are updated when
        at least a new container replica will be allocatable in the next remove-allocate-copy execution.
        :param dest_nodes: Elegible destination node to allocate copies of obsolete containers.
        :return: A tuple with the number of allocatable new containers and a command with the operations.
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
        required_mem = replicas_to_allocate * cc.mem[0]

        # The state must be recovered when we fail to allocate at least one replica, so create backups.
        # The state is changed by the remove-allocate algorithm executed on each destination node.
        dest_nodes_modified = []
        dest_nodes_backup = {
            node: (node.free_cores, node.free_mem, defaultdict(lambda: 0, node.replicas))
            for node in dest_nodes
        }
        zero_perf = RequestsPerTime("0 req/s")
        app_perf_surplus_backup = defaultdict(lambda: zero_perf, self._app_perf_surplus)
        app_perf_increment_backup = defaultdict(lambda: zero_perf, self._app_perf_increment)
        obsolete_containers_backup = {
            node: {cc: replicas for cc, replicas in cc_replicas.items()}
            for node, cc_replicas in self._recycling.obsolete_containers.items()
        }
        obsolete_containers_backup = defaultdict(lambda: 0, dict(obsolete_containers_backup))

        # Next, allocate obsolete container copies in destination nodes

        command = Command()

        # In case of success, these obsolete replicas will be no longer elegible. Available obsolete containers
        # and free computational resources in nodes are updated accordingly        
        copied_obsolete_replicas = []

        # In case of success, these obsolete replicas will be removed from destination nodes. Available obsolete
        # containers and free computational resources in nodes are updated accordingly
        removed_obsolete_replicas = []

        for removable_cc, available_replicas in available_src_node_obsolete_containers.items():
            # Required obsolete containers to free up enough computational resurces to allocate
            # replicas_to_allocate replicas in the node, considering rounding errors and the maximum
            # number of available replicas
            required_obsolete_replicas = max(
                ((required_cores - free_src_cores) / removable_cc.cores).magnitude,
                ((required_mem - free_src_mem) / removable_cc.mem[0]).magnitude
            )
            required_obsolete_replicas = int(ceil(required_obsolete_replicas - TransitionRBT._DELTA))
            replicas_to_remove = min(required_obsolete_replicas, available_replicas)

            # If no more replicas are required to free up enough computational resources
            if replicas_to_remove <= 0:
                break

            # Try copying required_obsolete_replicas to other nodes
            for dest_node in dest_nodes:
                if dest_node == src_node:
                    # Cannot copy obsolete containers to the same node
                    continue
                obsolete_replicas, removed_replicas = self._remove_allocate(removable_cc, replicas_to_remove,
                                                                            dest_node, command, obsolete=True)
                if obsolete_replicas > 0:
                    copied_obsolete_replicas.append((removable_cc, obsolete_replicas, dest_node))
                    removed_obsolete_replicas.extend(removed_replicas)
                    dest_nodes_modified.append(dest_node)
                    required_cores -= obsolete_replicas * removable_cc.cores
                    required_mem -= obsolete_replicas * removable_cc.mem[0]
                    replicas_to_remove -= obsolete_replicas
                    if replicas_to_remove == 0:
                        # We have copied all the required obsolete replicas to free up enough resources
                        break

        # Calculate the number of replicas that would be allocatable in the next remove-allocate-copy step
        free_cores = free_src_cores + replicas_to_allocate * cc.cores - required_cores
        free_mem = free_src_mem + replicas_to_allocate * cc.mem[0] - required_mem
        allocatable_replicas = min(
            replicas_to_allocate,
            int((free_cores.magnitude + TransitionRBT._DELTA) / cc.cores.magnitude),
            int((free_mem.magnitude + TransitionRBT._DELTA) / cc.mem[0].magnitude),
        )
        if allocatable_replicas == 0:
            # Recover from backups
            for mod_node in dest_nodes_modified:
                mod_node.free_cores, mod_node.free_mem, mod_node.replicas = dest_nodes_backup[mod_node]
            self._app_perf_surplus = app_perf_surplus_backup
            self._app_perf_increment = app_perf_increment_backup
            self._recycling.obsolete_containers = obsolete_containers_backup
        else:
            # The allocation is preserved and the state is updated for the next new containers to allocate
            command2 = self._update_next_copy_state(allocatable_replicas, cc, src_node,
                                                    copied_obsolete_replicas, removed_obsolete_replicas,
                                                    available_obsolete_containers, available_node_free_resources)            
            # Extend the comand with the removal of copies of obsolete containers in the source node
            command.extend(command2)

        return allocatable_replicas, command

    def _allocate_with_free_obsolete(self, command: Command):
        """
        Allocate unallocated new containers using free computational resources in the same node and removing    
        obsolete containers from the same node if it were necessary.
        :param command: A command with container allocations and removals.
        """
        # Allocate the new container replicas set up during the algorithm’s previous copy phase. 
        # All set up replicas must be allocatable.
        for src_node, cc, allocatable_replicas in self._allocatable_cs_next_step:
            # Allocate using free computational resources on the same node and removing obsolete
            # containers from the same node if it were necessary
            allocated_replicas, _ = self._remove_allocate(cc, allocatable_replicas, src_node, command)
            assert allocated_replicas == allocatable_replicas, "Containers must be allocatable"
        self._allocatable_cs_next_step.clear() # At this point there are not set up new container replicas

        # Try to allocate the remaining unallocated new container replicas using free computational resources
        unalloc_node_cs = self._unalloc_node_cs[:] # Make a copy since it is modified during the process
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
        Perform one remove-allocate-copy transition step by removing obsolete containers and nodes, 
        allocating unallocated new containers, and copying obsolete containers to other nodes, 
        in preparation for the next transition step.
        :param copy_nodes: A list of nodes where obsolete containers can be copied. All the nodes in the
        current allocation are used when this parameter is not set.
        :return: A command with container/node operations.
        """

        # Check that obsolete containers have not the copy label 'c'. It can be commented in production
        assert self._debug_check_label_obsolete_containers(), "Obsolete containers are not properly labelled"
        
        command = Command()

        # Remove the last obsolete containers of applications with all its new containers allocated,
        # gaining extra computational resources for the allocation of new containers, without affecting
        # the performance surplus of the remaining applications
        self.remove_last_obsolete_containers(command)

        # Allocate using free computational resources in the same node and removing obsolete
        # containers from the same node if it were necessary
        self._allocate_with_free_obsolete(command)

        # If all the new container replicas have been allocated, it is time to remove obsolete nodes
        # and return the command
        if len(self._unalloc_node_cs) == 0:
            # Check if obsolete nodes can be removed and update the command
            self.remove_obsolete_nodes(command)
            return command

        # Set up the allocation of new container replicas for the next remove-allocate-copy step.
        # Try copying obsolete containers from the node to other nodes (destination nodes),
        # yielding enough application's performance surplus to allocate the replicas of unallocated
        # containers in the next transition step

        # Get nodes with free or freeable capacity, leaving empty nodes at the end, sorted by
        # increasing price. Thus, empty nodes will be used as a last option to copy obsolete containers
        if copy_nodes is None:
            copy_nodes = self._current_alloc
        copy_nodes = self._get_free_capacity_nodes(copy_nodes)
        elegible_nodes = copy_nodes[:] 

        # Available obsolete containers for copy in each node
        available_obsolete_containers = {
            node: {cc: replicas for cc, replicas in cc_replicas.items()}
            for node, cc_replicas in self._recycling.obsolete_containers.items()
        }

        # Available free computational resources in the node in the next transtion step 
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
                # a command with the allocation of obsolete containers in elegible nodes 
                allocatable_replicas, command2 = \
                    self._copy_obsolete_containers(unalloc_node_cc, available_obsolete_containers, 
                                                   available_node_free_resources, elegible_nodes)
                if allocatable_replicas > 0:
                    # Remove the node from the elegible nodes, since it will allocate new containers in the next step
                    if src_node in elegible_nodes:
                        elegible_nodes.remove(src_node)
                    if not self._minimize_transition_commands:
                        # Complete the list of containers allocatable in the next transisition step and remove them
                        # from the list of unallocated containers
                        self._allocatable_cs_next_step.append((src_node, cc, allocatable_replicas))
                        index = self._unalloc_node_cs.index((src_node, cc, replicas_to_allocate))
                        replicas_to_allocate -= allocatable_replicas
                        if replicas_to_allocate > 0:
                            self._unalloc_node_cs[index] = (src_node, cc, replicas_to_allocate)
                        else:
                            self._unalloc_node_cs.pop(index)
                    command.extend(command2)

        # Remove common allocations and removals. One container can be allocated and removed in the same command
        # when an obsolete container is copied from node 1 to node 2, next from node 2 to node 3, and so on
        command.simplification()  

        # Check if obsolete nodes can be removed and update the command
        self.remove_obsolete_nodes(command)

        return command

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
        - For each obsolete node, remove all except its last remove operation. Note that obsolete nodes are not
        actually removed from the allocation.
        - Remove empty commands.
        - Replace Vmt nodes by Vm nodes.
        - Remove obsolete nodes from the current allocation.
        """
        # Node removal commands are generated when all the containers of an obsolete node are removed.
        # However, the nodes could be useful in future to help in the transition of recycled nodes, so
        # they could be used after a removal command. A node removal command is removed when there is
        # a later allocation in the node.
        null_command = Command() 
        node_removal_command = defaultdict(lambda: null_command) # Start with no removals for any node
        nodes_removed_once = set()
        for command in self._commands:
            for node, _, _ in command.allocate_containers:
                if not node_removal_command[node].is_null():
                    removal_command = node_removal_command[node]
                    removal_command.remove_nodes.remove(node)
                    removal_command = null_command
            for node_to_remove in command.remove_nodes:
                nodes_removed_once.add(node_to_remove)
                if node_removal_command[node_to_remove].is_null():
                    # First removal command for the node or the previous removal command has been removed
                    node_removal_command[node_to_remove] = command

        # Check the removals
        for node in nodes_removed_once:
            assert node in node_removal_command, "Node removal command not found"
            assert not node_removal_command[node].is_null(), "Node removal command is null"

        # Remove empty commands
        for command in self._commands:
            if command.is_null():
                self._commands.remove(command)

        # Remove the nodes from the current allocation
        for node, command in node_removal_command.items():
            if not command.is_null():
                # The node is removed in the last removal command
                self._current_alloc.remove(node)

        # Replace Vmt nodes by nodes Vm nodes in the commands
        for command in self._commands:
            command.vmt_to_vm()

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

    def _complete_allocation_in_temporary_nodes(self, create_nodes_command: Command,
                                               allocate_new_nodes_command: Command):
        """
        Complete the allocation using temporary nodes.
        :param create_nodes_command: Command where temporary node creations are appended.
        :param allocate_new_nodes_command: Command where container allocations in temporary nodes are appended.
        """
        # The following approaches are valid in the calculation of the temporary nodes:
        # - Execute the remove_allocate method allowing negative values for application's performance surplus. 
        # Next, they are compensated by container replicas in temporary nodes.
        # - Add a dummy node to the system with unlimited computational capacity, so all the remaining containers 
        # can be allocated executing the remove_allocate_copy method. Next, the allocations in the dummy node 
        # are compensated with container replicas in temporary nodes.
        # The second approach is preferred, since it does not modify the remove_allocate method.

        # Add a dummy node with enough capacity to allocate any number of containers. It is used for calculation 
        # purposes only
        dummy_node = Vmt(Vm(self._current_alloc[0].ic, ignore_ic_index=True))
        dummy_node.free_cores *= 10E12
        dummy_node.free_mem *= 10E12
        self._current_alloc.append(dummy_node)

        # Only the dummy node is used to copy obsolete containers
        copy_nodes = [dummy_node]

        # Perform a remove-allocate-copy operation allowing allocations in the dummy node
        command = self._remove_allocate_copy(copy_nodes)
        zero_rps = RequestsPerTime("0 rps")
        temporary_nodes_app_perf = {app: zero_rps for app in self._app_perf_surplus}
        for node, cc, replicas in command.allocate_containers:
            if node == dummy_node:
                # Allocations in the temporary nodes will be performed in a previous command, 
                # just after completing the creation of the new nodes (and temporary nodes), so
                # performance values need to be updated
                self._app_perf_increment[cc.app] -= replicas * cc.perf
                self._app_perf_surplus[cc.app] += replicas * cc.perf
                temporary_nodes_app_perf[cc.app] += cc.perf * replicas

        # Get an allocation from application's performance on temporary nodes
        temporary_nodes = [Vmt(node) for node in self.get_allocation(temporary_nodes_app_perf)]
        for tmp_node_index in range(len(temporary_nodes)):
            # Change the id of temporary nodes to negative values to be easily identified
            temporary_nodes[tmp_node_index].id = -(tmp_node_index + 1)
            temporary_nodes[tmp_node_index].vm.id = -(tmp_node_index + 1)

        # Add temporary nodes to the creation command
        create_nodes_command.create_nodes.extend(temporary_nodes)

        # Move allocations in the command from the dummy node to the temporary nodes
        containers_in_temporary_nodes = [
            (node, cc, replicas)
            for node in temporary_nodes for cc, replicas in node.replicas.items()
        ]
        allocate_new_nodes_command.allocate_containers.extend(containers_in_temporary_nodes)

        # The command must not allocate containers in the dummy node
        command.allocate_containers = [
            (node, cc, replicas)
            for node, cc, replicas in command.allocate_containers[:]
            if node != dummy_node
        ]

        # Obsolete containers and nodes in the recycling object must be moved from the dummy node 
        # to temporary nodes
        if dummy_node in self._recycling.obsolete_containers:
            del self._recycling.obsolete_containers[dummy_node]
        for temporary_node in temporary_nodes:
            self._recycling.obsolete_containers[temporary_node] = dict(temporary_node.replicas)
            self._recycling.obsolete_nodes.append(temporary_node)

        # Replace the dummy node with temporary nodes in the current allocation
        self._current_alloc.remove(dummy_node)
        self._current_alloc.extend(temporary_nodes)

        # All the unallocated replicas can be allocated after creating the temporary nodes and allocating
        # the required containers in them
        self._allocate_with_free_obsolete(command)

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
        self._recycling_vm = Recycling(initial_alloc, final_alloc, self._hot_node_scale_up, self._hot_replicas_scale)
        self._recycling = RecyclingVmt(self._recycling_vm, vm_to_vmt)

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
        application's minimum performance requirement. It is based on the remove-allocate-copy algorithm, executed
        by the allocation loop.

        :param initial_alloc: Initial allocation.
        :param final_alloc: Final allocation.
        :return: A list of commands and the time to perform the transition.
        """
        # Initialize the transition and check if a transition is necessary
        if not self._transition_init(initial_alloc, final_alloc):
            return [], 0

        # Node creation and node upgrade operations are the most time-consuming. Therefore, they are the first
        # operations to be performed. The creation command part may be empty and extended later to include the 
        # creation of temporary nodes
        upgrade_node_info = [(n1, n2.ic) for n1, n2 in self._recycling.upgraded_node_pairs.items()]
        create_upgrade_nodes_command = Command(create_nodes=self._recycling.new_nodes, upgrade_nodes=upgrade_node_info)
        self._append_command(create_upgrade_nodes_command, append_null_command=True)

        # Allocation loop until node upgrading completes. It should be noted that node upgrading time
        # is less than or equal to node creation required time
        elapsed_time = 0
        if not self._minimize_transition_commands:
            elapsed_time = self.allocation_loop(self._timing_args.hot_node_scale_up_time)

        # Update the elapsed time and current allocation after completing the node upgrading
        elapsed_time = max(elapsed_time, self._timing_args.hot_node_scale_up_time)
        for initial_node, final_node in self._recycling.upgraded_node_pairs.items():
            initial_node.upgrade(final_node)

        # Allocation loop until the time limit, appending commands to the command list while waiting 
        # for node scaling up or creation. This loop increases the number of commands, but can help 
        # to reduce the number of temporary nodes required, and so the transition cost
        if not self._minimize_transition_commands:
            max_time = self._timing_args.node_creation_time - self._timing_args.hot_node_scale_up_time
            elapsed_time += self.allocation_loop(max_time)

        # Update the elapsed time and current allocation after completing the node creation
        elapsed_time = max(elapsed_time, self._timing_args.node_creation_time)
        self._current_alloc.extend(self._recycling.new_nodes)

        # Allocate new containers in new nodes. The corresponding command may be empty if there
        # are no new nodes in the final allocation. This command can be extended later to include
        # containers in temporary nodes.
        allocate_in_new_nodes_command = Command(allocate_containers=self._unallocated_containers_in_new_nodes[:])
        allocate_in_new_nodes_command.sync_on_nodes_creation = True # It can not start before node creation completes
        for _, cc, replicas in self._unallocated_containers_in_new_nodes:
            self._app_perf_increment[cc.app] += replicas * cc.perf
            self._app_unalloc_perf[cc.app] -= replicas * cc.perf
            assert self._app_unalloc_perf[cc.app].magnitude >= - Transition._DELTA, "Invalid performance"
        self._unallocated_containers_in_new_nodes.clear()
        self._append_command(allocate_in_new_nodes_command, append_null_command=True)

        # Allocation loop until the time limit when the time limit has not been reached yet
        if not self._minimize_transition_commands:
            # Allocation loop until the time limit, appending commands to the command list while waiting 
            # for node scaling up or creation. This loop can help to reduce the number of temporary nodes required,
            # and so the transition cost
            self.allocation_loop(self._time_limit - elapsed_time)

        # If there are still unallocated containers in recycled nodes
        if len(self._allocatable_cs_next_step) > 0 or len(self._unalloc_node_cs) > 0:
            # If the transition can be completed with a single new command and the transition commands need not
            # to be minimized, extend the transition time a little beyond the time limit, instead of creating 
            # temporary nodes
            if len(self._unalloc_node_cs) == 0 and not self._minimize_transition_commands:
                self._append_command(self._remove_allocate_copy())
            # The transition may last too much, so it is completed creating temporary nodes
            else:
                self._complete_allocation_in_temporary_nodes(create_upgrade_nodes_command, allocate_in_new_nodes_command)
        
        # Final cleanup command: remove last obsolete containers and obsolete nodes
        command = Command()
        self.remove_last_obsolete_containers(command)
        self.remove_obsolete_nodes(command)
        self._append_command(command)
        
        # Post-processing operations to obtain the final command list
        self._post_process_commands()

        # Check whether the commands implement a valid transition between the initial and the final allocations
        assert self.check_transition(initial_alloc, final_alloc, self._commands), "Invalid transition"

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

    def _debug_check_label_obsolete_containers(self) -> bool:            
        """ 
        Check if at least one obsolete replica in self._recycling.obsolete_containers has label 'c'
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
    
    def get_worst_case_transition_time(self) -> int:
        """
        Get the worst-case transition time.
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

    def get_recycling_levels(self) -> tuple[float, float]:
        """
        Get node and container recycling levels for the last transition.
        :return: A tuple with node and container recycling levels.
        """
        return self._recycling.node_recycling_level, self._recycling.container_recycling_level

    def get_recycled_node_pairs(self):
        """
        Get the recycled node pairs.
        :return: Recycled node pairs.
        """
        return self._recycling_vm.recycled_node_pairs


