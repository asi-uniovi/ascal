"""
It defines the TimedOps class to create/remove container and nodes using an event-driven architecture.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable
from fcma import ContainerGroup, ContainerClass, Vm, RequestsPerTime
from nodestates import NodeStates

class TimedOps:
    """
    This class manages the creation and removal of nodes and containers at predefined times,
    taking into account the time required for these operations.
    """

    # Constant used to deal with numerical approximations
    _DELTA = 0.000001

    class EventTypes(Enum):
        # Number express event priorities
        LOG_MESSAGE = 0
        CREATE_NODE_BEGIN = 1
        CREATE_NODE_BILLED = 2
        CREATE_NODE_END = 3
        START_REPLICAS_GRACE_PERIOD = 4
        REMOVE_CONTAINER_REPLICAS_BEGIN = 5
        REMOVE_CONTAINER_REPLICAS_END = 6
        REMOVE_NODE_BEGIN = 7
        REMOVE_NODE_END = 8
        ALLOCATE_CONTAINER_REPLICAS_BEGIN = 9
        ALLOCATE_CONTAINER_REPLICAS_END = 10

    @dataclass
    class Event:
        """
        Event for creating/removing nodes or containers.
        """
        type: 'TimedOps.EventTypes'
        containers: tuple[ContainerClass, int, Vm, ContainerClass] | None = None # Information of containers
        node: Vm | None = None # Node to create/remove
        message: str = None # Message to log
        callback: Callable[..., None] | None = None # Function called when the event is fired


    @dataclass(frozen=True)
    class TimingArgs:
        """
        Times required to create/remove nodes and containers.
        """
        node_time_to_billing: int = 0 # Time from the beginning of node creation until the node is billed
        node_creation_time: int = 0 # Time from the beginning of node creation until it is ready to allocate containers
        node_removal_time: int = 0 # Time required to remove a node
        container_creation_time: int = 0 # Time required to create a container
        container_removal_time: int = 0 # Time required to remove a container

    def __init__(self, time_args: TimingArgs, priorize_events: bool=False):
        """
        Create an event-driven timing system for creation/removal of nodes and containers.
        :param time_args: Times required to create/remove nodes and containers.
        :param priorize_events: Set to priorize the processing of events released at the same time.
        """
        self.time_args = time_args # Times required to create/remove containers and nodes
        self._event_list: list[tuple[int, TimedOps.Event]] = [] # List of events to handle in the form (time, event)
        self._sorting_required = False # Events must be ordered by time before being processed
        self.node_billing_changed = False # True if node changes affecting node billing occurred at the current time
        self.new_nodes_ready = False # True if there are new nodes ready at the current time
        self.perf_changed = False # True if containers are removed or allocated at the current time
        self._last_dispatched_time = -1 # Current time. It is the time of the last dispatched event
        self.log: Callable[[...], None] = lambda _: None # Method used to print a log message
        self._priorize_events = priorize_events # Set to priorize events released at the same time

    def is_event_list_empty(self) -> bool:
        """
        Check if the event list is empty.
        :return: True when the event list is empty.
        """
        return len(self._event_list) == 0

    def _add_event(self, at_time: int, event: Event):
        """
        Add an event to the event list at the given time.
        :param at_time: Time of addition.
        :param event: Event.
        """
        assert at_time >= self._last_dispatched_time, "Can not dispatch events in the past"

        # Check if the event is previous to the last event in the list. Events must be dispatched in time order
        if len(self._event_list) > 0 and at_time < self._event_list[-1][0]:
            self._sorting_required = True
        self._event_list.append((at_time, event))
        # If the event occurs at the last dispatched time, it is dispatched to avoid unnecesary delays
        if at_time == self._last_dispatched_time:
            self._dispatch_at_last_time()

    def _update_changes(self, event_type: EventTypes):
        """
        Update billing, performance and new node change properties after processing an event.
        :param event_type: Type of event just processed.
        """
        self.node_billing_changed = (event_type == TimedOps.EventTypes.CREATE_NODE_BILLED) or \
                                    (event_type == TimedOps.EventTypes.REMOVE_NODE_END) or \
                                    self.node_billing_changed
        self.perf_changed = (event_type == TimedOps.EventTypes.ALLOCATE_CONTAINER_REPLICAS_END) or \
                            (event_type == TimedOps.EventTypes.REMOVE_CONTAINER_REPLICAS_BEGIN) or \
                            self.perf_changed
        self.new_nodes_ready = (event_type == TimedOps.EventTypes.CREATE_NODE_END) or \
                               self.new_nodes_ready

    def _dispatch_at_last_time(self):
        """
        Dispatch events in the list with the same time as the last dispatched time.
        Dispatching an event may modify nodes and container allocations.
        """
        if len(self._event_list) == 0:
            return
        if self._sorting_required:
            # Sort events by increasing fire time
            self._event_list.sort(key=lambda event: event[0])
        while len(self._event_list) > 0 and self._event_list[0][0] == self._last_dispatched_time:
            # Get the first even in the event list
            # (and those with the same release time if priorities are set)
            release_time, next_event = self._event_list.pop(0)
            next_events = [next_event]
            if self._priorize_events:
                while len(self._event_list) > 0 and self._event_list[0][0] == release_time:
                    next_events.append(self._event_list.pop(0)[1])
                # Sort events using priorities
                next_events.sort(key=lambda ev: ev.type.value)
            # Execute the callback functions and update node billing, performance and new nodes changes
            for next_event in next_events:
                next_event.callback(next_event)
                self._update_changes(next_event.type)

    def dispatch_events(self, until_time: int) -> bool:
        """
        Dispatch all events up to a specified maximum time. Dispatching an event may modify nodes and
        container allocations.
        :param until_time: Last time to dispatch events.
        :return: True if some event has been dispatched.
        """

        assert until_time >= self._last_dispatched_time, "Can not dispatch events in the past"

        # If the time is equal to the last dispatched time, then dispatch updating billing,
        # performance and new node changes
        if until_time == self._last_dispatched_time:
            self._dispatch_at_last_time()
            return

        # Billing, perfromance and new node changes are recalculated when until_time > self._last_disptached_time,
        # so they start as False
        self.node_billing_changed = False
        self.perf_changed = False
        self.new_nodes_ready = False

        # Update the last dispatched time
        self._last_dispatched_time = until_time

        # If the list of events is empty, there is nothing to be done
        if len(self._event_list) == 0:
            self._sorting_required = False
            return 0

        if self._sorting_required:
            # Sort events by increasing fire time
            self._event_list.sort(key=lambda event: event[0])

        # Dispatch the events
        dispatched_some_event = False
        while len(self._event_list) > 0:
            # Check if we have processed all the events with release time less than or equal to untiL_time
            if self._event_list[0][0] > until_time:
                return dispatched_some_event
            # Get the first event in the event list
            # (and those with the same release time if priorities are set)
            dispatched_some_event = True
            release_time, next_event = self._event_list.pop(0)
            next_events = [next_event]
            if self._priorize_events:
                while len(self._event_list) > 0 and self._event_list[0][0] == release_time:
                    next_events.append(self._event_list.pop(0)[1])
                # Sort events using priorities
                next_events.sort(key=lambda ev: ev.type.value)
            # Execute the callback functions and update node billing, performance and new nodes changes
            for next_event in next_events:
                if next_event.callback is not None:
                    next_event.callback(next_event)
                    self._update_changes(next_event.type)

    def timed_log(self, at_time: int, message: str):
        """
        Add and event to log a message at a given time.
        :param at_time: Log time.
        :param message: Message to log.
        """
        event = TimedOps.Event(TimedOps.EventTypes.LOG_MESSAGE, message=message, callback=self.at_log)
        self._add_event(at_time, event)

    def at_log(self, event: Event):
        """
        Log a message when the event is fired.
        :param event: Event that has just being fired.
        """
        self.log(event.message)

    def allocate_container_replicas(self, at_time: int, cc: ContainerClass, replicas: int, node: Vm) -> int:
        """
        Allocate new replicas of a container class in a node. The new replicas have zero performance until
        the container creation time elapses. Thus, they do not increase the performance while the container is
        being created.
        :param at_time: Container creation starts at this time.
        :param cc: Container class.
        :param replicas: The exact number of replicas to allocate if allocation occurs at a future time (unless
        the allocation is aborted if the node enters meanwhile in its removing state). The maximum number
        of replicas to allocate when the allocation starts at the current time (self._last_dispatched_time).
        :param node: Node where replicas will be allocated.
        :return: The number of replicas actually allocated when the allocation occurs at the current time.
        None if allocation occurs at a future time and replicas > 0.
        """
        assert at_time >= self._last_dispatched_time, "Can not allocate containers in the past"
        if replicas == 0:
            return 0

        allocatable_replicas = replicas
        # If allocation occurs just now
        if at_time == self._last_dispatched_time:
            cpu_allocatable_replicas = int(node.free_cores.magnitude / cc.cores.magnitude + TimedOps._DELTA)
            mem_allocatable_replicas = int(node.free_mem.magnitude / cc.mem[0].magnitude + TimedOps._DELTA)
            allocatable_replicas = min(cpu_allocatable_replicas, mem_allocatable_replicas, replicas)
            if allocatable_replicas == 0:
                return 0

        event = TimedOps.Event(TimedOps.EventTypes.ALLOCATE_CONTAINER_REPLICAS_BEGIN,
                               containers=(allocatable_replicas, node, cc),
                               callback=self._at_allocate_container_replicas_begin)
        self._add_event(at_time, event)
        if at_time == self._last_dispatched_time:
            return allocatable_replicas
        else:
            return None

    def _at_allocate_container_replicas_begin(self, event: Event):
        """
        Start the allocation of container replicas when the event is fired.
        :param event: Event that has just being fired.
        """
        replicas, node, cc = event.containers

        assert NodeStates.get_state(node) == NodeStates.READY, "Can not allocate on nodes that are not ready"

        # Firstly, allocate containers with zero performance
        zero_perf_cc = ContainerClass(cc.app, cc.ic, cc.fm, cc.cores, cc.mem,
                                      RequestsPerTime("0 req/s"), cc.aggs, cc.agg_level)
        cpu_allocatable_replicas = int(node.free_cores.magnitude / cc.cores.magnitude + TimedOps._DELTA)
        mem_allocatable_replicas = int(node.free_mem.magnitude / cc.mem[0].magnitude + TimedOps._DELTA)
        allocatable_replicas = min(cpu_allocatable_replicas, mem_allocatable_replicas, replicas)
        assert allocatable_replicas == replicas, "Can not allocate the required replicas"
        node.cgs.append(ContainerGroup(zero_perf_cc, allocatable_replicas))
        node.free_cores -= allocatable_replicas * zero_perf_cc.cores
        node.free_mem -= allocatable_replicas * zero_perf_cc.mem[0]

        # Create the related event to complete the containers allocation
        event = TimedOps.Event(TimedOps.EventTypes.ALLOCATE_CONTAINER_REPLICAS_END,
                               containers=(replicas, node, cc, zero_perf_cc),
                               callback=self._at_allocate_container_replicas_end)
        self._add_event(self._last_dispatched_time + self.time_args.container_creation_time, event)
        self.log(f'Allocating {replicas} replicas {cc.app} on node {node}')

    def _at_allocate_container_replicas_end(self, event: Event):
        """
        Complete the allocation of replicas when the event is fired.
        :param event: Event that has just being fired.
        """
        replicas, node, cc, zero_perf_cc = event.containers

        # The container replicas with zero performance will become active.
        # Firstly, remove the zero performance container replicas.
        # Secondly, create the real replicas.

        # Find the zero-performance container group with the same number of replicas
        found_cg = False
        cgs = [cg for cg in node.cgs]
        for cg in cgs:
            if cg.cc == zero_perf_cc and cg.replicas == replicas:
                node.cgs.remove(cg)
                found_cg = True
                break
        assert found_cg is True, "Error allocating container replicas"
        if NodeStates.get_state(node) == NodeStates.REMOVING:
            # The allocation is aborted before being completed
            self.log(f'Aborting the allocation of {replicas} replicas of {cc.app} on node {node}')
            return
        # Find a container group with ready replicas for the same container class and increment the number of replicas
        found_cg = False
        for cg in node.cgs:
            if cg.cc == cc:
                cg.replicas += replicas
                found_cg = True
                break
        # If it is not found, create a new container group with the replicas
        if not found_cg:
            node.cgs.append(ContainerGroup(cc, replicas))
        self.log(f'Completed the allocation of {replicas} replicas {cc} on node {node}')

    def remove_container_replicas(self, at_time: int, cc: ContainerClass, replicas: int, node: Vm) -> int:
        """
        Remove container replicas in a node. The replicas will become zero performance replicas with
        None application until they are completly removed, once the removal time elapses.
        :param at_time: Container removal starts at this time.
        :param cc: Container class.
        :param replicas: The number of replicas to remove.
        :param node: Node where replicas will be removed.
        :return: The number of replicas that are actually removed if the removal occurs at the current time,
        or None if the removal occurs at a future time and replicas > 0.
        """
        assert at_time >= self._last_dispatched_time, "Can not remove containers in the past"
        if replicas == 0:
            return 0
        # Calculate the number of replicas to remove
        if at_time == self._last_dispatched_time:
            replicas_to_remove = 0
            for cg in node.cgs:
                # Replicas that are in the process of being created or being removed can not be removed.
                # Thus, we use cc as a container to compare with
                if cg.cc == cc:
                    replicas_to_remove = min(cg.replicas, replicas)
                    break
            if replicas_to_remove == 0:
                return 0
        else:
            replicas_to_remove = replicas

        event = TimedOps.Event(TimedOps.EventTypes.REMOVE_CONTAINER_REPLICAS_BEGIN,
                               containers=(replicas_to_remove, node, cc),
                               callback=self._at_remove_container_replicas_begin)
        self._add_event(at_time, event)
        if at_time == self._last_dispatched_time:
            return replicas_to_remove
        else:
            return None

    def _at_remove_container_replicas_begin(self, event: Event):
        """
        Start the removal of container replicas when the event is fired.
        :param event: Event that has just being fired.
        """
        replicas, node, cc = event.containers # The exact number of replicas to remove

        # Check the number of replicas that can be removed and get the related container group
        removable_replicas = 0
        cg_with_replicas = None
        for cg in node.cgs:
            # Replicas that are in the process of being created or being removed can not be removed,
            # so we can use the container class to compare with
            if cg.cc == cc:
                removable_replicas = min(cg.replicas, replicas)
                cg_with_replicas = cg
                break
        if removable_replicas == 0:
            return

        self.log(f'Removing {removable_replicas} replicas {cc.app} from node {node}')

        # Move the replicas to remove to a new container group with zero performance
        # replicas and None application (None application means replicas being removed)
        zero_perf_cc = ContainerClass(None, cc.ic, cc.fm, cc.cores, cc.mem,
                                      RequestsPerTime("0 req/s"), cc.aggs, cc.agg_level)
        cg_with_replicas.replicas -= removable_replicas
        if cg_with_replicas.replicas == 0:
            node.cgs.remove(cg_with_replicas)
        node.cgs.append(ContainerGroup(zero_perf_cc, removable_replicas))

        # Create the related event to complete the containers removal
        event = TimedOps.Event(TimedOps.EventTypes.REMOVE_CONTAINER_REPLICAS_END,
                               containers=(removable_replicas, node, cc, zero_perf_cc),
                               callback=self._at_remove_container_replicas_end)
        self._add_event(self._last_dispatched_time + self.time_args.container_removal_time, event)

    def _at_remove_container_replicas_end(self, event: Event):
        """
        Complete the removal of replicas when the event is fired.
        :param event: Event that has just being fired.
        """
        replicas, node, cc, zero_perf_cc = event.containers

        # Update free computational resources in the node
        node.free_cores += replicas * cc.cores
        node.free_mem += replicas * cc.mem[0]
        node.cgs.remove(ContainerGroup(zero_perf_cc, replicas))
        self.log(f'Completed the removal of {replicas} replicas {cc} from node {node}')

    def create_node(self, at_time: int, node: Vm):
        """
        A preconfigured node is created going through three states:
        1) State TIMING.NODE_BOOTING. It is the initial state. In this state the node is not billed.
        2) State TIMING.NODE_BILLED. The node is billed, but it can not allocate containers yet.
        2) State TIMING.NODE_READY. The node is billed and can allocate containers.
        :param at_time: Node creation starts at this time.
        :param node: Preconfigured node.
        """
        assert at_time >= self._last_dispatched_time, "Can not crete nodes in the past"
        assert node is not None, "Nodes need to be preconfigured to be created"
        event = TimedOps.Event(TimedOps.EventTypes.CREATE_NODE_BEGIN, node=node,
                               callback=self._at_create_node_begin)
        self._add_event(at_time, event)

    def _at_create_node_begin(self, event: Event):
        """
        Start the creation of a node when the event is fired.
        :param event: Event that has just being fired.
        """
        node = event.node
        NodeStates.set_state(node, NodeStates.BOOTING)
        event = TimedOps.Event(TimedOps.EventTypes.CREATE_NODE_BILLED, node=node,
                               callback=self._at_start_node_billing)
        self._add_event(self._last_dispatched_time + self.time_args.node_time_to_billing, event)
        self.log(f'Creating node {node}')

    def _at_start_node_billing(self, event: Event):
        """
        Start billing the node, although it can not allocate containers.
        :param event: Event that has just being fired.
        """
        node = event.node
        assert NodeStates.get_state(node) == NodeStates.BOOTING,\
        "The node requires a booting time before being billed"
        NodeStates.set_state(node, NodeStates.BILLED)
        event = TimedOps.Event(TimedOps.EventTypes.CREATE_NODE_END, node=node,
                               callback=self._at_create_node_end)
        new_time = self._last_dispatched_time + \
                   (self.time_args.node_creation_time - self.time_args.node_time_to_billing)
        self._add_event(new_time, event)
        self.log(f'Billing of node {node} starts')

    def _at_create_node_end(self, event: Event):
        """
        Make the node elegible to allocate containers.
        :param event: Event that has just being fired.
        """
        node = event.node
        assert NodeStates.get_state(node) == NodeStates.BILLED,\
        "The node requires being billed before being ready"
        NodeStates.set_state(node, NodeStates.READY)
        self.log(f'Node {node} is ready to execute containers')

    def remove_node(self, at_time: int, node: Vm):
        """
        Create an event to remove a node. Once removed the node is no longer billed.
        :param at_time: Node removal starts at this time.
        :param node: Node to be removed.
        """
        assert at_time >= self._last_dispatched_time, "Can not remove nodes in the past"
        assert node is not None, "Can not remove an invalid node"
        event = TimedOps.Event(TimedOps.EventTypes.REMOVE_NODE_BEGIN, node=node,
                               callback=self._at_remove_node_begin)
        self._add_event(at_time, event)

    def _at_remove_node_begin(self, event: Event):
        """
        Start removing a node, so it can not allocate containers.
        :param event: Event that has just being fired.
        """
        node = event.node
        if len(node.cgs) > 0:
            # First complete the removal of containers at the current time
            self._dispatch_at_last_time()
        assert len(node.cgs) == 0, "Can not remove a node with containers"
        NodeStates.set_state(node, NodeStates.REMOVING)
        event = TimedOps.Event(TimedOps.EventTypes.REMOVE_NODE_END, node=node,
                               callback=self._at_remove_node_end)
        self._add_event(self._last_dispatched_time + self.time_args.node_removal_time, event)
        self.log(f"Removing node {node}")

    def _at_remove_node_end(self, event: Event):
        """
        Complete the removal of the node when the event is fired.
        :param event: Event that has just being fired.
        """
        node = event.node
        assert NodeStates.get_state(node) == NodeStates.REMOVING, "The node can not be removed"
        NodeStates.set_state(node, NodeStates.REMOVED)
        self.log(f'Node {node} is removed')
