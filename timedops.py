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

    class EventTypes(Enum):
        LOG_MESSAGE = 0
        ALLOCATE_CONTAINER_REPLICAS_BEGIN = 1
        ALLOCATE_CONTAINER_REPLICAS_END = 2
        START_REPLICAS_GRACE_PERIOD = 3
        REMOVE_CONTAINER_REPLICAS_BEGIN = 4
        REMOVE_CONTAINER_REPLICAS_END = 5
        CREATE_NODE_BEGIN = 6
        CREATE_NODE_BILLED = 7
        CREATE_NODE_END = 8
        REMOVE_NODE_BEGIN = 9
        REMOVE_NODE_END = 10

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

    def __init__(self, time_args: TimingArgs):
        """
        Create an event-driven timing system for creation/removal of nodes and containers.
        :param time_args: Times required to create/remove nodes and containers.
        """
        self.time_args = time_args # Times required to create/remove containers and nodes
        self._event_list: list[tuple[int, TimedOps.Event]] = [] # List of events to handle in the form (time, event)
        self._sorting_required = False # Events must be ordered by time before being processed
        self.node_billing_changed = False # True if node changes affecting node billing occurred at the current time
        self.new_nodes_ready = False # True if there are new nodes ready at the current time
        self.perf_changed = False # True if containers are removed or allocated at the current time
        self._last_dispatched_time = -1 # Time of the last dispatched event
        self.log: Callable[[...], None] = lambda _: None # Method used to print a log message

    def add_event(self, at_time: int, event: Event):
        """
        Add an event to the event list at the given time.
        :param at_time: Time of addition.
        :param event: Event.
        """
        assert at_time >=self._last_dispatched_time, "Can not dispatch events in the past"

        # Check if the event is previous to the las event in the list. Events must be dispatched in order.
        if len(self._event_list) > 0 and at_time < self._event_list[-1][0]:
            self._sorting_required = True
        self._event_list.append((at_time, event))
        # If the event occurs at the last disptached time, it is dispatched to avoid unnecesary delays
        if at_time == self._last_dispatched_time:
            self.dispatch_at_last_time()

    def _update_changes(self, event_type: EventTypes):
        """
        Update properties after processing an event.
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

    def dispatch_at_last_time(self):
        """
        Dispatch any event in the list with the same time as the last dispatched time.
        """
        if len(self._event_list) == 0:
            return
        if self._sorting_required:
            # Sort events by increasing fire time
            self._event_list.sort(key=lambda event: event[0])
        while len(self._event_list) > 0 and self._event_list[0][0] == self._last_dispatched_time:
            # Get the first even in the event list and execute its callback function
            _, next_event = self._event_list.pop(0)
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

        # If the time is equal to the last dispatched time, then dispatch updating the property changes
        if until_time == self._last_dispatched_time:
            self.dispatch_at_last_time()
            return

        # Property changes are recalculated when until_time > self._last_disptached_time, so they start as False
        self.node_billing_changed = False
        self.perf_changed = False
        self.new_nodes_ready = False

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
            # Check if the earliest event is later than the maximum time
            if self._event_list[0][0] > until_time:
                return dispatched_some_event
            # Get the first event in the event list and execute its callback function
            _, next_event = self._event_list.pop(0)
            dispatched_some_event = True
            if next_event.callback is not None:
                next_event.callback(next_event)
                self._update_changes(next_event.type)

    def timed_log(self, at_time: int, message: str):
        """
        Log a message at the given time.
        :param at_time: Log time.
        :param message: Message to log.
        """
        event = TimedOps.Event(TimedOps.EventTypes.LOG_MESSAGE, message=message, callback=self.at_log)
        self.add_event(at_time, event)

    def at_log(self, event: Event):
        """
        Log a message when the event is fired.
        :param event: Event that has just being fired.
        """
        self.log(event.message)

    def allocate_container_replicas(self, at_time: int, cc: ContainerClass, replicas: int, node: Vm) -> int:
        """
        Allocate new replicas of a container class on a node. The new replicas have zero performance until
        the container creation time elapses. Thus, they do not increase the performance while the container is
        created.
        :param at_time: Container creation starts at this time.
        :param cc: Container class.
        :param replicas: The exact number of replicas to allocate if allocation occurs at a future time, unless
        the node enters meanwhile in its removing state. The maximum number of replicas to allocate when the
        allocation starts at the current time.
        :param node: Node where replicas will be allocated.
        :return: The number of replicas that are allocated at current time, or None if allocation
        will occur at a future time.
        """
        assert at_time >= self._last_dispatched_time, "Can not allocate containers in the past"
        if replicas == 0:
            return 0

        allocatable_replicas = replicas

        # If allocation occurs just now
        if at_time == self._last_dispatched_time:
            cpu_allocatable_replicas = int(node.free_cores.magnitude / cc.cores.magnitude)
            mem_allocatable_replicas = int(node.free_mem.magnitude / cc.mem[0].magnitude)
            allocatable_replicas = min(cpu_allocatable_replicas, mem_allocatable_replicas, replicas)
            if allocatable_replicas == 0:
                return 0

        event = TimedOps.Event(TimedOps.EventTypes.ALLOCATE_CONTAINER_REPLICAS_BEGIN,
                               containers=(allocatable_replicas, node, cc),
                               callback=self._at_allocate_container_replicas_begin)
        self.add_event(at_time, event)
        if at_time == self._last_dispatched_time:
            return allocatable_replicas
        else:
            return None

    def _at_allocate_container_replicas_begin(self, event: Event):
        """
        Start the allocation of container replicas when the event is fired.
        :param event: Event that has just being fired.
        """
        replicas = event.containers[0] # The exact number of replicas to allocate, unless allocation is aborted
        node = event.containers[1]
        cc = event.containers[2]
        assert NodeStates.get_state(node) == NodeStates.READY, "Can not allocate on nodes that are not ready"

        # Firstly, allocate containers with zero performance
        zero_perf_cc = ContainerClass(cc.app, cc.ic, cc.fm, cc.cores, cc.mem,
                                      RequestsPerTime("0 req/s"), cc.aggs, cc.agg_level)
        cpu_allocatable_replicas = int(node.free_cores.magnitude / cc.cores.magnitude)
        mem_allocatable_replicas = int(node.free_mem.magnitude / cc.mem[0].magnitude)
        allocatable_replicas = min(cpu_allocatable_replicas, mem_allocatable_replicas, replicas)
        assert allocatable_replicas == replicas, "Can not allocate the required replicas"
        node.cgs.append(ContainerGroup(zero_perf_cc, allocatable_replicas))
        node.free_cores -= allocatable_replicas * zero_perf_cc.cores
        node.free_mem -= allocatable_replicas * zero_perf_cc.mem[0]

        # Create the related event to complete the containers allocation
        event = TimedOps.Event(TimedOps.EventTypes.ALLOCATE_CONTAINER_REPLICAS_END,
                               containers=(replicas, node, cc, zero_perf_cc),
                               callback=self._at_allocate_container_replicas_end)
        self.add_event(self._last_dispatched_time + self.time_args.container_creation_time, event)
        self.log(f'Allocating {replicas} replicas of {cc.app} on node {node}')

    def _at_allocate_container_replicas_end(self, event: Event):
        """
        Complete the allocation of replicas when the event is fired.
        :param event: Event that has just being fired.
        """
        replicas = event.containers[0]
        node = event.containers[1]
        cc = event.containers[2]
        zero_perf_cc = event.containers[3]

        # The container replicas with zero performance will become active.
        # Firstly, remove the zero performance container replicas.
        # Secondly, create the replicas.
        found_cg = False  # Found container group with zero performance replicas
        cgs = [cg for cg in node.cgs]
        for cg in cgs:
            # Several container groups of the same application with zero performance replicas are possible
            # when application containers are removed at different times
            if cg.cc == zero_perf_cc and cg.replicas == replicas:
                node.cgs.remove(cg)
                found_cg = True
                break
        assert found_cg is True, "Error allocating container replicas"
        if NodeStates.get_state(node) == NodeStates.REMOVING:
            self.log(f'Aborting the allocation of {replicas} replicas of {cc.app} on node {node}')
            return
        found_cg = False # Found container group with ready replicas
        for cg in node.cgs:
            if cg.cc == cc:
                cg.replicas += replicas
                found_cg = True
                break
        if not found_cg:
            node.cgs.append(ContainerGroup(cc, replicas))
        self.log(f'Completed the allocation of {replicas} replicas of {cc.app} on node {node}')

    def remove_container_replicas(self, at_time: int, cc: ContainerClass, replicas: int, node: Vm) -> int:
        """
        Remove container replicas in a node. The replicas will become zero performance replicas with
        None application until they are completly removed, once the removal time elapses.
        :param at_time: Container removal starts at this time.
        :param cc: Container class.
        :param replicas: The exact number of replicas to remove if removal occurs at a future time, or
        otherwise, the maximum number of replicas to remove.
        :param node: Node where replicas will be removed.
        :return: The number of replicas that are removed at the current time, or None if removal
        is at a future time.
        """
        assert at_time >= self._last_dispatched_time, "Can not remove containers in the past"
        if replicas == 0:
            return 0
        # Calculate the number of replicas to remove
        if at_time == self._last_dispatched_time:
            replicas_to_remove = 0
            for cg in node.cgs:
                # Replicas that are in the process of being created or being removed can not be removed.
                # Thus, we use cc as container to compare
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
        self.add_event(at_time, event)
        if at_time == self._last_dispatched_time:
            return replicas_to_remove
        else:
            return None

    def _at_remove_container_replicas_begin(self, event: Event):
        """
        Start the removal of container replicas when the event is fired.
        :param event: Event that has just being fired.
        """
        replicas = event.containers[0] # The exact number of replicas to remove
        node = event.containers[1]
        cc = event.containers[2]

        # Check the number of replicas that can be removed and get the related container group
        removable_replicas = 0
        cg_with_replicas = None
        for cg in node.cgs:
            # Replicas that are in the process of being created or being removed can not be removed
            if cg.cc == cc:
                removable_replicas = min(cg.replicas, replicas)
                cg_with_replicas = cg
                break
        if removable_replicas == 0:
            return

        self.log(f'Removing {removable_replicas} replicas of {cc.app} from node {node}')

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
        self.add_event(self._last_dispatched_time + self.time_args.container_removal_time, event)

    def _at_remove_container_replicas_end(self, event: Event):
        """
        Complete the removal of replicas when the event is fired.
        :param event: Event that has just being fired.
        """
        replicas = event.containers[0]
        node = event.containers[1]
        cc = event.containers[2]
        zero_perf_cc = event.containers[3]

        # Update free computational resources in the node
        node.free_cores += replicas * cc.cores
        node.free_mem += replicas * cc.mem[0]
        node.cgs.remove(ContainerGroup(zero_perf_cc, replicas))
        self.log(f'Completed the removal of {replicas} replicas of {cc.app} from node {node}')

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
        assert node is not None, "Can not create a node with None configuration"
        event = TimedOps.Event(TimedOps.EventTypes.CREATE_NODE_BEGIN, node=node,
                               callback=self._at_create_node_begin)
        self.add_event(at_time, event)

    def _at_create_node_begin(self, event: Event):
        """
        Start the creation of a node when the event is fired.
        :param event: Event that has just being fired.
        """
        node = event.node
        NodeStates.set_state(node, NodeStates.BOOTING)
        event = TimedOps.Event(TimedOps.EventTypes.CREATE_NODE_BILLED, node=node,
                               callback=self._at_start_node_billing)
        self.add_event(self._last_dispatched_time + self.time_args.node_time_to_billing, event)
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
        self.add_event(new_time, event)
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
        assert at_time >= self._last_dispatched_time, "Can not crete nodes in the past"
        assert node is not None, "Can not remove an invalid node"
        assert NodeStates.get_state(node) == NodeStates.REMOVING, "A node is not labelled as removable"
        event = TimedOps.Event(TimedOps.EventTypes.REMOVE_NODE_BEGIN, node=node,
                               callback=self._at_remove_node_begin)
        self.add_event(at_time, event)

    def _at_remove_node_begin(self, event: Event):
        """
        Start removing a node, so it can not allocate containers.
        :param event: Event that has just being fired.
        """
        node = event.node
        if len(node.cgs) > 0:
            # Complete the removal of containers that are finally removed at the current time
            self.dispatch_at_last_time()
        assert len(node.cgs) == 0, "Can not remove a node with containers"
        event = TimedOps.Event(TimedOps.EventTypes.REMOVE_NODE_END, node=node,
                               callback=self._at_remove_node_end)
        self.add_event(self._last_dispatched_time + self.time_args.node_removal_time, event)
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
