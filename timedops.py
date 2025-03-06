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
        ALLOCATE_CONTAINER_REPLICAS = 1
        GRACE_PERIOD_REPLICAS = 2
        REMOVE_CONTAINER_REPLICAS = 3
        NODE_BILLED = 4
        NODE_READY = 5
        NODE_REMOVED = 6

    @dataclass
    class Event:
        """
        Event for creating/removing nodes or containers.
        """
        type: 'TimedOps.EventTypes'
        containers: tuple[ContainerClass, int, Vm, ContainerClass] | None = None # Information of containers
        node: Vm | None = None # Node to create/remove
        callback: Callable[..., None] | None = None # Function called when the event is fired

    @dataclass(frozen=True)
    class TimingArgs:
        """
        Times required to create/remove nodes and containers.
        """
        node_time_to_billing: int = 0
        node_creation_time: int = 0
        node_removal_time: int = 0
        container_creation_time: int = 0
        container_removal_time: int = 0

    def __init__(self, time_args: TimingArgs):
        """
        Create an event-driven timing system for creation and removal of nodes and containers.
        :param time_args: Times required to create/remove nodes and containers.
        """
        self.time_args = time_args
        self._event_list: list[tuple[int, TimedOps.Event]] = [] # List of events to handle in the form (time, event)
        self._sorting_required = False # Events must be in time order before being processed
        self.node_billing_changed = False # True if node changes affecting node billing occurred at the current time
        self.nodes_ready_changed = False # True if there are new nodes ready at the current time
        self.allocation_changed = False # True if containers are removed or allocated
        self._last_dispatched_time = None # Timne of the ñlast disptached event

    def add_event(self, at_time: int, event: Event):
        """
        Add an event to the event list at the given time.
        :param at_time: Time of addition.
        :param event: Event.
        """
        if len(self._event_list) > 0 and at_time < self._event_list[-1][0]:
            self._sorting_required = True
        self._event_list.append((at_time, event))
        if at_time == self._last_dispatched_time:
            self.dispatch_at_last_time()

    def _update_changes(self, event_type: EventTypes):
        """
        Update changed properties after processing an event.
        :param event_type: Type of event just processed.
        """
        self.node_billing_changed = (event_type == TimedOps.EventTypes.NODE_BILLED) or \
                                    (event_type == TimedOps.EventTypes.NODE_REMOVED) or \
                                    self.node_billing_changed
        self.allocation_changed = (event_type == TimedOps.EventTypes.ALLOCATE_CONTAINER_REPLICAS) or \
                                  (event_type == TimedOps.EventTypes.GRACE_PERIOD_REPLICAS) or \
                                  self.allocation_changed
        self.nodes_ready_changed = (event_type == TimedOps.EventTypes.NODE_READY) or \
                                   self.nodes_ready_changed

    def dispatch_at_last_time(self):
        """
        Dispatch any event in the list with the same time as the last dispatched time.
        It is an opportunity to work with node and container changes at the very first time.
        """
        while len(self._event_list) > 0 and self._event_list[0][0] == self._last_dispatched_time:
            # Get the first even in the event list and execute its callback function
            _, next_event = self._event_list.pop(0)
            if next_event.callback is not None:
                next_event.callback(next_event)
            self._update_changes(next_event.type)

    def dispatch_events(self, until_time: int) -> bool:
        """
        Dispatch all events up to a specified maximum time. Dispatching an event may modify nodes and
        container allocations.
        :param until_time: Last time to dispatch events.
        :return: True if some event has been dispatched.
        """
        self.node_billing_changed = False
        self.allocation_changed = False
        self.nodes_ready_changed = False

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
            # Get the first even in the event list and execute its callback function
            _, next_event = self._event_list.pop(0)
            dispatched_some_event = True
            if next_event.callback is not None:
                next_event.callback(next_event)
                self._update_changes(next_event.type)

    def allocate_container_replicas(self, at_time: int, cc: ContainerClass, replicas: int, node: Vm) -> int:
        """
        Allocate new replicas of a container class on a node. The new replicas have zero performance until
        the container creation time elapses.
        :param at_time: Container creation starts at this time.
        :param cc: Container class.
        :param replicas: Number of replicas to allocate.
        :param node: Node where replicas will be allocated.
        :return: The number of replicas that will be finally allocated.
        """
        if replicas == 0:
            return 0

        # Firstly, allocate containers with zero performance
        zero_perf_cc = ContainerClass(cc.app, cc.ic, cc.fm, cc.cores, cc.mem,
                                      RequestsPerTime("0 req/s"), cc.aggs, cc.agg_level)
        cpu_allocatable_replicas = int(node.free_cores.magnitude / cc.cores.magnitude)
        mem_allocatable_replicas = int(node.free_mem.magnitude / cc.mem[0].magnitude)
        allocatable_replicas = min(cpu_allocatable_replicas, mem_allocatable_replicas, replicas)

        if allocatable_replicas > 0:
            node.cgs.append(ContainerGroup(zero_perf_cc, allocatable_replicas))
            node.free_cores -= allocatable_replicas * zero_perf_cc.cores
            node.free_mem -= allocatable_replicas * zero_perf_cc.mem[0]
            # Create the related event to complete the containers allocation
            event = TimedOps.Event(TimedOps.EventTypes.ALLOCATE_CONTAINER_REPLICAS,
                                   containers=(zero_perf_cc, allocatable_replicas, node, cc),
                                   callback=TimedOps._at_allocate_container_replicas_end)
            new_time = at_time + self.time_args.node_creation_time
            self.add_event(new_time, event)

        return allocatable_replicas

    @staticmethod
    def _at_allocate_container_replicas_end(event: Event):
        """
        Complete the allocation of replicas when the event is fired.
        :param event: Event that has just being fired.
        """
        zero_perf_cc = event.containers[0]
        replicas = event.containers[1]
        node = event.containers[2]
        cc = event.containers[3]

        # The replicas with zero performance will become active
        cgs = [cg for cg in node.cgs]
        for cg in cgs:
            if cg.cc.app == zero_perf_cc.app and cg.replicas == replicas:
                node.cgs.remove(cg)
                break
        found_cg = False # Found container group with ready replicas
        for cg in node.cgs:
            if cg.cc == cc:
                cg.replicas += replicas
                found_cg = True
                break
        if not found_cg:
            node.cgs.append(ContainerGroup(cc, replicas))

    def remove_container_replicas(self, at_time: int, cc: ContainerClass, replicas: int, node: Vm) -> int:
        """
        Remove container replicas in a node. The replicas will become zero performance replicas until they
        are completly removed once the removal time elapses.
        :param at_time: Container removal starts at this time.
        :param cc: Container class.
        :param replicas: Number of replicas to remove.
        :param node: Node where replicas will be removed.
        :return: The number of replicas that will be finally removed.
        """
        assert NodeStates.get_state(node) == NodeStates.READY

        if replicas == 0:
            return 0

        # Calculate the real number of replicas to remove and the related container group
        replicas_to_remove = 0
        cg_with_replicas = None
        for cg in node.cgs:
            if cg.cc == cc:
                replicas_to_remove = min(cg.replicas, replicas)
                cg_with_replicas = cg
                break
        if replicas_to_remove == 0:
            return 0

        # Move the replicas to remove to a new container group with zero performance
        # replicas without associated application (no application means in the process of removing)
        zero_perf_cc = ContainerClass(None, cc.ic, cc.fm, cc.cores, cc.mem,
                                      RequestsPerTime("0 req/s"), cc.aggs, cc.agg_level)
        cg_with_replicas.replicas -= replicas_to_remove
        if cg_with_replicas.replicas == 0:
            node.cgs.remove(cg_with_replicas)
        node.cgs.append(ContainerGroup(zero_perf_cc, replicas_to_remove))

        # Create the event to start the containers grace period. It is used only to detect the performance change
        self.add_event(at_time, TimedOps.Event(TimedOps.EventTypes.GRACE_PERIOD_REPLICAS))

        # Create the related event to complete the containers removal
        event = TimedOps.Event(TimedOps.EventTypes.REMOVE_CONTAINER_REPLICAS,
                               containers=(zero_perf_cc, replicas_to_remove, node, cc),
                               callback=TimedOps._at_remove_container_replicas_end)
        new_time = at_time + self.time_args.container_removal_time
        self.add_event(new_time, event)
        return replicas_to_remove

    @staticmethod
    def _at_remove_container_replicas_end(event: Event):
        """
        Complete the removal of replicas when the event is fired.
        :param event: Event that has just being fired.
        """
        zero_perf_cc = event.containers[0]
        replicas = event.containers[1]
        node = event.containers[2]

        # Remove the zero performance replicas from the container class
        cgs = [cg for cg in node.cgs]
        for cg in cgs:
            if cg.cc == zero_perf_cc and cg.replicas == replicas:
                node.cgs.remove(cg)
                break

        # Update free computational resources in the node
        node.free_cores += replicas * zero_perf_cc.cores
        node.free_mem += replicas * zero_perf_cc.mem[0]

    def create_node(self, at_time: int, node: Vm):
        """
        A preconfigured node is created going through three states:
        1) State TIMING.NODE_BOOTING. It is the initial state. In this state the node is not billed.
        2) State TIMING.NODE_BILLED. The node is billed, but it can not allocate containers yet.
        2) State TIMING.NODE_READY. The node is billed and can allocate containers.
        :param at_time: Node creation starts at this time.
        :param node: Preconfigured node.
        """
        NodeStates.set_state(node, NodeStates.BOOTING)
        booting_event = TimedOps.Event(TimedOps.EventTypes.NODE_BILLED,
                                       node=node, callback=TimedOps._at_start_node_billing)
        new_time = at_time + self.time_args.node_time_to_billing
        self.add_event(new_time, booting_event)

        ready_event = TimedOps.Event(TimedOps.EventTypes.NODE_READY,
                                     node=node, callback=TimedOps._at_create_node_end)
        new_time = at_time + self.time_args.node_creation_time
        self.add_event(new_time, ready_event)

    @staticmethod
    def _at_start_node_billing(event: Event):
        """
        Start billing the node, although it can not allocate containers.
        :param event: Event that has just being fired.
        """
        node = event.node
        NodeStates.set_state(node, NodeStates.BILLED)

    @staticmethod
    def _at_create_node_end(event: Event):
        """
        Make the node elegible to allocate containers.
        :param event: Event that has just being fired.
        """
        node = event.node
        assert NodeStates.get_state(node) == NodeStates.BILLED
        NodeStates.set_state(node, NodeStates.READY)

    def remove_node(self, at_time: int, node: Vm):
        """
        Create an event to remove a node. Once removed the node state becomes TIMING.NODE_REMOVED, and it is
        no longer billed.
        :param at_time: Node removal starts at this time.
        :param node: Node to be removed.
        """
        # Check that it holds only containers in the process of being removed
        for cg in node.cgs:
            assert cg.cc.app is None

        assert NodeStates.get_state(node) == NodeStates.READY
        NodeStates.set_state(node, NodeStates.REMOVING)

        event = TimedOps.Event(TimedOps.EventTypes.NODE_REMOVED,
                               node=node, callback=TimedOps._at_remove_node_end)
        new_time = at_time + self.time_args.node_removal_time
        self.add_event(new_time, event)

    @staticmethod
    def _at_remove_node_end(event: Event):
        """
        Complete the removal of the node when the event is fired.
        :param event: Event that has just being fired.
        """
        node = event.node
        assert NodeStates.get_state(node) == NodeStates.REMOVING
        NodeStates.set_state(node, NodeStates.REMOVED)

