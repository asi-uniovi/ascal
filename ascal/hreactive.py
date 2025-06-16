"""
Implement the horizontal reactive autoscaler
"""

import copy
from collections import defaultdict
from time import time as current_time
from math import ceil
from fcma import (
    App,
    Allocation,
    Vm,
    ContainerClass,
    InstanceClass,
    ContainerGroup,
    RequestsPerTime,
)
from ascal.timedops import TimedOps
from ascal.nodestates import NodeStates
from ascal.autoscalers import Autoscaler, AutoscalerStatistics
from ascal.helper import get_app_ccs, get_required_nodes, mncf_allocation


class HReactiveAutoscaler(Autoscaler):
    """
    Horizontal and reactive autoscaler for containers and nodes.
    """
    def __init__(self, time_period:int = 60, desired_cpu_utilization: float = 0.6,
                 node_utilization_threshold:float = 0.5, aggs: dict[App, int] = None,
                 timing_args: TimedOps.TimingArgs | None = None):
        """
        Constructor for the horizontal and reactive autoscaler.
        :param time_period: Time period to evaluate a new autoscaling.
        :param desired_cpu_utilization: Desired CPU utilization for the application containers.
        :param node_utilization_threshold: Below this threshold, a node is tried to be removed.
        :param aggs: The aggregation level for each application. A None value allows the use of the
        application aggregation levels.
        :param timing_args: Timings for creation/removal of nodes and containers.
        """
        super().__init__(timing_args)
        self.time_period = time_period
        self.desired_cpu_utilization = desired_cpu_utilization
        self.node_utilization_threshold = node_utilization_threshold
        self._app_loads: dict[App, list[RequestsPerTime]] = {} # Application workloads in a time period
        self._ics: list[InstanceClass] = None # Instance class family
        self._app_ccs: dict[App: list[ContainerClass]] = {} # Application container classes
        self._desired_app_replicas: dict[App, int] = {} # Desired application replicas
        self._enable_node_creation = True # Set to enable node creation
        self._enable_node_removal = True # Set to enable node removal
        self._enable_container_allocation = True # Set to enable container allocation
        self._enable_container_removal = True # Set to enable container removal
        self._new_nodes_required = False # True if new nodes are required to allocate more recplicas
        self._aggs: dict[App, list[int]] = aggs

    def _initial_allocation(self, workloads: dict[App, RequestsPerTime]) -> Allocation:
        """
        Initial allocation for all the applications, based on their first workload.
        Creation/removal times for nodes and containers are assumed to be zero in the initial alocation.
        :param workloads: First workload sample for each application.
        :return: The initial allocation.
        """

        incremented_workloads = {}
        for app in workloads:
            incremented_workloads[app] = workloads[app] / self.desired_cpu_utilization
        allocation = mncf_allocation(self.system, incremented_workloads)
        for node in allocation:
            NodeStates.set_state(node, NodeStates.READY)
        self.log(f'Initial allocation with {tuple(str(node) for node in allocation)}')
        for node in allocation:
            for cg in node.cgs:
                self.log(f'  - Allocated {cg.replicas} replicas {cg.cc} on node {str(node)}')

        return allocation

    def _get_replicas(self, app: App, node: Vm = None) -> int:
        """
        Get the number of equivalent agg=1 replicas of an application in the current allocation.
        Replicas that are in the process of being removed are ignored.
        :param app: Application.
        :param node: Restrict to this node.
        :return: Number of replicas.
        """

        if node is None:
            nodes = [n for n in self.allocation if NodeStates.get_state(n) in (NodeStates.READY, NodeStates.MOVINGC)]
        elif NodeStates.get_state(node) not in (NodeStates.READY, NodeStates.MOVINGC):
            return 0
        else:
            nodes = [node]
        return sum(
            cg.replicas * cg.cc.agg_level
            for node in nodes
            for cg in node.cgs
            if cg.cc.app == app
        )

    def _remove_excess_of_replicas(self):
        """
        Reduce the surplus number of application's replicas.
        """
        # Number of equivalent 1x replicas, i.e., with aggregation level 1
        replicas_to_remove = {
            app: self._get_replicas(app) - self._desired_app_replicas[app]
            for app in self.apps
            if self._get_replicas(app) - self._desired_app_replicas[app] > 0
        }
        if len(replicas_to_remove) <= 0:
            return

        # Firstly, sort the nodes by increasing size, so replicas are tried to be removed from the
        # smallest nodes to reduce cluster fragmentation
        nodes = [node for node in self.allocation if NodeStates.get_state(node) == NodeStates.READY]
        nodes_size = {node: node.ic.cores.magnitude * node.ic.mem.magnitude for node in nodes}
        nodes.sort(key=lambda node: nodes_size[node])

        for app, replicas in replicas_to_remove.items():
            for node in nodes:
                # Select container groups allocating the application containers, sorted by decreasing
                # aggregation levels
                cgs = [cg for cg in node.cgs if cg.cc.app == app]
                cgs.sort(key=lambda cg: cg.cc.agg_level, reverse=True)
                while len(cgs) > 0:
                    replicas_to_remove = replicas // cgs[0].cc.agg_level
                    if replicas_to_remove > 0:
                        removed_replicas = \
                            self._timedops.remove_container_replicas(self.time, cgs[0].cc, replicas_to_remove, node)
                        replicas -= removed_replicas * cgs[0].cc.agg_level
                        if replicas == 0:
                            break
                    cgs.pop(0)

    def _allocate_deficit_replicas(self) -> list[ContainerGroup]:
        """
        Try allocating replicas for those applications with a deficit.
        :return: The replicas that can not be allocated.
        """
        # Number of equivalent agg=1 replicas
        replicas_to_add = {
            app: self._desired_app_replicas[app] - self._get_replicas(app)
            for app in self.apps
            if self._desired_app_replicas[app] - self._get_replicas(app) > 0
        }
        if len(replicas_to_add) == 0:
            return []

        if not self._enable_container_allocation:
            return replicas_to_add

        cgs = [] # List of container groups that can not be allocated in the current nodes
        nodes = [node for node in self.allocation if NodeStates.get_state(node) == NodeStates.READY]
        for app, replicas_agg1 in replicas_to_add.items():
            # Allocate starting with the largest containers
            for cc in self._app_ccs[app]:
                for node in nodes:
                    if cc != self._app_ccs[app][-1]:
                        replicas_to_allocate = replicas_agg1 // cc.agg_level
                    else:
                        replicas_to_allocate = int(ceil(replicas_agg1 / cc.agg_level))
                    if replicas_to_allocate > 0:
                        allocated_replicas =\
                            self._timedops.allocate_container_replicas(self.time, cc, replicas_to_allocate, node)
                        replicas_agg1 -= allocated_replicas * cc.agg_level
                    if replicas_agg1 == 0:
                        break
                if replicas_agg1 == 0:
                    break
            if replicas_agg1 > 0:
                # The remaining replicas go to unallocated container groups with the avaliable allocation levels
                for cc in self._app_ccs[app]:
                    agg_level = cc.agg_level
                    if cc != self._app_ccs[app][-1]:
                        replicas_agg_level = replicas_agg1 // agg_level
                    else:
                        replicas_agg_level = int(ceil(replicas_agg1 / agg_level))
                    if replicas_agg_level > 0:
                        cgs.append(ContainerGroup(self._app_ccs[app][-1], replicas_agg_level))
                        replicas_agg1 -= replicas_agg_level * agg_level
            assert -self._app_ccs[app][-1].agg_level < replicas_agg1 <= 0, "Invalid number of deficit containers"

        return cgs

    def _create_required_nodes(self, cgs: list[ContainerGroup]) -> bool:
        """
        Create new nodes to allocate the remaining container replicas.
        :param cgs: Container groups with the replicas to allocate.
        :return: True if new nodes need to be created.
        """
        # Ignore those containers that can be allocated on nodes that are not ready yet
        cgs_to_allocate = [cg for cg in cgs]
        no_ready_nodes = [
            node
            for node  in self.allocation
            if NodeStates.get_state(node) in [NodeStates.BOOTING, NodeStates.BILLED]
        ]
        cgs_allocated = []
        for cg in cgs:
            cc = cg.cc
            replicas = cg.replicas
            for node in no_ready_nodes:
                cpu_allocatable_replicas = int(node.free_cores.magnitude / cc.cores.magnitude + Autoscaler._DELTA)
                mem_allocatable_replicas = int(node.free_mem.magnitude / cc.mem[0].magnitude + Autoscaler._DELTA)
                allocatable_replicas = min(cpu_allocatable_replicas, mem_allocatable_replicas, replicas)
                replicas -= allocatable_replicas
                cg.replicas -= allocatable_replicas
                node.free_cores -= allocatable_replicas * cc.cores
                node.free_mem -= allocatable_replicas * cc.mem[0]
                cgs_allocated.append((node, ContainerGroup(cc, allocatable_replicas)))
                if cg.replicas == 0:
                    cgs_to_allocate.remove(cg)
                    break

        # Recover the allocation state of no ready nodes
        for node in no_ready_nodes:
            node.cgs = []
            node.free_cores = node.ic.cores
            node.free_mem = node.ic.mem

        # Create the nodes
        if len(cgs_to_allocate) > 0 and self._enable_node_creation:
            new_nodes = get_required_nodes(self._ics, cgs)
            # Change node IDs so that they are new
            current_ids = defaultdict(lambda: [])
            for node in self.allocation:
                current_ids[node.ic].append(node.id)
            for ic in current_ids:
                current_ids[ic].sort() # Sort IDs by increasing values
            for node in new_nodes:
                if node.ic in current_ids:
                    node.id = current_ids[node.ic][-1] + 1
                    current_ids[node.ic].append(node.id)
            # Now new nodes are ready to be created
            for new_node in new_nodes:
                self._timedops.create_node(self.time, new_node)
            self.allocation.extend(new_nodes)
            # Try to allocate replicas, since new ready nodes may be available inmediately when
            # the creation time is zero.
            self._allocate_deficit_replicas()

        return len(cgs_to_allocate) > 0

    def _allocate_node_replicas(self, node: Vm, other_nodes: Allocation, sim: bool = False) -> bool:
        """
        Allocate application replicas currently allocated in one node in the other nodes.
        Replicas in the process of being removed are ignored.
        :param node: The node where the replicas are allocated.
        :param other_nodes: Nodes where the replicas will be allocated.
        :param sim: When it is true, allocation is simmulated and so it is not actually performed.
        :return: True if the replicas can be allocated.
        """

        if sim:
            other_nodes = copy.deepcopy(other_nodes)
        else:
            NodeStates.set_state(node, NodeStates.MOVINGC)
            self._timedops.timed_log(self.time, f'Moving containers from node {node} to other nodes')

        # Get equivalent agg=1 application replicas, including those that are starting and ignoring those
        # that are being removed
        app_replicas_agg1 = {app: self._get_replicas(app, node) for app in self.apps}
        for app, replicas_agg1 in app_replicas_agg1.items():
            if replicas_agg1 == 0:
                continue
            for cc in self._app_ccs[app]:
                # The number of replicas is incremented if it is lower than the minimum aggregation level
                if replicas_agg1  < self._app_ccs[app][-1].agg_level:
                    replicas_agg1 = self._app_ccs[app][-1].agg_level
                for other_node in other_nodes:
                    allocatable_replicas_cpu = \
                        int((other_node.free_cores / cc.cores).magnitude + Autoscaler._DELTA)
                    allocatable_replicas_mem = \
                        int((other_node.free_mem / cc.mem[0]).magnitude + Autoscaler._DELTA)
                    allocatable_replicas = min(allocatable_replicas_cpu, allocatable_replicas_mem,
                                               replicas_agg1 // cc.agg_level)
                    if allocatable_replicas > 0:
                        if not sim:
                            allocated_replicas = self._timedops.allocate_container_replicas(self.time, cc,
                                                                                            allocatable_replicas,
                                                                                            other_node)
                            assert allocatable_replicas == allocated_replicas, "Can not allocate replicas"
                        replicas_agg1 -= allocatable_replicas * cc.agg_level
                        if replicas_agg1 == 0:
                            break
                if replicas_agg1 == 0:
                    break
            if replicas_agg1 > 0:
                return False
        return True

    def _remove_low_utilization_nodes(self) -> None:
        """
        Try to remove nodes with CPU and memory utilization below the utilization threshold.
        """

        if not self._enable_container_allocation:
            return

        # Nodes can not be removed while creating new nodes
        for node in self.allocation:
            node_state = NodeStates.get_state(node)
            if node_state == NodeStates.BOOTING or node_state == NodeStates.BILLED:
                return 0

        # Firstly, try to remove the largest nodes to reduce cost
        # Only nodes in the ready state are elegible
        nodes = [node for node in self.allocation if NodeStates.get_state(node) == NodeStates.READY]
        nodes_size = {node: node.ic.cores.magnitude * node.ic.mem.magnitude for node in nodes}
        nodes.sort(key=lambda n: nodes_size[n], reverse=True)

        for node in nodes:
            # If the node is empty
            if len(node.cgs) == 0:
                self._timedops.remove_node(self.time, node)
            # Check the threshold utilization condition
            elif node.free_cores / node.ic.cores > self.node_utilization_threshold and \
                    node.free_mem / node.ic.mem > self.node_utilization_threshold:
                other_nodes = [
                    other_node
                    for other_node in nodes
                    if other_node != node and NodeStates.get_state(other_node) == NodeStates.READY
                ]
                # Simulate the allocation of the node replicas in other nodes
                if self._allocate_node_replicas(node, other_nodes, sim=True):
                    # Now node replicas are actually allocated in the other nodes
                    assert self._allocate_node_replicas(node, other_nodes), "Replicas must be allocated"
                    # Remove containers in the node
                    cgs = [cg for cg in node.cgs]
                    for cg in cgs:
                        # If the application is not being removed at this time
                        if cg.cc.app is not None:
                            new_time = self.time + self.timing_args.container_creation_time
                            self._timedops.remove_container_replicas(new_time, cg.cc, cg.replicas, node)
                    # Start the node removal after the allocation of the moved containers
                    new_time = self.time + self.timing_args.container_creation_time + \
                                self.timing_args.container_removal_time
                    self._timedops.remove_node(new_time, node)

    def _clear_removed_nodes(self):
        """
        Clear the removed nodes.
        """
        nodes_to_clear = [node for node in self.allocation if NodeStates.get_state(node) == NodeStates.REMOVED]
        for node in nodes_to_clear:
            self.allocation.remove(node)

    def _set_desired_replicas(self):
        """
        Set the desired number of equivalent agg=1 replicas for each application.
        """
        for app, icf in self.system:
            replica_perf = self.system[(app, icf)].perf
            current_replicas = self._get_replicas(app)
            average_load = sum(self._app_loads[app][-self.time_period:]) / self.time_period
            average_cpu_utilization = (average_load / (replica_perf * current_replicas)).magnitude
            if average_cpu_utilization == 0:
                # At least one replica with the minimum aggregation is required
                self._desired_app_replicas[app] = self._app_ccs[app][-1].agg_level
            else:
                self._desired_app_replicas[app] = \
                    ceil(current_replicas * average_cpu_utilization / self.desired_cpu_utilization)
            self._timedops.timed_log(self.time,
                                     f'Load of {app}: {average_load.to("req/s").magnitude:.2f} req/s')
            self._timedops.timed_log(self.time, f'Current replicas 1x of {app} {current_replicas}, '
                                                 f'desired {self._desired_app_replicas[app]}')

    def run(self, app_workloads: dict[App, RequestsPerTime]) -> AutoscalerStatistics:
        """
        Simulate horizontal reactive autoscaling of containers and nodes in the one second.
        :param app_workloads: Workload for all the applications at the current time.
        :return: Simulation statistics.
        """
        initial_time = current_time() # Reference to calculate the processing time

        # If it is the first execution
        if self.time == 0:
            # Prepare data required in the next times
            self._app_loads = {app: [workload] for app, workload in app_workloads.items()}
            self._ics = list(self.system.keys())[0][1].ics # Available instance classes
            # Application container classes sorted by decreasing aggregation level
            self._app_ccs = get_app_ccs(self.system, self._aggs)
            for app in self._app_ccs:
                self._app_ccs[app].sort(key=lambda c: c.agg_level, reverse=True)
            # Node creation time and container allocation time are assumed to be zero for the initial allocation
            self.allocation = self._initial_allocation(app_workloads)
            self.time += 1
            statistics = AutoscalerStatistics(True, True, 0, current_time() - initial_time,
                                              Autoscaler.INVALID_RECYCLING, Autoscaler.INVALID_RECYCLING)
            return statistics
        else:
            # Update the application loads
            for app in app_workloads:
                self._app_loads[app].append(app_workloads[app])
            # Dispatch events until the current time
            self._timedops.dispatch_events(self.time)
            # Clear from the allocation nodes in the removed state
            self._clear_removed_nodes()
            # At the beginning of each time period
            if self.time % self.time_period == 0:
                self._set_desired_replicas()
                if self._enable_container_removal:
                    self._remove_excess_of_replicas()
                unallocatable_replicas = self._allocate_deficit_replicas()
                self._new_nodes_required = self._create_required_nodes(unallocatable_replicas)
                if self._enable_node_removal:
                    self._remove_low_utilization_nodes()
                # Reset loads
                for app in self._app_loads:
                    self._app_loads[app].clear()

            # At any other time try to allocate replicas of applications with deficit if
            # new nodes are available or the allocation has changed
            if self._timedops.new_nodes_ready or self._timedops.perf_changed:
                self._allocate_deficit_replicas()

            self.time += 1
            statistics = AutoscalerStatistics(self._timedops.perf_changed, self._timedops.node_billing_changed,
                                              0, current_time() - initial_time, Autoscaler.INVALID_RECYCLING,
                                              Autoscaler.INVALID_RECYCLING)
            return statistics

