from time import time as current_time
from enum import Enum
from math import ceil
from abc import ABC, abstractmethod
from copy import copy
from numpy import percentile as percentile
from fcma import App, Fcma, SolvingPars
from fcma.model import (
    Allocation,
    Vm,
    ContainerClass,
    ContainerGroup,
    InstanceClassFamily,
    RequestsPerTime
)


class AutoscalerTypes(Enum):
    H_REACTIVE = 1     # Horizontal reactive
    HV_REACTIVE = 2    # Horizontal/vertival reactive.
    H_PREDICTIVE = 3   # Horizontal predictive
    HV_PREDICTIVE = 4  # Horizontal/vertical predictive
    FCMA = 5           # FCMA algorithm with speed level 1
    FCMA1 = 5          # FCMA algorithm with speed level 1
    FCMA2 = 6          # FCMA algorithm with speed level 2
    FCMA3 = 7          # FCMA algorithm with speed level 3

class Autoscaler(ABC):
    """
    Abstract class for autoscalers.
    """
    def __init__(self):
        self.container_creation_time = 0  # Container creation time in seconds
        self.container_removal_time = 0   # Container removal time in seconds
        self.node_creation_time = 0       # Node creation time in seconds
        self.node_removal_time = 0        # Node removal time in seconds
        self.system = None                # Application performances on instances class families
        self.time = -1                     # Current time in seconds
        self.apps = None                  # Applications
        self.allocation = None            # Current allocation
        self._changed_allocation = False  # True if allocation changed in the last autoscaling
        self._changed_nodes = False       # True if the nodes changed in the last allocation

    @abstractmethod
    def run(self, workloads: dict[App, RequestsPerTime]) -> tuple[bool, bool, float]:
        pass


class HReactiveAutoscaler(Autoscaler):
    """
    Horizontal and reactive autoscaler for containers and nodes.
    """
    def __init__(self, time_period = 60, desired_cpu_utilization = 0.6, node_utilization_threshold = 0.5):
        """
        Constructor for the horizontal and reactive autoscaler.
        :param time_period: Time period to evaluate a new autoscaling.
        :param desired_cpu_utilization: Desired CPU utilization for the application containers.
        :param node_utilization_threshold: Below this threshold, containers in a node are tried to be allocated
        in other nodes.
        """
        super().__init__()
        self.time_period = time_period
        self.desired_cpu_utilization = desired_cpu_utilization
        self.node_utilization_threshold = node_utilization_threshold
        self._app_load_sum = {}
        self._icf = None # Instance class family
        self._app_cc = {} # Application container classes

    def _initial_allocation(self, workloads: dict[App, RequestsPerTime]) -> Allocation:
        """
        Initial allocation for all the applications, based on their first workload.
        :param workloads: First workload sample for each application.
        :return: The initial allocation
        """

        cgs = []
        for app in workloads:
            cc = self._app_cc[app]
            workload = workloads[app] / self.desired_cpu_utilization
            replicas = int((workload // cc.perf).magnitude)
            if replicas * cc.perf < workloads[app]:
                replicas += 1
            cgs.append(ContainerGroup(cc, replicas))
        return self._allocate_cgs_new_nodes(cgs)

    def _get_replicas(self, app: App, node: Vm = None) -> int:
        """
        Get the number of replicas of an application in the current allocation.
        :param app: Application.
        :param node: Restrict to this node.
        :return: Number of replicas.
        """
        if node is None:
            nodes = list(self.allocation.values())[0]
        else:
            nodes = [node]
        return sum(
            cg.replicas
            for _ in self.allocation
            for node in nodes
            for cg in node.cgs
            if cg.cc.app == app
        )

    def _allocate_cgs_new_nodes(self, cgs: list[ContainerGroup]) -> Allocation:
        """
        Allocate the container groups in nodes of the instance class family.
        :param cgs: Container groups, defined by a container classes and number of replicas.
        :return: An allocation in new nodes.
        """

        self._changed_nodes = True
        self._changed_allocation = True
        allocation = {self._icf:[]}

        # Sort available instance classes by decreasing number of resources
        ics = [ic for ic in self._icf.ics]
        ics_value = {ic: ic.cores.magnitude * ic.mem.magnitude for ic in self._icf.ics}
        ics.sort(key=lambda ic: ics_value[ic], reverse=True)

        # Allocate using the minimum number of the biggest instance class
        new_node = Vm(ics[0])
        allocation[self._icf].append(new_node)
        for cg in cgs:
            replicas = cg.replicas
            while replicas > 0:
                allocated_replicas = new_node.allocate(cg.cc, replicas)
                if allocated_replicas == 0:
                    new_node = Vm(ics[0])
                    allocation[self._icf].append(new_node)
                else:
                    replicas -= allocated_replicas
        # Try to reduce the cost of the latest added virtual machine
        lowest_cost_ic = new_node.ic
        cpu_usage = new_node.ic.cores - new_node.free_cores
        mem_usage = new_node.ic.mem - new_node.free_mem
        for ic in ics:
            if ic.cores >= cpu_usage and ic.mem >= mem_usage and ic.price < lowest_cost_ic.price:
                lowest_cost_ic = ic
        # If a cheaper instance class is found with enough capacity
        if lowest_cost_ic != new_node.ic:
            last_node = Vm(lowest_cost_ic)
            last_node.cgs = new_node.cgs
            last_node.free_cores = last_node.ic.cores - cpu_usage
            last_node.free_mem = last_node.ic.mem - mem_usage
            allocation[self._icf][-1] = last_node

        return allocation

    def _remove_replicas(self, reduced_replicas_app: list[tuple[App, InstanceClassFamily, int]]) -> None:
        """
        Reduce the number of replicas of the given applications.
        :param reduced_replicas_app: A list with the number of replicas to reduce for each application.
        """

        if len(reduced_replicas_app) == 0:
            return

        # First, sort the nodes by increasing size, so replicas are tried to be removed from the
        # smallest nodes to reduce cluster fragmentation
        nodes = self.allocation[self._icf]
        sorted_nodes = copy(nodes)
        nodes_size = {node: node.ic.cores.magnitude * node.ic.mem.magnitude for node in sorted_nodes}
        sorted_nodes.sort(key=lambda node: nodes_size[node])

        for app, replicas_to_remove in reduced_replicas_app:
            for node in sorted_nodes:
                for cg in copy(node.cgs):
                    cc = cg.cc
                    if app == cc.app:
                        if cg.replicas > 0 and replicas_to_remove > 0:
                            self._changed_allocation = True
                        if cg.replicas > replicas_to_remove:
                            node.free_cores += cc.cores * replicas_to_remove
                            node.free_mem += cc.mem[0] * replicas_to_remove
                            cg.replicas -= replicas_to_remove
                            replicas_to_remove = 0
                        else:
                            node.free_cores += cc.cores * cg.replicas
                            node.free_mem += cc.mem[0] * cg.replicas
                            replicas_to_remove -= cg.replicas
                            node.cgs.remove(cg)
                        if replicas_to_remove == 0:
                            break
                if replicas_to_remove == 0:
                    break

    def _allocate_replicas(self, increased_replicas_app: list[tuple[App, InstanceClassFamily, int]]) -> None:
        """
        Allocate application replicas.
        :param increased_replicas_app: A list with the number of replicas to increase for each application.
        """

        if len(increased_replicas_app) == 0:
            return

        cgs = [] # List of container groups, pairs container class and replicas, that can not be allocated
        for app, replicas_to_allocate in increased_replicas_app:
            cc = self._app_cc[app]
            for node in self.allocation[self._icf]:
                if replicas_to_allocate > 0:
                    self._changed_allocation = True
                    allocated_replicas = node.allocate(cc, replicas_to_allocate)
                    replicas_to_allocate -= allocated_replicas
                if replicas_to_allocate == 0:
                    break
            if replicas_to_allocate > 0:
                cgs.append(ContainerGroup(cc, replicas_to_allocate))
        # Allocate the remainder container groups using new nodes
        if len(cgs) > 0:
            new_nodes_allocation = self._allocate_cgs_new_nodes(cgs)
            self.allocation[self._icf].extend(new_nodes_allocation[self._icf])

    def _try_remove_nodes(self) -> None:
        """
        Try to remove nodes with CPU and memory utilization below the utilization threshold.
        """

        # First, try to remove the smallest nodes to reduce cluster fragmentation
        nodes = self.allocation[self._icf]
        nodes_size = {node: node.ic.cores.magnitude * node.ic.mem.magnitude for node in nodes}
        nodes.sort(key=lambda node: nodes_size[node])

        node_list = copy(nodes)
        for node in node_list:
            if node.free_cores / node.ic.cores > self.node_utilization_threshold and \
                    node.free_mem / node.ic.mem > self.node_utilization_threshold:
                # If the node is empty
                if len(node.cgs) == 0:
                    nodes.remove(node)
                    self._changed_allocation = True
                    self._changed_nodes = True
                    continue

                # If the node is not empty, try to allocate its containers into other nodes
                replicas_to_allocate = {app: self._get_replicas(app, node) for app in self.apps}
                # Free capacity in other nodes and replicas to allocate
                other_nodes_free_capacity = {
                    other_node: [other_node.free_cores, other_node.free_mem]
                    for other_node in nodes if other_node != node
                }
                # Simulate the allocation of the replicas in other nodes
                for other_node in other_nodes_free_capacity:
                    apps = {app for app in replicas_to_allocate.keys()}
                    for app in apps:
                        cc = self._app_cc[app]
                        allocatable_replicas = min(other_nodes_free_capacity[other_node][0] // cc.cores,
                                                   other_nodes_free_capacity[other_node][1] // cc.mem[0]).magnitude
                        if allocatable_replicas > 0:
                            allocated_replicas = int(min(allocatable_replicas, replicas_to_allocate[app]))
                            other_nodes_free_capacity[other_node][0] -= allocated_replicas * cc.cores
                            other_nodes_free_capacity[other_node][1] -= allocated_replicas * cc.mem[0]
                            if allocated_replicas == replicas_to_allocate[app]:
                                del replicas_to_allocate[app]
                            else:
                                replicas_to_allocate[app] -= allocated_replicas
                # If all the node replicas can be allocated in other nodes, then allocate them and remove the node
                if len(replicas_to_allocate) == 0:
                    replicas_to_allocate = {app: self._get_replicas(app, node) for app in self.apps}
                    for other_node in [other_node for other_node in node_list if other_node != node]:
                        apps = {app for app in replicas_to_allocate.keys()}
                        for app in apps:
                            if replicas_to_allocate[app] > 0:
                                allocated_replicas = other_node.allocate(self._app_cc[app], replicas_to_allocate[app])
                                replicas_to_allocate[app] -= allocated_replicas
                            if replicas_to_allocate[app] == 0:
                                del replicas_to_allocate[app]
                        if len(replicas_to_allocate) == 0:
                            break
                    nodes.remove(node)
                    self._changed_allocation = True
                    self._changed_nodes = True

    def run(self, workloads: dict[App, RequestsPerTime]) -> tuple[bool, bool, float]:
        """
        Simulate horizontal and reactive autoscaling of containers and nodes.
        :param workloads: Workload for all the applications at the current time.
        :return: A tuple with boolean values for allocation changes, node changes and allocation calculation time.
        """

        initial_time = current_time()
        self.time += 1
        if self.time == 0:
            self._app_load_sum = {app: workloads[app] for app in workloads}
            self._icf = list(self.system.keys())[0][1]
            for app in self.apps:
                cc = ContainerClass(
                    app=app,
                    ic=None,
                    fm=self._icf,
                    cores=self.system[(app, self._icf)].cores,
                    mem=self.system[(app, self._icf)].mem[0],
                    perf=self.system[(app, self._icf)].perf,
                    aggs=(1,)
                )
                self._app_cc[app] = cc
            self.allocation = self._initial_allocation(workloads)
            return True, True, current_time() - initial_time
        else:
            # Update average loads
            for app in self._app_load_sum:
                self._app_load_sum[app] += workloads[app]
            # If not at the time period
            if self.time  % self.time_period > 0:
                return False, False, 0 # Changes may appear only at every time period
            else:
                # Get the applications that require a number of replicas lower or higher than
                # the current number
                reduced_replicas_apps = []
                increased_replicas_apps = []
                for app, icf in self.system:
                    replica_perf = self.system[(app, icf)].perf
                    current_replicas = self._get_replicas(app)
                    average_load = self._app_load_sum[app] / self.time_period
                    average_cpu_utilization = (average_load / (replica_perf * current_replicas)).magnitude
                    desired_replicas = ceil(current_replicas * average_cpu_utilization / self.desired_cpu_utilization)
                    if desired_replicas > current_replicas:
                        increased_replicas_apps.append((app, desired_replicas - current_replicas))
                    elif desired_replicas < current_replicas:
                        reduced_replicas_apps.append((app, current_replicas - desired_replicas))

                # Remove replicas
                self._remove_replicas(reduced_replicas_apps)

                # Allocate new replicas
                self._allocate_replicas(increased_replicas_apps)

                # Try to remove nodes with a low utilization
                self._try_remove_nodes()

                # Get and reset changes
                changes = (self._changed_allocation, self._changed_nodes)
                self._changed_allocation = False
                self._changed_nodes = False

                # Reset load averages
                for app in self._app_load_sum:
                    self._app_load_sum[app] = RequestsPerTime('0 req/hour')

                return changes[0], changes[1], current_time() - initial_time

class HVReactiveAutoscaler(Autoscaler):
    def __init__(self, time_period = 60, desired_cpu_utilization = 0.6, algorithm = AutoscalerTypes.FCMA):
        """
        Constructor for the horizontal and reactive autoscaler.
        :param time_period: Time period to evaluate a new autoscaling.
        :param desired_cpu_utilization: Desired CPU utilization for the application containers.
        """
        super().__init__()
        self.time_period = time_period
        self.desired_cpu_utilization = desired_cpu_utilization
        self._app_load_sum = {}
        self._fcma_speed_level = 1
        if algorithm == AutoscalerTypes.FCMA2:
            self._fcma_speed_level = 2
        elif algorithm == AutoscalerTypes.FCMA3:
            self._fcma_speed_level = 3

    def run(self, app_workloads: dict[App, RequestsPerTime]) -> tuple[bool, bool, float]:
        """
        Simulate horizontal/vertical and reactive autoscaling of containers and nodes.
        :param app_workloads: Workload for all the applications at the current time.
        :return: A tuple with boolean values for allocation changes, node changes and allocation calculation time.
        """

        initial_time = current_time()
        self.time += 1
        if self.time == 0:
            self._app_load_sum = {app: app_workloads[app] for app in app_workloads}
        else:
            # Update average loads
            for app in self._app_load_sum:
                self._app_load_sum[app] += app_workloads[app]
        # If not at the time period
        if self.time % self.time_period > 0:
            return False, False, 0  # Changes may appear only at every time period
        else:
            # Average workloads are artificially incremented to obtain the desired CPU utilization
            incremented_workloads = {}
            for app in self._app_load_sum:
                incremented_workloads[app] = self._app_load_sum[app] / self.desired_cpu_utilization
                if self.time > 0:
                    incremented_workloads[app] /= self.time_period
            # Use FCMA algorithm to get the initial allocation
            fcma_problem = Fcma(self.system, workloads=incremented_workloads)
            solving_pars = SolvingPars(speed_level=self._fcma_speed_level)
            self.allocation = fcma_problem.solve(solving_pars).allocation
            # Reset load averages
            for app in self._app_load_sum:
                self._app_load_sum[app] = RequestsPerTime('0 req/hour')
            return True, True, current_time() - initial_time


class HVPredictiveAutoscaler(Autoscaler):
    def __init__(self, prediction_window = 3600, prediction_percentile = 95, algorithm = AutoscalerTypes.FCMA):
        """
        Constructor for the horizontal and reactive autoscaler.
        :param prediction_percentile: Load prediction percentile.
        :param, prediction_window: prediction window in seconds.
        """
        super().__init__()
        self.prediction_percentile = prediction_percentile
        self.prediction_window = prediction_window
        self._icf = None # Instance class family
        self._fcma_speed_level = 1
        if algorithm == AutoscalerTypes.FCMA2:
            self._fcma_speed_level = 2
        elif algorithm == AutoscalerTypes.FCMA3:
            self._fcma_speed_level = 3
        self._workloads_at_percentil = None

    def workload_predictions(self, app_workloads: dict[App, [RequestsPerTime]]):
        """
        Calculate the load predictions at times multiple of the configured time period and percentil.
        :param app_workloads: Application workloads.
        """

        self._workloads_at_percentil = {app: {} for app in app_workloads}
        n_seconds = len(list(app_workloads.values())[0])
        time = 0
        while time < n_seconds:
            for app, workload_values in app_workloads.items():
                workload = percentile(workload_values[time: min(time + self.prediction_window, n_seconds)],
                                      self.prediction_percentile)
                self._workloads_at_percentil[app][time] = RequestsPerTime(f'{workload} req/s')
            time += self.prediction_window

    def run(self, dummy) -> tuple[bool, bool, float]:
        """
        Simulate horizontal/vertical and predictive autoscaling of containers and nodes.
        :param dummy: This parameter is ignored.
        :return: A tuple with boolean values for allocation changes, node changes and allocation calculation time.
        """

        initial_time = current_time()
        self.time += 1
        if self.time % self.prediction_window == 0:
            # Use FCMA algorithm to get the initial allocation
            app_workloads = {app: self._workloads_at_percentil[app][self.time] for app in self.apps}
            fcma_problem = Fcma(self.system, workloads=app_workloads)
            solving_pars = SolvingPars(speed_level=self._fcma_speed_level)
            self.allocation = fcma_problem.solve(solving_pars).allocation
            return True, True, current_time() - initial_time
        else:
            return False, False, 0


