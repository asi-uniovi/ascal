import random
import csv
import time
from scipy.stats import truncnorm
from aws_eu_west_1_c5m5r5 import c5_9xlarge, c5_m5_r5_fm
from ascal.recycling import Recycling, RecyclingSolverType
from fcma import Vm, ContainerClass, App, ContainerGroup, RequestsPerTime

# Default solver for recycling.
# Options: Recycling.hungarian_solver, Recycling.greedy_solver, Recycling.ilp_solver
RECYCLING_SOLVER = Recycling.hungarian_solver

CONTAINER_CLASSES_RAND_IN = range(1, 101) 
M_RAND_IN = range(1, 101) 
CPU_MEM_PER_CONTAINER_CLASS_RAND_TRUNC = [0.005, 0.999]
CPU_MEM_CONTAINER_CLASS_RAND_RESOLUTION = 0.0001
CPU_MEM_PER_CONTAINER_CLASS_RAND_MEAN = 0.0075
CPU_MEM_PER_CONTAINER_CLASS_RAND_STD = 0.005

# Number of repeated experiments for each number of nodes, M value
N_EXP_PER_VALUE = 10

# Values for calculating performance varying the number of nodes
M_VALUES = range(1, 101) # Range of M values

# Values for calculating performance varying the number of container classes
CC_VALUES = range(1, 101) # Range of different container classes
M_VALUES_CC = (10, 20, 40, 80) # M values for CC experiments

PERFORMANCE_FILE_M = "hungarianperf-m.csv"
PERFORMANCE_FILE_CC = "hungarianperf-cc.csv"

def sample_truncated_normal(mean, std, low, high, resolution):
    """
    Sample from a truncated normal distribution using scipy.
    The distribution is truncated between 'low' and 'high'.
    :param mean: Mean.
    :param std: Standard deviation.
    :param low: Low value to truncate.
    :param high: High value to truncate.
    :param resolution: Returned value is a rounded multiple of this parameter.
    :return: A sample of the truncated normal distribution.
    """
    a, b = (low - mean) / std, (high - mean) / std
    return round(truncnorm.rvs(a, b, loc=mean, scale=std) / resolution) * resolution

def allocate_random_containers(nodes: list[Vm], ccs: list[ContainerClass]):
    """
    Allocate random containers in the nodes.
    :param nodes: A list with the nodes.
    :param ccs: A list with the available container classes.
    """
    for node in nodes:
        for cc in ccs:
            node.cgs.append(ContainerGroup(cc, 0))
        replicas = [0 for _ in ccs]
        free_cores = node.ic.cores
        free_mem = node.ic.mem
        while True:
            cc_index = random.randint(0, len(ccs) - 1)
            cc_cores = ccs[cc_index].cores
            cc_mem = ccs[cc_index].mem
            if free_cores < cc_cores or free_mem < cc_mem[0]:
                break
            replicas[cc_index] += 1
            free_cores -= cc_cores
            free_mem -= cc_mem[0]
        for cc_cg_index in range(len(ccs)):
            node.cgs[cc_cg_index].replicas = replicas[cc_cg_index]
        for cg in node.cgs[:]:
            if cg.replicas == 0:
                node.cgs.remove(cg)
        node.free_cores = free_cores
        node.free_mem = free_mem

def random_ccs(n: int = -1) -> list[Vm]:
    """
    Return a list of random container classes.
    param n: Number of container classes to generate.
    :return: A list of random container classes.
    """
    if n == -1:
        n = random.randint(CONTAINER_CLASSES_RAND_IN[0], CONTAINER_CLASSES_RAND_IN[-1])
    n_container_classes = random.randint(1, n)
    ccs = []
    for _ in range(n_container_classes):
        rel_cpu = sample_truncated_normal(CPU_MEM_PER_CONTAINER_CLASS_RAND_MEAN,
                                          CPU_MEM_PER_CONTAINER_CLASS_RAND_STD,
                                          CPU_MEM_PER_CONTAINER_CLASS_RAND_TRUNC[0],
                                          CPU_MEM_PER_CONTAINER_CLASS_RAND_TRUNC[1],
                                          CPU_MEM_CONTAINER_CLASS_RAND_RESOLUTION)
        rel_mem = sample_truncated_normal(CPU_MEM_PER_CONTAINER_CLASS_RAND_MEAN,
                                          CPU_MEM_PER_CONTAINER_CLASS_RAND_STD,
                                          CPU_MEM_PER_CONTAINER_CLASS_RAND_TRUNC[0],
                                          CPU_MEM_PER_CONTAINER_CLASS_RAND_TRUNC[1],
                                          CPU_MEM_CONTAINER_CLASS_RAND_RESOLUTION)
        cc = ContainerClass(App('app'), c5_9xlarge, c5_m5_r5_fm, rel_cpu * c5_9xlarge.cores,
                            rel_mem * c5_9xlarge.mem, RequestsPerTime("1 req/s"), (1,), 1)
        ccs.append(cc)
    return ccs

def random_nodes(m: int, ccs: list[ContainerClass]) -> list[Vm]:
    """
    Return a list of nodes of instance class c5.9xlarge with ramdom containers.
    :param m: The number of nodes.
    :param ccs: List of container classes.
    :return: A list of nodes with random containers.
    """
    nodes = [Vm(c5_9xlarge) for _ in range(m)]
    allocate_random_containers(nodes, ccs)
    return nodes

def reclying_experiment(initial_nodes, final_nodes, partition: int = 1,
                        solver: RecyclingSolverType = Recycling.hungarian_solver) -> tuple[float, float]:
    """
    Run a single experiment with the given initial and final nodes.
    :param initial_nodes: The initial nodes.
    :param final_nodes: The final nodes.
    :param  solver: Solver for the recycleing problem.
    :param partition: The partition value for the recycling algorithm.
    :return: A tuple with the duration in seconds and the recycling value of the experiment.
    """
    start = time.time()
    nodes_recycling = Recycling.calculate_node_recycling(initial_nodes, final_nodes,
                                                         solver=solver,
                                                         partitions=partition)
    duration = time.time() - start
    return duration, nodes_recycling["level"]


def main():
    # Experiments to get calculation time statistics and estimate optimality using hungarian algorithm
    initial_time = time.time()

    print(f"\nRecycling experiments (from CC={CC_VALUES[0]} to CC={CC_VALUES[-1]}) and M={M_VALUES_CC})")
    with open(PERFORMANCE_FILE_CC, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["CC"] + [f"\t\tavg({m})" for m in M_VALUES_CC])
        for number_ccs in CC_VALUES:
            durations = {m: [] for m in M_VALUES_CC}
            for _ in range(N_EXP_PER_VALUE):
                ccs = random_ccs(number_ccs)    
                for m in M_VALUES_CC:
                    initial_nodes = random_nodes(m, ccs)
                    final_nodes = random_nodes(m, ccs)
                    duration_opt, _ = reclying_experiment(initial_nodes, final_nodes)
                    durations[m].append(duration_opt)
            avgs = {m: sum(durations[m]) / N_EXP_PER_VALUE for m in M_VALUES_CC}
            writer.writerow([number_ccs] + [f"\t\t{avgs[m]:.4f}" for m in M_VALUES_CC])
            f.flush()
            print(f"Time: {time.time()-initial_time:.1f} seconds. Completed CC={number_ccs}")
        f.close

    print(f"\nRecycling experiments (from M={M_VALUES[0]} to M={M_VALUES[-1]})")
    with open(PERFORMANCE_FILE_M, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M"] + ["\t\tmin"] + ["\t\tavg"] + ["\t\tmax"])
        for m in M_VALUES:
            durations = []
            for _ in range(N_EXP_PER_VALUE):
                ccs = random_ccs()
                initial_nodes = random_nodes(m, ccs)
                final_nodes = random_nodes(m, ccs)
                duration_opt, _ = reclying_experiment(initial_nodes, final_nodes)
                durations.append(duration_opt) 
            min_p = min(durations)
            avg_p = sum(durations) / N_EXP_PER_VALUE
            max_p = max(durations)
            writer.writerow([m] + [f"\t\t{min_p:.4f}"] + [f"\t\t{avg_p:.4f}"] + [f"\t\t{max_p:.4f}"])
            f.flush()
            print(f"Time: {time.time()-initial_time:.1f} seconds. Completed M={m}")

if __name__ == "__main__":
    main()
