import random
import csv
import time
from scipy.stats import truncnorm
from aws_eu_west_1_c5m5r5 import c5_9xlarge, c5_m5_r5_fm
from ascal.recycling import Recycling
from fcma import Vm, ContainerClass, App, ContainerGroup, RequestsPerTime

RECYCLING_SOLVER = Recycling.hungarian_solver
M_VALUES = range(1, 1001)
M_PARTITION_1 = range(1, 126)
M_PARTITION_2 = range(126, 251)
M_PARTITION_4 = range(251, 501)
M_PARTITION_8 = range(501, 1001)
N_EXP_PER_M = 2
CONTAINER_CLASSES_RAND_IN = range(1, 100)
CPU_MEM_PER_CONTAINER_CLASS_RAND_TRUNC = [0.005, 0.999]
CPU_MEM_CONTAINER_CLASS_RAND_RESOLUTION = 0.0001
CPU_MEM_PER_CONTAINER_CLASS_RAND_MEAN = 0.0075
CPU_MEM_PER_CONTAINER_CLASS_RAND_STD = 0.005

PERFORMANCE_FILE = "performance.csv"
RECYCLING_FILE = "recycling.csv"

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

def random_ccs() -> list[Vm]:
    """
    Return a list of random container classes.
    """
    n_container_classes = random.randint(CONTAINER_CLASSES_RAND_IN[0], CONTAINER_CLASSES_RAND_IN[-1])
    ccs = []
    for cc_index in range(n_container_classes):
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

def reclying_experiment(initial_nodes, final_nodes, partition: int = 1) -> tuple[float, float]:
    """
    Run a single experiment with the given initial and final nodes.
    :param initial_nodes: The initial nodes.
    :param final_nodes: The final nodes.
    :param partition: The partition value for the recycling algorithm.
    :return: A tuple with the duration in seconds and the recycling value of the experiment.
    """

    start = time.time()
    nodes_recycling = Recycling.calculate_node_recycling(initial_nodes, final_nodes,
                                                         solver=RECYCLING_SOLVER,
                                                         partitions=partition)
    duration = time.time() - start
    return duration, nodes_recycling["level"]

def main():
    # Experiment to get calculation times statistics
    print(f"Experiment to get calculation times (from M={M_VALUES[0]} to M={M_VALUES[-1]})")
    duration_stats = []
    for m in M_VALUES:
        partition = 1
        if m > M_PARTITION_1[-1]:
            partition = 2
        if m > M_PARTITION_2[-1]:
            partition = 4
        if m > M_PARTITION_4[-1]:
            partition = 8
        durations = []
        for _ in range(N_EXP_PER_M):
            ccs = random_ccs()
            initial_nodes = random_nodes(m, ccs)
            final_nodes = random_nodes(m, ccs)
            duration, _ = reclying_experiment(initial_nodes, final_nodes, partition)
            durations.append(duration)
        duration_stats.append((min(durations), sum(durations) / N_EXP_PER_M, max(durations)))
        print(f"Completed M={m}")

    # Write calculation time statistics
    with open(PERFORMANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "\t\t min", "\t\t avg", "\t\t max"])
        m = 1
        for calc_time in duration_stats:
            writer.writerow([m, f"\t\t{calc_time[0]:.3f}", f"\t\t{calc_time[1]:.3f}", f"\t\t{calc_time[2]:.3f}"])
            m += 1

    # Experiment to get recycling value statistics using different algorithms
    print(f"\nExperiment to get calculation times (from M={M_PARTITION_1[0]} to M={M_PARTITION_1[-1]})")
    recycling_stats = {"part2": [], "part4": [], "part8": []}
    for m in M_PARTITION_1:
        for _ in range(N_EXP_PER_M):
            recycling_rel_diff_2 = []
            recycling_rel_diff_4 = []
            recycling_rel_diff_8 = []
            for _ in range(N_EXP_PER_M):
                ccs = random_ccs()
                initial_nodes = random_nodes(m, ccs)
                final_nodes = random_nodes(m, ccs)
                _, recycling_value1 = reclying_experiment(initial_nodes, final_nodes, 1)
                _, recycling_value2 = reclying_experiment(initial_nodes, final_nodes, 2)
                _, recycling_value4 = reclying_experiment(initial_nodes, final_nodes, 4)
                _, recycling_value8 = reclying_experiment(initial_nodes, final_nodes, 1)
                recycling_rel_diff_2.append((recycling_value1 - recycling_value2) / recycling_value1)
                recycling_rel_diff_4.append((recycling_value1 - recycling_value4) / recycling_value1)
                recycling_rel_diff_8.append((recycling_value1 - recycling_value8) / recycling_value1)
            recycling_stats["part2"].append((sum(recycling_rel_diff_2) / N_EXP_PER_M, max(recycling_rel_diff_2)))
            recycling_stats["part4"].append((sum(recycling_rel_diff_4) / N_EXP_PER_M, max(recycling_rel_diff_4)))
            recycling_stats["part8"].append((sum(recycling_rel_diff_8) / N_EXP_PER_M, max(recycling_rel_diff_8)))
        print(f"Completed M={m}")

    # Write recycling value statistics
    with open(RECYCLING_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "\t\tavg(2)", "\t\tmax(2)\t\tavg(4)", "\t\tmax(4)",
                         "\t\tavg(8)", "\t\tmax(8)"])
        for m in range(len(recycling_stats["part2"])):
            avg_2, max_2 = recycling_stats["part2"][m]
            avg_4, max_4 = recycling_stats["part4"][m]
            avg_8, max_8 = recycling_stats["part8"][m]
            writer.writerow([m + 1, f"\t\t{avg_2:.3f}", f"\t\t{max_2:.3f}", f"\t\t{avg_4:.3f}", f"\t\t{max_4:.3f}",
                            f"\t\t{avg_8:.3f}", f"\t\t{max_8:.3f}"])

if __name__ == "__main__":
    main()
