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

CONTAINER_CLASSES_RAND_IN = range(1, 100)
CPU_MEM_PER_CONTAINER_CLASS_RAND_TRUNC = [0.005, 0.999]
CPU_MEM_CONTAINER_CLASS_RAND_RESOLUTION = 0.0001
CPU_MEM_PER_CONTAINER_CLASS_RAND_MEAN = 0.0075
CPU_MEM_PER_CONTAINER_CLASS_RAND_STD = 0.005

# Number of repeated experiments for each number of nodes, M value
N_EXP_PER_M = 5

# Values for calculating performance
M_VALUES_PERF = range(1, 201) # Range of M values
PARTITION_MIN_VALUE = 101 # Minimum m value to use partitions
N_PARTITIONS = 4 # Number of partitions for  m>= PARTITION_MIN_VALUE

# Values for calculating solution optimality with partitions
M_VALUES_OPT = range(1, 101) # Range of M values
PARTITIONS = (2, 4, 8) # Partitions for all the M values

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
    initial_time = time.time()
    # Experiments to get calculation time statistics
    print(f"Experiment to get calculation times (from M={M_VALUES_PERF[0]} to M={M_VALUES_PERF[-1]})")
    with open(PERFORMANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M", "\t\t min", "\t\t avg", "\t\t max"])
        for m in M_VALUES_PERF:
            partition = 1
            if m >= PARTITION_MIN_VALUE:
                partition = N_PARTITIONS
            durations = []
            for _ in range(N_EXP_PER_M):
                ccs = random_ccs()
                initial_nodes = random_nodes(m, ccs)
                final_nodes = random_nodes(m, ccs)
                duration, _ = reclying_experiment(initial_nodes, final_nodes, partition)
                durations.append(duration)
            min_durarion = min(durations)
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            writer.writerow([m, f"\t\t{min_durarion:.3f}", f"\t\t{avg_duration:.3f}", f"\t\t{max_duration:.3f}"])
            f.flush()
            print(f"Time: {time.time()-initial_time:.1f} seconds. Completed M={m}")

    # Experiments to estimate optimality using different algorithms
    print(f"\nExperiments to estimate optimality (from M={M_VALUES_OPT[0]} to M={M_VALUES_OPT[-1]})")
    with open(RECYCLING_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["M"] + ["\t\tavg(g)"] + ["\t\tmax(g)"] +
                        [item for p in PARTITIONS for item in (f"\t\tavg({p})", f"\t\tmax({p})")])
        for m in M_VALUES_OPT:
            recycling_rel_diffs = {p: [] for p in PARTITIONS}
            recycling_rel_diffs['g'] = []
            for _ in range(N_EXP_PER_M):
                ccs = random_ccs()
                initial_nodes = random_nodes(m, ccs)
                final_nodes = random_nodes(m, ccs)
                _, recycling_value_opt = reclying_experiment(initial_nodes, final_nodes, 1)
                _, recycling_value_greedy = reclying_experiment(initial_nodes, final_nodes, 1,
                                                                Recycling.greedy_solver)
                recycling_rel_diffs['g'].append((recycling_value_opt - recycling_value_greedy) / recycling_value_opt)
                for p in PARTITIONS:
                    _, recycling_value = reclying_experiment(initial_nodes, final_nodes, p)
                    recycling_rel_diffs[p].append((recycling_value_opt - recycling_value) / recycling_value_opt)
            avg_parts = {p: sum(recycling_rel_diffs[p]) / N_EXP_PER_M for p in PARTITIONS}
            max_parts = {p: max(recycling_rel_diffs[p]) for p in PARTITIONS}
            avg_g = sum(recycling_rel_diffs['g']) / N_EXP_PER_M
            max_g = max(recycling_rel_diffs['g'])
            writer.writerow([m] + [f"\t\t{avg_g:.3f}"] + [f"\t\t{max_g:.3f}"] +
                            [item for p in PARTITIONS for item in (f"\t\t{avg_parts[p]:.3f}",
                                                                   f"\t\t{max_parts[p]:.3f}")])
            print(f"Time: {time.time()-initial_time:.1f} seconds. Completed M={m}")

if __name__ == "__main__":
    main()
