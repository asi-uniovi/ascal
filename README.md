# ASCAL

ASCAL is a simulation environment for analyzing autoscaling algorithms for
containers and nodes. It focuses on minimizing costs for Container as a Service
(CaaS) deployments on Infrastructure as a Service (IaaS) clouds, and currently
targets CPU-bound applications.

## Autoscaling algorithms

- **RHA**: reactive horizontal autoscaling of containers and nodes.
- **RHVA**: reactive horizontal and vertical autoscaling. At each fixed
  deployment window, FCMA calculates an allocation and ASCAL plans the
  transition to it.
- **PHVA**: predictive horizontal and vertical autoscaling. It uses a workload
  percentile for the next prediction window to calculate the next allocation.
- **RHA-RHVA** and **RHA-PHVA**: combine reactive horizontal autoscaling with
  reactive or predictive horizontal and vertical autoscaling.

## Inputs and outputs

Inputs include application workload traces, container CPU/performance/memory
requirements, one or more instance class families, autoscaler settings, and
node/container operation timings. Each application can have different resource
requirements in each instance class family. A family can have child families:
for example, the C5, M5, and R5 child families can be grouped under one root
family.

ASCAL reports application workloads and performance, quality-of-service
measurements, and system cost. It can also write simulation results and
allocations to files and plot them.

## Installation

Clone the repository and install the dependencies with `uv`:

```bash
git clone https://github.com/asi-uniovi/ascal.git
cd ascal
uv sync
```

Install `uv` using the
[official instructions](https://docs.astral.sh/uv/getting-started/installation/).
`uv sync` creates a `.venv` environment and installs ASCAL in editable mode.
Commands can be run with `uv run` without manually activating the environment.

## Run the example

The [examples/ex1/](examples/ex1) directory contains a runnable Python example,
configuration, and notebook:

```bash
cd examples/ex1
uv run ex1.py
```

## Use ASCAL from Python

The example defines two root instance class families in
[aws_eu_west_1_c5_m5_r5.py](examples/aws_eu_west_1_c5_m5_r5.py) and
[aws_eu_west_1_c7a_m7a_r7a.py](examples/aws_eu_west_1_c7a_m7a_r7a.py).

```python
from ascal import Ascal, AscalConfig
from examples import aws_eu_west_1_c5_m5_r5, aws_eu_west_1_c7a_m7a_r7a

ic_families = [
    aws_eu_west_1_c5_m5_r5.c5_m5_r5_fm,
    aws_eu_west_1_c7a_m7a_r7a.c7a_m7a_r7a_fm,
]
config = AscalConfig.get_from_config_yaml("config.yaml", ic_families)
problem = Ascal(config, log="config.log")
problem.run()

problem.write_workload_csv("workloads.csv")
problem.write_performance_csv("performances.csv")
problem.write_cost_csv("cost.csv")
problem.write_allocations("allocations.yaml")
```

The optional `log` argument enables logging to the specified file. See
[examples/ex1/ex1.py](examples/ex1/ex1.py) for plots and additional simulation
data.

## YAML configuration

The YAML file has five top-level properties: `autoscalers`, `autoscaler`,
`timing_args`, `apps`, and `simulation_time`. The following examples follow the
syntax in [examples/ex1/config.yaml](examples/ex1/config.yaml).

### Autoscaler

Define one or more autoscalers under `autoscalers`, then choose one by name in
`autoscaler`. This is the reactive horizontal/vertical autoscaler used by the
example:

```yaml
autoscalers:
  hv_reactive:
    time_period: 60
    desired_cpu_utilization: 0.7
    algorithm: fcma, rbt2
    hot_node_scale_up: true
    hot_container_scale: true

autoscaler: hv_reactive
```

`time_period` is the interval in seconds between allocation calculations.
`desired_cpu_utilization` is the target CPU utilization. `algorithm` specifies
the allocation solver and, optionally, the transition algorithm, separated by
a comma. For example, `fcma, rbt2` selects FCMA and RBT2. The
transition can be omitted to use the default. `hot_node_scale_up` and
`hot_container_scale` enable hot scaling when supported by the simulation.

Other supported autoscaler names are `h_reactive`, `hv_predictive`,
`h_reactive_hv_reactive`, and `h_reactive_hv_predictive`. Their settings and
defaults are validated by ASCAL; see the example configurations in
[examples/pub1/](examples/pub1).

### Timings

`timing_args` specifies operation durations in seconds. For example:

```yaml
timing_args:
  node_time_to_billing: 20
  node_creation_time: 100
  node_removal_time: 10
  container_creation_time: 1
  container_removal_time: 5
```

`node_time_to_billing` is the delay from requesting a node until billing starts;
the node becomes available at `node_creation_time`. Optional hot-scaling times
can also be set: `hot_node_scale_up_time`,
`hot_container_scale_up_time`, and `hot_container_scale_down_time`. By default,
they use the corresponding regular operation times.

### Applications, workload, and container families

Each application has a `load` trace and a `container` mapping. The mapping keys
are the names of the instance class families defined in Python and passed to
`get_from_config_yaml`. Each family entry has its own `cpu`, `mem`, `perf`, and
`aggs` values:

```yaml
apps:
  app0:
    load:
      file: ../../traces/triangle_rand.csv
      time_interval: 1
      repeat: 1
      load_offset: 10
      load_mult: 4
      time_offset: 0
    container:
      c5_m5_r5:
        cpu: 260 mcores
        mem: 0.480 GiB, 0.620 GiB, 0.820 GiB
        perf: 1 req/s
        aggs: [1, 2, 4]
      c7a_m7a_r7a:
        cpu: 300 mcores
        mem: 0.380 GiB, 0.520 GiB
        perf: 3 req/s
        aggs: [1, 2]
```

In this example, `app0` can be placed on either family, with separate
performance and resource requirements for each. `cpu` and `perf` describe the
aggregation-1 container; at aggregation levels 2 and 4, those values are
multiplied by the aggregation level.

`aggs` is an ascending list of positive integers that starts with `1`. `mem`
can be a single memory value, in which case that value is used at every
aggregation level, or a comma-separated list with one value per entry in
`aggs`, in the same order. For example, the three memory values above
correspond to aggregation levels 1, 2, and 4. When providing separate values,
they must increase with the aggregation level, and the per-replica memory at
each level must be less than the level-1 value.

The workload `file` is a CSV trace with an integer sample in its first column
on each row (the first row is treated as a header). `time_interval` is the
number of seconds between samples; `repeat` repeats the trace; `load_offset` is
added to each sample; and `load_mult` multiplies the result.
`time_offset` repeats the first processed sample at the beginning of the trace
for the given number of samples.

### Simulation time

Set `simulation_time` in seconds:

```yaml
simulation_time: 3600
```

If the workload trace is longer than the simulation, it is truncated. If it is
shorter, its last workload value is repeated until the simulation ends.
