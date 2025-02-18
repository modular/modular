# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the benchmark package for runtime benchmarking.

You can import these APIs from the `benchmark` package. For example:

```mojo
import benchmark
from time import sleep
```

You can pass any `fn` as a parameter into `benchmark.run[...]()`, it will return
a `Report` where you can get the mean, duration, max, and more:

```mojo
fn sleeper():
    sleep(.01)

var report = benchmark.run[sleeper]()
print(report.mean())
```

```output
0.012256487394957985
```

You can print a full report:

```mojo
report.print()
```

```output
---------------------
Benchmark Report (s)
---------------------
Mean: 0.012265747899159664
Total: 1.459624
Iters: 119
Warmup Mean: 0.01251
Warmup Total: 0.025020000000000001
Warmup Iters: 2
Fastest Mean: 0.0121578
Slowest Mean: 0.012321428571428572

```

Or all the batch runs:

```mojo
report.print_full()
```

```output
---------------------
Benchmark Report (s)
---------------------
Mean: 0.012368649122807017
Total: 1.410026
Iters: 114
Warmup Mean: 0.0116705
Warmup Total: 0.023341000000000001
Warmup Iters: 2
Fastest Mean: 0.012295586956521738
Slowest Mean: 0.012508099999999999

Batch: 1
Iterations: 20
Mean: 0.012508099999999999
Duration: 0.250162

Batch: 2
Iterations: 46
Mean: 0.012295586956521738
Duration: 0.56559700000000002

Batch: 3
Iterations: 48
Mean: 0.012380562499999999
Duration: 0.59426699999999999
```

If you want to use a different time unit you can bring in the Unit and pass
it in as an argument:

```mojo
from benchmark import Unit

report.print(Unit.ms)
```

```output
---------------------
Benchmark Report (ms)
---------------------
Mean: 0.012312411764705882
Total: 1.465177
Iters: 119
Warmup Mean: 0.012505499999999999
Warmup Total: 0.025010999999999999
Warmup Iters: 2
Fastest Mean: 0.012015649999999999
Slowest Mean: 0.012421204081632654
```

The unit's are just aliases for `StringLiteral`, so you can for example:

```mojo
print(report.mean("ms"))
```

```output
12.199145299145298
```

Benchmark.run takes four arguments to change the behaviour, to set warmup
iterations to 5:

```mojo
r = benchmark.run[sleeper](5)
```

```output
0.012004808080808081
```

To set 1 warmup iteration, 2 max iterations, a min total time of 3 sec, and a
max total time of 4 s:

```mojo
r = benchmark.run[sleeper](1, 2, 3, 4)
```

Note that the min total time will take precedence over max iterations
"""

from .bencher import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchmarkInfo,
    BenchMetric,
    Mode,
    ThroughputMeasure,
    Format,
)
from .benchmark import Batch, Report, Unit, run
from .compiler import keep
from .memory import clobber_memory
from .quick_bench import QuickBench
