"""Blackwell benchmark wheel library targets.

These targets wrap pre-built SM100 wheels for external benchmark baselines
(currently DeepGEMM) used for Blackwell GPU benchmarking.
"""

def blackwell_bench_wheel(name):
    """Returns the Bazel target for a Blackwell benchmark wheel.

    Args:
        name: The wheel name (e.g., "deep-gemm")

    Returns:
        The fully-qualified Bazel target label
    """
    return "//max/kernels/benchmarks/misc/blackwell_bench:{}".format(name)
