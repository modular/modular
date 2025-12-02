# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Blackwell benchmark wheel library targets.

These targets wrap pre-built SM100 wheels for external benchmark baselines
(DeepGEMM, FlashInfer, flash-attention) used for Blackwell GPU benchmarking.
"""

def blackwell_bench_wheel(name):
    """Returns the Bazel target for a Blackwell benchmark wheel.

    Args:
        name: The wheel name (e.g., "deep-gemm", "flashinfer", "flash-attn")

    Returns:
        The fully-qualified Bazel target label
    """
    return "//bazel/pip/blackwell_bench:{}".format(name)
