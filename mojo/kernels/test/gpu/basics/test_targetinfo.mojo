# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s

from sys.info import simdbitwidth, simdwidthof, warpsize

from gpu.host._compile import _get_nvptx_target
from gpu.host.info import _get_info_from_target
from testing import assert_equal


def test_simdbitwidth():
    assert_equal(128, simdbitwidth[target = _get_nvptx_target()]())
    assert_equal(4, simdwidthof[Float32, target = _get_nvptx_target()]())


def test_warpsize():
    assert_equal(32, warpsize[target = _get_nvptx_target()]())
    assert_equal(
        64, warpsize[target = _get_info_from_target["mi300x"]().target()]()
    )


def main():
    test_simdbitwidth()
    test_warpsize()
