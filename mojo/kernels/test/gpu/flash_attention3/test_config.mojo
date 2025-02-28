# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s -t

from flash_attention3.config import Config
from testing import assert_equal


def test_config():
    var config = Config(
        function_name="flash_attention3",
        hdim=1024,
        dtype=DType.float16,
        split=True,
        paged=True,
        softcap=True,
        pack_gqa=True,
        arch="sm_90",
    )

    assert_equal(
        str(config),
        (
            '{"function_name":flash_attention3, "hdim":1024, "dtype":float16,'
            ' "split":True, "paged":True, "softcap":True, "pack_gqa":True,'
            ' "arch": "sm_90", "binary_hash": ""}'
        ),
    )

    assert_equal(config.function_name, "flash_attention3")
    assert_equal(config.hdim, 1024)
    assert_equal(config.dtype, DType.float16)
    assert_equal(config.split, True)
    assert_equal(config.paged, True)
    assert_equal(config.softcap, True)
    assert_equal(config.pack_gqa, True)
    assert_equal(config.arch, "sm_90")


def main():
    test_config()
