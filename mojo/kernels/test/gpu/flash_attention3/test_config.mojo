# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
        String(config),
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
