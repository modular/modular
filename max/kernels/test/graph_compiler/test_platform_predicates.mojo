# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
# The platform predicates that registrations can use. Each one must convert to
# a `PlatformPredicate` and carry the expected levels. The predicates that Mojo
# must reject are in `test_platform_predicates_fail.mojo`.

from extensibility import (
    ACCELERATOR,
    CPU,
    CUDA,
    HIP,
    METAL,
    PlatformExpr,
    PlatformPredicate,
    api,
    arch,
    model,
)
from std.testing import TestSuite, assert_true

comptime HOPPER = ACCELERATOR & CUDA & arch("sm_90a")
comptime SM100 = ACCELERATOR & CUDA & arch("sm_100a")
comptime SM103 = ACCELERATOR & CUDA & arch("sm_103a")
comptime SM100_FAMILY = SM100 | SM103
comptime B200 = SM100 & model("NVIDIA B200")
comptime TMA = HOPPER | SM100_FAMILY


def _accepts[expr: PlatformExpr](cond: PlatformPredicate[expr]) -> Bool:
    """Takes a predicate the way `register` does."""
    return True


def test_shipped_aliases() raises:
    assert_true(_accepts(CPU))
    assert_true(_accepts(ACCELERATOR))
    assert_true(_accepts(CUDA))
    assert_true(_accepts(HIP))
    assert_true(_accepts(METAL))


def test_chains() raises:
    assert_true(_accepts(CUDA & arch("sm_90a")))
    assert_true(_accepts(CUDA & arch("sm_100a") & model("NVIDIA B200")))
    assert_true(_accepts(CPU & api("cpu")))
    # Repeating an upper level that a helper already states is allowed.
    assert_true(_accepts(HOPPER))
    assert_true(_accepts(B200))
    assert_true(_accepts(ACCELERATOR & api("tpu") & arch("tpu-v6e")))


def test_unions() raises:
    assert_true(_accepts(CUDA | HIP | METAL))
    assert_true(_accepts(CUDA & (arch("sm_100a") | arch("sm_103a"))))
    assert_true(_accepts(SM100_FAMILY & model("NVIDIA B200")))
    assert_true(_accepts((CUDA | HIP) & arch("sm_90a")))
    assert_true(_accepts(SM100 & (model("NVIDIA B200") | model("X"))))
    # A union of different levels can be registered.
    assert_true(_accepts(CPU | CUDA))
    assert_true(_accepts(HOPPER | B200))
    assert_true(_accepts(CPU | (CUDA & arch("sm_100a"))))


def test_either_order() raises:
    assert_true(_accepts(arch("sm_90a") & CUDA))
    assert_true(_accepts(api("cuda") & ACCELERATOR))
    assert_true(_accepts(model("NVIDIA B200") & SM100))
    assert_true(
        _accepts(
            model("NVIDIA B200") & arch("sm_100a") & api("cuda") & ACCELERATOR
        )
    )
    assert_true(_accepts((arch("sm_90a") & api("cuda")) & ACCELERATOR))
    # `&` and `|` order their operands, so both spellings have one type.
    var cuda_or_hip: type_of(CUDA | HIP) = HIP | CUDA
    var hopper: type_of(CUDA & arch("sm_90a")) = arch("sm_90a") & CUDA
    _ = cuda_or_hip
    _ = hopper


def test_intersections() raises:
    assert_true(_accepts(TMA & SM100_FAMILY))
    assert_true(_accepts((TMA & SM100_FAMILY) & model("NVIDIA B200")))
    assert_true(
        _accepts(
            (SM100_FAMILY & (CUDA & (arch("sm_103a") | arch("sm_110a"))))
            & model("X")
        )
    )
    # Contradictory conditions are valid Mojo; the graph compiler rejects a
    # predicate that holds for no device when it imports the package.
    assert_true(_accepts(CUDA & HIP))


def test_field_tests_that_wait_for_a_parent() raises:
    assert_true(
        _accepts(ACCELERATOR & (api("hip") | (api("cuda") & arch("sm_90a"))))
    )
    assert_true(
        _accepts(
            CUDA & (arch("sm_90a") | (arch("sm_100a") & model("NVIDIA B200")))
        )
    )
    assert_true(
        _accepts(
            ACCELERATOR & (api("cuda") & (arch("sm_90a") | arch("sm_100a")))
        )
    )
    assert_true(
        _accepts(
            ACCELERATOR & (api("cuda") & arch("sm_100a") & model("NVIDIA B200"))
        )
    )


def test_levels() raises:
    comptime assert type_of(CPU).top == 0
    comptime assert type_of(CPU).low == 0 and type_of(CPU).high == 0
    comptime assert type_of(CUDA).top == 0
    comptime assert type_of(CUDA).low == 1 and type_of(CUDA).high == 1
    comptime assert type_of(B200).low == 3 and type_of(B200).high == 3
    # A union whose branches end at different levels.
    comptime assert type_of(CUDA | HOPPER).low == 1
    comptime assert type_of(CUDA | HOPPER).high == 2
    # Field tests without their parent.
    comptime assert type_of(api("cuda")).top == 1
    comptime assert type_of(arch("sm_90a")).top == 2
    comptime assert type_of(model("NVIDIA B200")).top == 3
    comptime assert type_of(api("cuda") & arch("sm_90a")).top == 1
    comptime assert type_of(api("cuda") & arch("sm_90a")).high == 2


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
