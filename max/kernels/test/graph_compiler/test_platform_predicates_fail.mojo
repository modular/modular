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

# RUN: not %mojo-no-debug %s 2>&1 | FileCheck %s
#
# Mojo must reject each platform predicate below where it is written. Each
# check names the line of the mistake and the message of the rule that it
# breaks, so an unrelated compile error can't pass as a rejection.

import extensibility
from extensibility import (
    ACCELERATOR,
    CUDA,
    HIP,
    PlatformExpr,
    PlatformPredicate,
    api,
    arch,
    model,
)

comptime HOPPER = ACCELERATOR & CUDA & arch("sm_90a")
comptime SM100 = ACCELERATOR & CUDA & arch("sm_100a")
comptime B200 = SM100 & model("NVIDIA B200")


# A field test without its parent is not a condition.
# CHECK: test_platform_predicates_fail.mojo:[[@LINE+3]]:{{[0-9]+}}: error: no matching function in call to 'register'
# CHECK: note: candidate not viable: value passed to 'cond' cannot be converted from '_Platform
# CHECK-SAME: to 'PlatformPredicate[expr]'
@extensibility.register("pred_bare_arch", arch("sm_90a"))
struct BareArch:
    pass


def _accepts[expr: PlatformExpr](cond: PlatformPredicate[expr]) -> Bool:
    return True


def main():
    # CHECK: test_platform_predicates_fail.mojo:[[@LINE+2]]:{{[0-9]+}}: error: {{.*}}cannot be converted from '_Platform
    # CHECK-SAME: to 'PlatformPredicate[expr]'
    _ = _accepts(model("NVIDIA B200"))

    # A field test under a parent at the wrong level.
    # CHECK: test_platform_predicates_fail.mojo:[[@LINE+2]]:{{[0-9]+}}: error: invalid call to '__and__'
    # CHECK: arch() narrows an API-level condition: CUDA & arch("sm_90a")
    _ = _accepts(ACCELERATOR & arch("sm_90a"))
    # CHECK: test_platform_predicates_fail.mojo:[[@LINE+2]]:{{[0-9]+}}: error: invalid call to '__and__'
    # CHECK: arch() narrows an API-level condition: CUDA & arch("sm_90a")
    _ = _accepts(HOPPER & arch("sm_100a"))
    # CHECK: test_platform_predicates_fail.mojo:[[@LINE+2]]:{{[0-9]+}}: error: invalid call to '__and__'
    # CHECK: arch() narrows an API-level condition: CUDA & arch("sm_90a")
    _ = _accepts(arch("sm_90a") & ACCELERATOR)
    # CHECK: test_platform_predicates_fail.mojo:[[@LINE+2]]:{{[0-9]+}}: error: invalid call to '__and__'
    # CHECK: api() narrows a type-level condition: ACCELERATOR & api("cuda")
    _ = _accepts(CUDA & api("hip"))
    # CHECK: test_platform_predicates_fail.mojo:[[@LINE+2]]:{{[0-9]+}}: error: invalid call to '__and__'
    # CHECK: model() narrows an arch-level condition: SM100 & model("NVIDIA B200")
    _ = _accepts(CUDA & model("NVIDIA B200"))
    # CHECK: test_platform_predicates_fail.mojo:[[@LINE+2]]:{{[0-9]+}}: error: invalid call to '__and__'
    # CHECK: model() narrows an arch-level condition: SM100 & model("NVIDIA B200")
    _ = _accepts(B200 & model("NVIDIA GB200"))

    # A field test under a union whose branches end at different levels.
    # CHECK: test_platform_predicates_fail.mojo:[[@LINE+2]]:{{[0-9]+}}: error: invalid call to '__and__'
    # CHECK: add the field test to each branch of this union
    _ = _accepts((CUDA | HOPPER) & model("NVIDIA B200"))
    # CHECK: test_platform_predicates_fail.mojo:[[@LINE+4]]:{{[0-9]+}}: error: invalid call to '__and__'
    # CHECK: add the field test to each branch of this union
    _ = _accepts(
        ACCELERATOR
        & ((api("hip") | (api("cuda") & arch("sm_90a"))) & model("X"))
    )

    # Field tests that skip a level or test a level twice.
    # CHECK: test_platform_predicates_fail.mojo:[[@LINE+2]]:{{[0-9]+}}: error: invalid call to '__and__'
    # CHECK: field tests chain one level at a time: api("cuda") & arch("sm_90a")
    _ = _accepts(ACCELERATOR & (api("cuda") & model("NVIDIA B200")))
    # CHECK: test_platform_predicates_fail.mojo:[[@LINE+3]]:{{[0-9]+}}: error: invalid call to '__and__'
    # CHECK: field tests chain one level at a time: api("cuda") & arch("sm_90a")
    _ = _accepts(
        ACCELERATOR & ((api("cuda") & arch("sm_90a")) & arch("sm_100a"))
    )

    # `|` between a condition and a field test, or between field tests of
    # different levels.
    # CHECK: test_platform_predicates_fail.mojo:[[@LINE+2]]:{{[0-9]+}}: error: invalid call to '__or__'
    # CHECK: | joins two conditions, or two field tests of the same level
    _ = _accepts(HIP | arch("sm_90a"))
    # CHECK: test_platform_predicates_fail.mojo:[[@LINE+2]]:{{[0-9]+}}: error: invalid call to '__or__'
    # CHECK: | joins two conditions, or two field tests of the same level
    _ = _accepts(ACCELERATOR & (api("hip") | arch("sm_90a")))
