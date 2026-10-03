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

# Verifies that the host-side accelerator-vendor checks agree with the
# `--target-accelerator` value, regardless of whether it is given as a bare
# architecture ("gfx950", "sm_90") or a vendor-prefixed form
# ("amdgpu:gfx950", "amd:gfx950", "nvidia:sm_90"). A bare target must behave
# identically to its vendor-prefixed form.
#
# These are compile-only checks: `main()` contains no device code, so every
# `comptime assert` is evaluated on the host and the build succeeds only when
# the expected vendor is detected. `--emit object` stops before linking: the
# vendor detection is a compile-time property, and linking a cross-compiled
# (for example arm64-apple-darwin) executable on the x86 Linux test host fails
# at `ld.lld`, unrelated to what is under test.

# AMD: canonical values, finite model aliases, and vendor prefixes must all be
# AMD. Prefix patterns are tested separately because they are not finite values
# in the target-list synchronization test.
# RUN: %mojo-build --emit object --target-accelerator gfx950          -D EXPECT=amd_default_accelerator %s -o %t
# RUN: %mojo-build --emit object --target-accelerator mi250x          -D EXPECT=amd %s -o %t
# RUN: %mojo-build --emit object --target-accelerator mi300x          -D EXPECT=amd %s -o %t
# RUN: %mojo-build --emit object --target-accelerator mi355x          -D EXPECT=amd %s -o %t
# RUN: %mojo-build --emit object --target-accelerator amdgpu:gfx950   -D EXPECT=amd %s -o %t
# RUN: %mojo-build --emit object --target-accelerator amd:gfx950      -D EXPECT=amd %s -o %t
# RUN: %mojo-build --emit object --target-accelerator amdgpu:mi300x   -D EXPECT=amd %s -o %t
# RUN: %mojo-build --emit object --target-accelerator amd:mi355x      -D EXPECT=amd %s -o %t

# NVIDIA: bare arch and vendor prefix, must all be NVIDIA.
# RUN: %mojo-build --emit object --target-accelerator sm_90        -D EXPECT=nvidia %s -o %t
# RUN: %mojo-build --emit object --target-accelerator nvidia:sm_90 -D EXPECT=nvidia %s -o %t

# Apple: bare arch and "metal:" prefix must both be Apple.
# RUN: %mojo-build --emit object --target-triple arm64-apple-darwin \
# RUN:   --target-accelerator apple-m4 -D EXPECT=apple %s -o %t
# RUN: %mojo-build --emit object --target-triple arm64-apple-darwin \
# RUN:   --target-accelerator metal:4  -D EXPECT=apple %s -o %t

# Unknown/unrecognized accelerator: none of the vendor predicates fire (the
# False-on-unknown contract). "wombat42" contains none of the vendor-relevant
# substrings, so it also guards against loose-substring false positives.
# RUN: not %mojo-build --emit object --target-accelerator wombat42 -D EXPECT=error_has_functions %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-has_functions
# RUN: not %mojo-build --emit object --target-accelerator wombat42 -D EXPECT=error_gpu_info_from_name %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-GPUInfo_from_name
# RUN: not %mojo-build --emit object --target-accelerator wombat42 -D EXPECT=error_default_accelerator %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-INVALID-default_accelerator

# `current_accelerator()` is only valid when the current compilation target is an
# accelerator. Calling it from host code must fail with a message that names the
# actual target and points at `default_accelerator()`, while the same call inside
# an offload compilation must succeed.
# RUN: not %mojo-build --emit object -D EXPECT=error_current_accelerator_on_host %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-NON-ACCELERATOR
# RUN:     %mojo-build --emit object -D EXPECT=current_accelerator_in_offload --target-accelerator sm_90 %s -o %t

# `default_accelerator()` must agree with `--target-accelerator` even when called
# from an offload compilation. Offloading to an explicit target that matches the
# default succeeds, but offloading to a different explicit target must fail
# instead of silently resolving host and device code against different
# accelerators.
# RUN:     %mojo-build --emit object -D EXPECT=default_accelerator_in_offload_match --target-accelerator sm_90 %s -o %t
# RUN: not %mojo-build --emit object -D EXPECT=error_default_accelerator_in_offload_mismatch --target-accelerator sm_90 %s -o %t 2>&1 | FileCheck %s --check-prefix=CHECK-MISMATCH

from std.compile import compile_info
from std.sys import (
    get_defined_string,
    has_amd_gpu_accelerator,
    has_apple_gpu_accelerator,
    has_nvidia_gpu_accelerator,
)
from std.sys.info import current_accelerator, default_accelerator
from std._gpu.host.info import GPUInfo, TargetAccelerator, get_gpu_target


def main() raises:
    comptime expect = get_defined_string["EXPECT"]()

    comptime if expect == "amd" or expect == "amd_default_accelerator":
        comptime assert has_amd_gpu_accelerator()
        comptime assert not has_nvidia_gpu_accelerator()
        comptime assert not has_apple_gpu_accelerator()
        comptime if expect == "amd_default_accelerator":
            comptime target = TargetAccelerator.default_accelerator
            comptime assert target.gpu_info == GPUInfo.default_accelerator()
    elif expect == "nvidia":
        comptime assert has_nvidia_gpu_accelerator()
        comptime assert not has_amd_gpu_accelerator()
        comptime assert not has_apple_gpu_accelerator()
    elif expect == "apple":
        comptime assert has_apple_gpu_accelerator()
        comptime assert not has_amd_gpu_accelerator()
        comptime assert not has_nvidia_gpu_accelerator()
    elif expect == "error_has_functions":
        # CHECK-INVALID-has_functions: GPU architecture 'wombat42' is not supported.
        comptime assert not has_amd_gpu_accelerator()
        comptime assert not has_nvidia_gpu_accelerator()
        comptime assert not has_apple_gpu_accelerator()
    elif expect == "error_gpu_info_from_name":
        # CHECK-INVALID-GPUInfo_from_name: GPU architecture 'wombat42' is not supported.
        comptime info = GPUInfo.from_name["wombat42"]()
        comptime assert info.name == "wombat42"
    elif expect == "error_default_accelerator":
        # CHECK-INVALID-default_accelerator: GPU architecture 'wombat42' is not supported.
        # Force resolution of `.default_accelerator` to trigger error.
        comptime assert (
            TargetAccelerator.default_accelerator.gpu_info.name != ""
        )
    elif expect == "error_current_accelerator_on_host":
        # CHECK-NON-ACCELERATOR: current_accelerator() requires an accelerator compilation target, but the current compilation target is '
        # CHECK-NON-ACCELERATOR-SAME: This function is only valid in code compiled for an accelerator
        # CHECK-NON-ACCELERATOR-SAME: use `default_accelerator()` instead.
        _ = current_accelerator()
    elif expect == "current_accelerator_in_offload":

        def _is_nvidia_in_offload() -> Bool:
            return current_accelerator().is_nvidia_gpu()

        _ = compile_info[
            _is_nvidia_in_offload,
            emission_kind="llvm",
            target=default_accelerator(),
        ]()
    elif expect == "default_accelerator_in_offload_match":
        _ = compile_info[
            lambda () -> Bool: default_accelerator().is_nvidia_gpu(),
            emission_kind="llvm",
            target=get_gpu_target["sm_90"](),
        ]()
    elif expect == "error_default_accelerator_in_offload_mismatch":
        # CHECK-MISMATCH: default_accelerator() called while compiling for accelerator target '
        # CHECK-MISMATCH-SAME: (arch: 'gfx950')
        # CHECK-MISMATCH-SAME: which is not the default accelerator
        # CHECK-MISMATCH-SAME: (arch: 'sm_90a')

        # The explicit AMD target differs from `--target-accelerator sm_90`.
        _ = compile_info[
            lambda () -> Bool: default_accelerator().is_nvidia_gpu(),
            emission_kind="llvm",
            target=get_gpu_target["gfx950"](),
        ]()
    else:
        comptime assert False, "unknown EXPECT value"
