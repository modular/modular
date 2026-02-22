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
"""Tests for Stringable and Representable implementations on GPU types."""

from gpu.memory import CacheEviction, CacheOperation, Consistency, Fill, ReduceOp
from gpu.host.info import Vendor
from gpu.host.launch_attribute import AccessProperty, LaunchAttributeID
from gpu.sync import AMDScheduleBarrierMask

from testing import assert_equal, TestSuite


# ===----------------------------------------------------------------------=== #
# Helpers
# ===----------------------------------------------------------------------=== #


fn _test_repr[T: Representable & Writable](value: T, expected: String) raises:
    assert_equal(value.__repr__(), expected)
    var string = String()
    value.write_repr_to(string)
    assert_equal(string, expected)


# ===----------------------------------------------------------------------=== #
# CacheOperation
# ===----------------------------------------------------------------------=== #


def test_cache_operation_str():
    assert_equal(String(CacheOperation.ALWAYS), "ca")
    assert_equal(String(CacheOperation.GLOBAL), "cg")
    assert_equal(String(CacheOperation.STREAMING), "cs")
    assert_equal(String(CacheOperation.LAST_USE), "lu")
    assert_equal(String(CacheOperation.VOLATILE), "cv")
    assert_equal(String(CacheOperation.WRITE_BACK), "wb")
    assert_equal(String(CacheOperation.WRITE_THROUGH), "wt")
    assert_equal(String(CacheOperation.WORKGROUP), "wg")


def test_cache_operation_repr():
    _test_repr(CacheOperation.ALWAYS, "CacheOperation.ca")
    _test_repr(CacheOperation.GLOBAL, "CacheOperation.cg")
    _test_repr(CacheOperation.STREAMING, "CacheOperation.cs")
    _test_repr(CacheOperation.WRITE_BACK, "CacheOperation.wb")


# ===----------------------------------------------------------------------=== #
# CacheEviction
# ===----------------------------------------------------------------------=== #


def test_cache_eviction_str():
    assert_equal(String(CacheEviction.EVICT_NORMAL), "evict_normal")
    assert_equal(String(CacheEviction.EVICT_FIRST), "evict_first")
    assert_equal(String(CacheEviction.EVICT_LAST), "evict_last")
    assert_equal(String(CacheEviction.EVICT_UNCHANGED), "evict_unchanged")
    assert_equal(String(CacheEviction.NO_ALLOCATE), "no_allocate")


def test_cache_eviction_repr():
    _test_repr(CacheEviction.EVICT_NORMAL, "CacheEviction.evict_normal")
    _test_repr(CacheEviction.EVICT_FIRST, "CacheEviction.evict_first")
    _test_repr(CacheEviction.NO_ALLOCATE, "CacheEviction.no_allocate")


# ===----------------------------------------------------------------------=== #
# Fill
# ===----------------------------------------------------------------------=== #


def test_fill_str():
    assert_equal(String(Fill.NONE), "none")
    assert_equal(String(Fill.ZERO), "zero")
    assert_equal(String(Fill.NAN), "nan")


def test_fill_repr():
    _test_repr(Fill.NONE, "Fill.none")
    _test_repr(Fill.ZERO, "Fill.zero")
    _test_repr(Fill.NAN, "Fill.nan")


# ===----------------------------------------------------------------------=== #
# Consistency
# ===----------------------------------------------------------------------=== #


def test_consistency_str():
    assert_equal(String(Consistency.WEAK), "weak")
    assert_equal(String(Consistency.RELAXED), "relaxed")
    assert_equal(String(Consistency.ACQUIRE), "acquire")
    assert_equal(String(Consistency.RELEASE), "release")


def test_consistency_repr():
    _test_repr(Consistency.WEAK, "Consistency.weak")
    _test_repr(Consistency.RELAXED, "Consistency.relaxed")
    _test_repr(Consistency.ACQUIRE, "Consistency.acquire")
    _test_repr(Consistency.RELEASE, "Consistency.release")


# ===----------------------------------------------------------------------=== #
# ReduceOp
# ===----------------------------------------------------------------------=== #


def test_reduce_op_str():
    assert_equal(String(ReduceOp.ADD), "add")
    assert_equal(String(ReduceOp.MIN), "min")
    assert_equal(String(ReduceOp.MAX), "max")
    assert_equal(String(ReduceOp.AND), "and")
    assert_equal(String(ReduceOp.OR), "or")
    assert_equal(String(ReduceOp.XOR), "xor")


def test_reduce_op_repr():
    _test_repr(ReduceOp.ADD, "ReduceOp.add")
    _test_repr(ReduceOp.MIN, "ReduceOp.min")
    _test_repr(ReduceOp.XOR, "ReduceOp.xor")


# ===----------------------------------------------------------------------=== #
# Vendor
# ===----------------------------------------------------------------------=== #


def test_vendor_str():
    assert_equal(String(Vendor.NO_GPU), "no_gpu")
    assert_equal(String(Vendor.AMD_GPU), "amd_gpu")
    assert_equal(String(Vendor.NVIDIA_GPU), "nvidia_gpu")
    assert_equal(String(Vendor.APPLE_GPU), "apple_gpu")


def test_vendor_repr():
    _test_repr(Vendor.NO_GPU, "Vendor.no_gpu")
    _test_repr(Vendor.AMD_GPU, "Vendor.amd_gpu")
    _test_repr(Vendor.NVIDIA_GPU, "Vendor.nvidia_gpu")
    _test_repr(Vendor.APPLE_GPU, "Vendor.apple_gpu")


# ===----------------------------------------------------------------------=== #
# LaunchAttributeID
# ===----------------------------------------------------------------------=== #


def test_launch_attribute_id_str():
    assert_equal(String(LaunchAttributeID.IGNORE), "0")
    assert_equal(String(LaunchAttributeID.ACCESS_POLICY_WINDOW), "1")
    assert_equal(String(LaunchAttributeID.COOPERATIVE), "2")
    assert_equal(String(LaunchAttributeID.PRIORITY), "8")


def test_launch_attribute_id_repr():
    _test_repr(LaunchAttributeID.IGNORE, "LaunchAttributeID.0")
    _test_repr(LaunchAttributeID.ACCESS_POLICY_WINDOW, "LaunchAttributeID.1")
    _test_repr(LaunchAttributeID.COOPERATIVE, "LaunchAttributeID.2")


# ===----------------------------------------------------------------------=== #
# AccessProperty
# ===----------------------------------------------------------------------=== #


def test_access_property_str():
    assert_equal(String(AccessProperty.NORMAL), "NORMAL")
    assert_equal(String(AccessProperty.STREAMING), "STREAMING")
    assert_equal(String(AccessProperty.PERSISTING), "PERSISTING")


def test_access_property_repr():
    _test_repr(AccessProperty.NORMAL, "AccessProperty.NORMAL")
    _test_repr(AccessProperty.STREAMING, "AccessProperty.STREAMING")
    _test_repr(AccessProperty.PERSISTING, "AccessProperty.PERSISTING")


# ===----------------------------------------------------------------------=== #
# AMDScheduleBarrierMask
# ===----------------------------------------------------------------------=== #


def test_amd_schedule_barrier_mask_str():
    assert_equal(String(AMDScheduleBarrierMask.NONE), "NONE")
    assert_equal(String(AMDScheduleBarrierMask.ALL_ALU), "ALL_ALU")
    assert_equal(String(AMDScheduleBarrierMask.VALU), "VALU")
    assert_equal(String(AMDScheduleBarrierMask.SALU), "SALU")
    assert_equal(String(AMDScheduleBarrierMask.MFMA), "MFMA")
    assert_equal(String(AMDScheduleBarrierMask.ALL_VMEM), "ALL_VMEM")
    assert_equal(String(AMDScheduleBarrierMask.VMEM_READ), "VMEM_READ")
    assert_equal(String(AMDScheduleBarrierMask.VMEM_WRITE), "VMEM_WRITE")
    assert_equal(String(AMDScheduleBarrierMask.ALL_DS), "ALL_DS")
    assert_equal(String(AMDScheduleBarrierMask.DS_READ), "DS_READ")
    assert_equal(String(AMDScheduleBarrierMask.DS_WRITE), "DS_WRITE")
    assert_equal(String(AMDScheduleBarrierMask.TRANS), "TRANS")


def test_amd_schedule_barrier_mask_repr():
    _test_repr(AMDScheduleBarrierMask.NONE, "AMDScheduleBarrierMask.NONE")
    _test_repr(AMDScheduleBarrierMask.ALL_ALU, "AMDScheduleBarrierMask.ALL_ALU")
    _test_repr(AMDScheduleBarrierMask.MFMA, "AMDScheduleBarrierMask.MFMA")
    _test_repr(AMDScheduleBarrierMask.TRANS, "AMDScheduleBarrierMask.TRANS")


# ===----------------------------------------------------------------------=== #
# Main
# ===----------------------------------------------------------------------=== #


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
