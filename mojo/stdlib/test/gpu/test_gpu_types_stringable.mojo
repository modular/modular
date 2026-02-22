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
"""Tests for Writable and Representable implementations on GPU types."""

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


def test_cache_operation_write_to():
    assert_equal(String.write(CacheOperation.ALWAYS), "ca")
    assert_equal(String.write(CacheOperation.GLOBAL), "cg")
    assert_equal(String.write(CacheOperation.STREAMING), "cs")
    assert_equal(String.write(CacheOperation.LAST_USE), "lu")
    assert_equal(String.write(CacheOperation.VOLATILE), "cv")
    assert_equal(String.write(CacheOperation.WRITE_BACK), "wb")
    assert_equal(String.write(CacheOperation.WRITE_THROUGH), "wt")
    assert_equal(String.write(CacheOperation.WORKGROUP), "wg")


def test_cache_operation_repr():
    _test_repr(CacheOperation.ALWAYS, "CacheOperation(ca)")
    _test_repr(CacheOperation.GLOBAL, "CacheOperation(cg)")
    _test_repr(CacheOperation.STREAMING, "CacheOperation(cs)")
    _test_repr(CacheOperation.WRITE_BACK, "CacheOperation(wb)")


# ===----------------------------------------------------------------------=== #
# CacheEviction
# ===----------------------------------------------------------------------=== #


def test_cache_eviction_write_to():
    assert_equal(String.write(CacheEviction.EVICT_NORMAL), "evict_normal")
    assert_equal(String.write(CacheEviction.EVICT_FIRST), "evict_first")
    assert_equal(String.write(CacheEviction.EVICT_LAST), "evict_last")
    assert_equal(String.write(CacheEviction.EVICT_UNCHANGED), "evict_unchanged")
    assert_equal(String.write(CacheEviction.NO_ALLOCATE), "no_allocate")


def test_cache_eviction_repr():
    _test_repr(CacheEviction.EVICT_NORMAL, "CacheEviction(evict_normal)")
    _test_repr(CacheEviction.EVICT_FIRST, "CacheEviction(evict_first)")
    _test_repr(CacheEviction.NO_ALLOCATE, "CacheEviction(no_allocate)")


# ===----------------------------------------------------------------------=== #
# Fill
# ===----------------------------------------------------------------------=== #


def test_fill_str():
    assert_equal(Fill.NONE.__str__(), "none")
    assert_equal(Fill.ZERO.__str__(), "zero")
    assert_equal(Fill.NAN.__str__(), "nan")


def test_fill_repr():
    _test_repr(Fill.NONE, "Fill(none)")
    _test_repr(Fill.ZERO, "Fill(zero)")
    _test_repr(Fill.NAN, "Fill(nan)")


# ===----------------------------------------------------------------------=== #
# Consistency
# ===----------------------------------------------------------------------=== #


def test_consistency_str():
    assert_equal(Consistency.WEAK.__str__(), "weak")
    assert_equal(Consistency.RELAXED.__str__(), "relaxed")
    assert_equal(Consistency.ACQUIRE.__str__(), "acquire")
    assert_equal(Consistency.RELEASE.__str__(), "release")


def test_consistency_repr():
    _test_repr(Consistency.WEAK, "Consistency(weak)")
    _test_repr(Consistency.RELAXED, "Consistency(relaxed)")
    _test_repr(Consistency.ACQUIRE, "Consistency(acquire)")
    _test_repr(Consistency.RELEASE, "Consistency(release)")


# ===----------------------------------------------------------------------=== #
# ReduceOp
# ===----------------------------------------------------------------------=== #


def test_reduce_op_str():
    assert_equal(ReduceOp.ADD.__str__(), "add")
    assert_equal(ReduceOp.MIN.__str__(), "min")
    assert_equal(ReduceOp.MAX.__str__(), "max")
    assert_equal(ReduceOp.AND.__str__(), "and")
    assert_equal(ReduceOp.OR.__str__(), "or")
    assert_equal(ReduceOp.XOR.__str__(), "xor")


def test_reduce_op_repr():
    _test_repr(ReduceOp.ADD, "ReduceOp(add)")
    _test_repr(ReduceOp.MIN, "ReduceOp(min)")
    _test_repr(ReduceOp.XOR, "ReduceOp(xor)")


# ===----------------------------------------------------------------------=== #
# Vendor
# ===----------------------------------------------------------------------=== #


def test_vendor_str():
    assert_equal(Vendor.NO_GPU.__str__(), "no_gpu")
    assert_equal(Vendor.AMD_GPU.__str__(), "amd_gpu")
    assert_equal(Vendor.NVIDIA_GPU.__str__(), "nvidia_gpu")
    assert_equal(Vendor.APPLE_GPU.__str__(), "apple_gpu")


def test_vendor_repr():
    _test_repr(Vendor.NO_GPU, "Vendor(no_gpu)")
    _test_repr(Vendor.AMD_GPU, "Vendor(amd_gpu)")
    _test_repr(Vendor.NVIDIA_GPU, "Vendor(nvidia_gpu)")
    _test_repr(Vendor.APPLE_GPU, "Vendor(apple_gpu)")


# ===----------------------------------------------------------------------=== #
# LaunchAttributeID
# ===----------------------------------------------------------------------=== #


def test_launch_attribute_id_str():
    assert_equal(LaunchAttributeID.IGNORE.__str__(), "0")
    assert_equal(LaunchAttributeID.ACCESS_POLICY_WINDOW.__str__(), "1")
    assert_equal(LaunchAttributeID.COOPERATIVE.__str__(), "2")
    assert_equal(LaunchAttributeID.PRIORITY.__str__(), "8")


def test_launch_attribute_id_repr():
    _test_repr(LaunchAttributeID.IGNORE, "LaunchAttributeID(0)")
    _test_repr(LaunchAttributeID.ACCESS_POLICY_WINDOW, "LaunchAttributeID(1)")
    _test_repr(LaunchAttributeID.COOPERATIVE, "LaunchAttributeID(2)")


# ===----------------------------------------------------------------------=== #
# AccessProperty
# ===----------------------------------------------------------------------=== #


def test_access_property_str():
    assert_equal(AccessProperty.NORMAL.__str__(), "NORMAL")
    assert_equal(AccessProperty.STREAMING.__str__(), "STREAMING")
    assert_equal(AccessProperty.PERSISTING.__str__(), "PERSISTING")


def test_access_property_repr():
    _test_repr(AccessProperty.NORMAL, "AccessProperty(NORMAL)")
    _test_repr(AccessProperty.STREAMING, "AccessProperty(STREAMING)")
    _test_repr(AccessProperty.PERSISTING, "AccessProperty(PERSISTING)")


# ===----------------------------------------------------------------------=== #
# AMDScheduleBarrierMask
# ===----------------------------------------------------------------------=== #


def test_amd_schedule_barrier_mask_str():
    assert_equal(AMDScheduleBarrierMask.NONE.__str__(), "NONE")
    assert_equal(AMDScheduleBarrierMask.ALL_ALU.__str__(), "ALL_ALU")
    assert_equal(AMDScheduleBarrierMask.VALU.__str__(), "VALU")
    assert_equal(AMDScheduleBarrierMask.SALU.__str__(), "SALU")
    assert_equal(AMDScheduleBarrierMask.MFMA.__str__(), "MFMA")
    assert_equal(AMDScheduleBarrierMask.ALL_VMEM.__str__(), "ALL_VMEM")
    assert_equal(AMDScheduleBarrierMask.VMEM_READ.__str__(), "VMEM_READ")
    assert_equal(AMDScheduleBarrierMask.VMEM_WRITE.__str__(), "VMEM_WRITE")
    assert_equal(AMDScheduleBarrierMask.ALL_DS.__str__(), "ALL_DS")
    assert_equal(AMDScheduleBarrierMask.DS_READ.__str__(), "DS_READ")
    assert_equal(AMDScheduleBarrierMask.DS_WRITE.__str__(), "DS_WRITE")
    assert_equal(AMDScheduleBarrierMask.TRANS.__str__(), "TRANS")


def test_amd_schedule_barrier_mask_repr():
    _test_repr(AMDScheduleBarrierMask.NONE, "AMDScheduleBarrierMask(NONE)")
    _test_repr(AMDScheduleBarrierMask.ALL_ALU, "AMDScheduleBarrierMask(ALL_ALU)")
    _test_repr(AMDScheduleBarrierMask.MFMA, "AMDScheduleBarrierMask(MFMA)")
    _test_repr(AMDScheduleBarrierMask.TRANS, "AMDScheduleBarrierMask(TRANS)")


# ===----------------------------------------------------------------------=== #
# Main
# ===----------------------------------------------------------------------=== #


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
