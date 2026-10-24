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

# RUN: not kgen -emit=llvm %s 2>&1 | FileCheck %s
# RUN: not kgen -emit=llvm %s --target-triple=nvptx64-nvidia-cuda --target-cpu=sm_89 --target-features="" 2>&1 | FileCheck %s


@export
def cast_from_float8_e3m4(
    dst: Pointer[Float32, MutAnyOrigin], src: Scalar[.float8_e3m4]
) abi("Mojo"):
    # CHECK: error: conversion from 'f8e3m4' to 'f32' is not implemented
    dst[0] = src.cast[.float32]()


@export
def add_from_float8_e3m4(
    dst: Pointer[Scalar[.float8_e3m4], MutAnyOrigin],
    src: Scalar[.float8_e3m4],
) abi("Mojo"):
    # CHECK: error: operation with a type 'f8e3m4' is not implemented
    dst[0] += src
