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

# RUN: %parse-mojo-isolated %s | FileCheck %s

# Inferring a parameter through an implicit conversion must keep the
# dependencies between the parameters of the expected type. In
# `DepFoo[3, dt, 0]`, the bound value `0` has type `Scalar[dt]`, so it refers
# to the parameter `dt` that the conversion infers.


struct DepFoo[n: Int, dt: DType, v: Scalar[dt]](ImplicitlyCopyable):
    @implicit
    def __init__(out self, ref x: List[Scalar[Self.dt]]):
        pass


def take_dep[dt: DType](x: DepFoo[3, dt, 0]):
    pass


# CHECK-LABEL: lit.fn @"dependent_param_type_from_value()"
def dependent_param_type_from_value():
    var l = List[Float32]()
    # CHECK: lit.call @{{.*}}@"take_dep{{.*}}"
    take_dep(l)


