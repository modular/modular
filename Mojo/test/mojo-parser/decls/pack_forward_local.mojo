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

# RUN: %parse-mojo-isolated -verify-diagnostics %s
# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-diagnostics

# A local VariadicPack that does not have exactly one store is forwarded as a
# whole when it is unpacked.
# expected-no-diagnostics


def takes_pack[*Ts: AnyType](*pack: *Ts):
    pass


def copy_pack[*Ts: AnyType](*pack: *Ts) -> type_of(pack):
    return type_of(pack)(pack._value)


def make_pack[*Ts: AnyType](*pack: *Ts, out result: type_of(pack)):
    result = type_of(pack)(pack._value)


def forward_reassigned_local[*Ts: AnyType](*pack: *Ts):
    var local = copy_pack(*pack)
    takes_pack(*local)
    local = copy_pack(*pack)
    takes_pack(*local)


def forward_out_initialized_local[*Ts: AnyType](*pack: *Ts):
    var local = make_pack(*pack)
    takes_pack(*local)


def forward_ctor_from_reassigned_raw[*Ts: AnyType](*pack: *Ts):
    var v = pack._value
    _ = v
    v = pack._value
    var p = type_of(pack)(v)
    takes_pack(*p)
