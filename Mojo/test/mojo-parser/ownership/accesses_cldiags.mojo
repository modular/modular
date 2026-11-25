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

# RUN: %parse-mojo-isolated %s -mlir-print-debuginfo | kgen-opt -lower-semantic-cf -check-lifetimes -verify-parameters -verify-diagnostics

def mutate_through_pointer[origin: MutOrigin](ptr: Pointer[Int, origin]):
    ptr[] = 0

def observe_ref(ref value: Int):
    _ = value

# ===----------------------------------------------------------------------=== #
# Accesses to nested origins are affected by @__nested_origins_read_only.

@__nested_origins_read_only
def use_pointer[origin: MutOrigin](ptr: Pointer[Int, origin]):
    # expected-error @below {{function marked '__nested_origins_read_only' may not mutably access origin 'origin'}}
    mutate_through_pointer(ptr)

    # expected-error @below {{function marked '__nested_origins_read_only' may not mutably access origin 'origin'}}
    ptr[] = 0


def mutate_capture():
    var value = 0
    @__nested_origins_read_only
    def inner() {mut value}:  # Perfectly fine to mutate a mut-captured value.
        value = 1
    inner()


def mutate_through_captured_pointer[origin: MutOrigin](
    ptr: Pointer[Int, origin],
):
    @__nested_origins_read_only
    def inner() {ptr}:
        # expected-error @below {{function marked '__nested_origins_read_only' may not mutably access origin 'origin'}}
        ptr[] = 1

    inner()

# ===----------------------------------------------------------------------=== #
# Accesses to mut and unconditionally mutable arguments are not affected by
# @__nested_origins_read_only.

@__nested_origins_read_only
def use_mut_argument(mut value: Int):
    _ = value
    value = 0

@__nested_origins_read_only
def use_mut_ref_parameter[origin: Origin[mut=True]](ref[origin] value: Int):
    _ = value
    value = 0

@__nested_origins_read_only
def mutate_owned_parameter(var value: Int) -> Int:
    # Owned parameters are local values, so mutating them does not access the
    # caller's origin.
    value = 0
    return value

@__nested_origins_read_only
def mutate_local_pointer():
    var value = 0
    var ptr = Pointer(to=value)
    ptr[] = 1

struct NestedOriginsReadOnlyReceiver:
    var value: Int

    @__nested_origins_read_only
    def mutate(mut self):
        self.value = 0

@__nested_origins_read_only
def mutate_parameter_through_alias(mut value: Int):
    ref alias = value
    alias = 0


@__nested_origins_read_only
def consume_mut_parameter(mut value: String):
    var consumed = value^
    _ = consumed
    abort()

# ===----------------------------------------------------------------------=== #
# 'ref' with parametric mutability is always read-only considered a read.

# The `ref` convention makes the direct argument read-only without requiring
# an access contract on the callee.
@__nested_origins_read_only
def call_read_only_ref(ref value: Int):
    observe_ref(value)


# Register values can carry mutable nested origins, so callees that only
# inspect or forward them must state that contract.
@__nested_origins_read_only
def observe_pointer[origin: Origin](ptr: Pointer[Int, origin]):
    _ = ptr


@__nested_origins_read_only
def call_pointer_observer[origin: MutOrigin](ptr: Pointer[Int, origin]):
    observe_pointer(ptr)

# ===----------------------------------------------------------------------=== #
# Other more complicated logic that is also fine.

struct PointerView[origin: Origin]:
    var ptr: Pointer[Int, Self.origin]

    # Initializing a field forwards the pointer without accessing its pointee.
    @__nested_origins_read_only
    def __init__(out self, ptr: Pointer[Int, Self.origin]):
        self.ptr = ptr


@__nested_origins_read_only
def wrap_pointer[origin: MutOrigin](ptr: Pointer[Int, origin]) -> PointerView[origin]:
    return PointerView[origin](ptr)


trait ReadOnlySized:
    @__nested_origins_read_only
    def read_only_size(self) -> Int:
        ...


struct OriginSized[origin: Origin](ReadOnlySized):
    var ptr: Pointer[Int, Self.origin]

    @__nested_origins_read_only
    def read_only_size(self) -> Int:
        return 0


@__nested_origins_read_only
def forward_read_only_size[T: ReadOnlySized](value: T) -> Int:
    return value.read_only_size()


@__nested_origins_read_only
def call_forwarded_size[origin: MutOrigin](
    value: OriginSized[origin],
) -> Int:
    return forward_read_only_size(value)
