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

# RUN: %mojo %s | FileCheck %s

# `@__annotation` values attached to a struct and its fields are read back at
# comptime through `#kgen.get_num_annotations` and `#kgen.get_annotation_value`,
# then materialized and used at runtime.

comptime __get_struct_field_annotation[
    T: AnyType,
    field_idx: Int,
    annotation_idx: Int,
] = __mlir_attr[
    `#kgen.get_annotation_value<`,
    T,
    `,`,
    AnyType,
    `,`,
    annotation_idx._mlir_value,
    `,`,
    field_idx._mlir_value,
    `>`,
]


comptime __get_struct_annotation[
    T: AnyType,
    annotation_idx: Int,
] = __mlir_attr[
    `#kgen.get_annotation_value<`,
    T,
    `,`,
    AnyType,
    `,`,
    annotation_idx._mlir_value,
    `>`,
]


comptime __get_num_struct_annotation[
    T: AnyType,
] = Int(
    Scalar[DType.int](
        mlir_value=__mlir_attr[
            `#kgen.get_num_annotations<`,
            T,
            `>`,
        ]
    )
)


comptime __get_num_struct_field_annotation[
    T: AnyType,
    field_idx: Int,
] = Int(
    Scalar[DType.int](
        mlir_value=__mlir_attr[
            `#kgen.get_num_annotations<`,
            T,
            `,`,
            field_idx._mlir_value,
            `>`,
        ]
    )
)


@fieldwise_init
struct Tag(Deinitable, Movable):
    var name: StaticString


def fn_annotation():
    print("call annotated function")


@__annotation(Tag("x"))
@__annotation(fn_annotation)
struct Target:
    @__annotation(1)
    var value: Int

    var plain: Int


# Annotations are written in the struct's scope, so they can name its
# parameters; reading one back rebinds it against the queried instantiation.
@__annotation(Self.N)
struct Generic[N: Int]:
    @__annotation(Self.N + 1)
    var value: Int


def read_field[T: AnyType]() -> Int:
    comptime annotation = __get_struct_field_annotation[T, 0, 0]
    comptime assert conforms_to(type_of(annotation), Movable)
    return rebind_var[Int](materialize[annotation]())


def read_struct[T: AnyType]() -> Tag:
    comptime annotation = __get_struct_annotation[T, 0]
    comptime assert conforms_to(type_of(annotation), Movable)
    return rebind_var[Tag](materialize[annotation]())


def read_struct_int[T: AnyType]() -> Int:
    comptime annotation = __get_struct_annotation[T, 0]
    comptime assert conforms_to(type_of(annotation), Movable)
    return rebind_var[Int](materialize[annotation]())


def call_annotated_fn[T: AnyType]():
    comptime cb = rebind[def() thin](__get_struct_annotation[T, 1])
    cb()


def main():
    # CHECK: 1
    print(read_field[Target]())
    # CHECK-NEXT: x
    print(read_struct[Target]().name)

    # CHECK-NEXT: 2
    print(__get_num_struct_annotation[Target])
    # CHECK-NEXT: 1
    print(__get_num_struct_field_annotation[Target, 0])
    # CHECK-NEXT: 0
    print(__get_num_struct_field_annotation[Target, 1])

    # A function can be attached as an annotation and called.
    # CHECK-NEXT: call annotated function
    call_annotated_fn[Target]()

    # CHECK-NEXT: 3
    print(read_struct_int[Generic[3]]())
    # CHECK-NEXT: 8
    print(read_field[Generic[7]]())
