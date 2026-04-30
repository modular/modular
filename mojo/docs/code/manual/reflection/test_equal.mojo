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


trait MakeCopyable:
    def copy_to(self, mut other: Self):
        comptime r = reflect[Self]()
        comptime field_count = r.field_count()
        comptime field_types = r.field_types()

        comptime for idx in range(field_count):
            comptime field_type = field_types[idx]

            # Guard: field type must be copyable
            comptime if not conforms_to(field_type, Copyable):
                continue

            # Perform copy
            ref p_value = trait_downcast[ImplicitlyCopyable](
                r.field_ref[idx](self)
            )
            trait_downcast[ImplicitlyCopyable](
                r.field_ref[idx](other)
            ) = p_value


@fieldwise_init
struct MultiType(MakeCopyable, Writable):
    var w: String
    var x: Int
    var y: Bool
    var z: Float64

    def write_to[W: Writer](self, mut writer: W):
        writer.write("[{}, {}, {}, {}]".format(self.w, self.x, self.y, self.z))


def test_equality[T: AnyType](lhs: T, rhs: T) -> Bool:
    comptime r = reflect[T]()
    comptime field_count = r.field_count()
    comptime field_types = r.field_types()

    comptime for idx in range(field_count):
        # Guard: field type must be equatable
        comptime field_type = field_types[idx]

        comptime if not conforms_to(field_type, Equatable):
            continue

        # Fetch values
        ref lhs_value = r.field_ref[idx](lhs)
        ref rhs_value = r.field_ref[idx](rhs)

        # Early exit `False` when inequality found
        if trait_downcast[Equatable](lhs_value) != trait_downcast[Equatable](
            rhs_value
        ):
            return False

    return True


def main():
    var original_instance = MultiType("Hello", 1, True, 2.5)
    var target_instance = MultiType("", 0, False, 0.0)
    original_instance.copy_to(target_instance)
    print(
        "Values equal" if test_equality(
            original_instance, target_instance
        ) else "Values not equal"
    )  # Values equal

    original_instance.z = 42.0
    print(
        "Values equal" if test_equality(
            original_instance, target_instance
        ) else "Values not equal"
    )  # Values not equal
