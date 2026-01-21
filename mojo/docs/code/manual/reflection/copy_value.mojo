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

from reflection import struct_field_count, struct_field_types


trait MakeCopyable:
    fn copy_to(self, mut other: Self):
        comptime field_count = struct_field_count[Self]()
        comptime field_types = struct_field_types[Self]()

        @parameter
        for idx in range(field_count):
            comptime field_type = field_types[idx]

            # Guard: field type must be copyable
            @parameter
            if not conforms_to(field_type, Copyable):
                continue

            # Perform copy
            ref p_value = __struct_field_ref(idx, self)
            __struct_field_ref(idx, other) = p_value


@fieldwise_init
struct MultiType(MakeCopyable, Writable):
    var w: String
    var x: Int
    var y: Bool
    var z: Float64

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("[{}, {}, {}, {}]".format(self.w, self.x, self.y, self.z))


fn main():
    var original_instance = MultiType("Hello", 1, True, 2.5)
    var target_instance = MultiType("", 0, False, 0.0)

    print("Original instance:", original_instance)  # "Hello", 1, True, 2.5
    print("Target instance before: ", target_instance)  # "", 0, False, 0.0
    original_instance.copy_to(target_instance)
    print("Target instance after: ", target_instance)  # "Hello", 1, True, 2.5

    original_instance.z = 42.0
    print("Updated original:", original_instance)  # "Hello", 1, True, 42.0
    print("Target instance after: ", target_instance)  # "Hello", 1, True, 2.5
