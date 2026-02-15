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


@fieldwise_init
struct ValidationError(Copyable, Writable):
    var field: String
    var reason: String

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("ValidationError(", self.field, "): ", self.reason)


# def: implicitly raises Error — cannot use typed errors
def validate_def(value: Int) -> Int:
    if value < 0:
        raise "value cannot be negative"
    return value


# fn: can specify a typed error
fn validate_fn(value: Int) raises ValidationError -> Int:
    if value < 0:
        raise ValidationError(field="value", reason="cannot be negative")
    return value


# Wrapping pattern: fn wrapper converts Error to typed error
fn validated_operation(value: Int) raises ValidationError -> Int:
    try:
        return validate_def(value)
    except e:
        raise ValidationError(field="value", reason=String(e))


def main():
    # def: error has no fields
    print("--- def (Error) ---")
    try:
        _ = validate_def(-1)
    except e:
        print("Caught:", e)

    # fn: error has typed fields
    print("--- fn (typed error) ---")
    try:
        _ = validate_fn(-1)
    except e:
        print("Caught:", e)
        print("  field:", e.field)
        print("  reason:", e.reason)

    # Wrapping def in fn for typed errors
    print("--- wrapper (def -> typed) ---")
    try:
        _ = validated_operation(-1)
    except e:
        print("Caught:", e)
        print("  field:", e.field)
        print("  reason:", e.reason)
