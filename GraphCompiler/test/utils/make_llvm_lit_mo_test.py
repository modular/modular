# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""
This script automatically generates an LIT formatted pattern that should match
what you get from a mlir file.

E.g. Let's say we want to make a pattern for
```test.mlir

mo.graph @add_static_static(%arg0: !mo.tensor<[1, 2, 3], f32>, %arg1: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32> {
  %1 = rmo.mo.add(%arg0, %arg1) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32>
  mo.output %1 : !mo.tensor<?, f32>
}
```

We run `mo-opt test.mlir | python make_llvm_lit_test.py` will output:

```
// CHECK-LABEL: mo.graph
// CHECK-SAME: @add_static_static([[VARG0:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>, [[VARG1:[^ ]*]]: !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32> {
// CHECK-NEXT: [[V0:[^ ]*]] = rmo.mo.add([[VARG0]], [[VARG1]]) : (!mo.tensor<[1, 2, 3], f32>, !mo.tensor<[1, 2, 3], f32>) -> !mo.tensor<?, f32>
// CHECK-NEXT: mo.output [[V0]] : !mo.tensor<?, f32>
```

Which is ready to be used in Lit.

Note we currently use `// -----` as a delimiter between tests. This is probably
something we should be doing anyway (along with --split-input-file in the
opt call).

"""

import re
import sys


def process_module_string(module_str: str):  # noqa: ANN201
    # Process text format of mlir module to lit-test pattern
    var_names = set()

    def get_lit_var(m, group_num, prefix=""):  # noqa: ANN001, ANN202
        var_name = prefix + m.group(group_num).upper()
        var_names.add(var_name)
        return var_name

    # Replace all SSA values like %3, %arg0, %v with [[V3]], [[VARG0]], [[VV]]
    module_str = re.sub(
        r"%([a-zA-Z0-9_]+)",
        lambda m: f"[[{get_lit_var(m, 1, 'V')}]]",
        module_str,
    )

    # Replace all parameters like Sh1, D10 into [[Sh1]], [[D10]]
    module_str = re.sub(
        r"((RMO_SH_EQ|RMO_SH_CH|RMO_Sh|RMO_D|Sh|D)([0-9]+))",
        lambda m: f"[[{get_lit_var(m, 0)}]]",
        module_str,
    )

    # Replace the first occurrences of all the lit var names with things which
    # match against everything but whitespace.
    for var_name in var_names:
        module_str = re.sub(
            f"\\[\\[{var_name}\\]\\]",
            f"[[{var_name}:[^ ]*]]",
            module_str,
            count=1,
        )

    return module_str


def pretty_print_module_string(module_str: str) -> None:
    # remove first two lines for enclosing module {} statement:
    module_str = "\n".join(module_str.split("\n")[1:-1])

    # get all the stuff after mo.graph (which we use to generate a CHECK-LABEL)
    module_str = module_str.split("mgp.model ")[1]
    lines = module_str.split("\n")

    print("// CHECK-LABEL: mgp.model")
    for i, line in enumerate(lines):
        prefix = "CHECK-SAME" if i == 0 else "CHECK-NEXT"
        print(f"// {prefix}: {line.strip()}")


if __name__ == "__main__":
    delimiter = "// -----"
    data = sys.stdin.read()
    split_data = data.split(delimiter)
    for module in split_data:
        processed_module_str = process_module_string(module)
        try:
            pretty_print_module_string(processed_module_str)
        except:
            print("failed")
        print(delimiter)
