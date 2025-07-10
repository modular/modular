# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


def render_constrained_deps(
    var_name: str,
    deps_by_constraints: dict[str, list[str]],
    deps_by_gpu_constraints: dict[str, list[str]],
) -> str:
    if not deps_by_constraints and not deps_by_gpu_constraints:
        return ""

    output = f"""\
    {var_name} = [
"""

    if "" in deps_by_constraints:
        unconstrained_deps = deps_by_constraints.pop("")
        for dep in sorted(unconstrained_deps):
            output += f'        ":{dep}",\n'

    had_select = False
    if deps_by_constraints:
        output += "    ] + select({\n"
        had_select = True

    for constraint, constrained_deps in sorted(deps_by_constraints.items()):
        output += f'        "{constraint}": [\n'
        for dep in sorted(constrained_deps):
            output += f'            ":{dep}",\n'
        output += "        ],\n"

    if had_select:
        output += '        "//conditions:default": [],\n'

    if deps_by_gpu_constraints:
        if had_select:
            output += "    }) + select({\n"
        else:
            output += "    ] + select({\n"
        had_select = True

    for constraint, constrained_deps in sorted(deps_by_gpu_constraints.items()):
        output += f'        "{constraint}": [\n'
        for dep in sorted(constrained_deps):
            output += f'            ":{dep}",\n'
        output += "        ],\n"

    if had_select:
        output += "    })\n"
    else:
        output += "    ]\n"

    output += "\n"
    return output
