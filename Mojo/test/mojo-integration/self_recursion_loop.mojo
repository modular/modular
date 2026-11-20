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

# RUN: kgen -elaborate %s -verify-diagnostics

# Issue #7016: Self recursive calls inside a loop that can exit via break
# should not trigger false-positive infinite recursion warnings.


def visit_a(node: Int, children: List[List[Int]]):
    print(node)
    for c in children[node]:
        visit_a(c, children)


def infinite_loop(x: Int):
    # expected-warning @+1 {{self recursive call will cause an infinite loop}}
    infinite_loop(x)
