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
# RUN: not %mojo %s 2>&1 | FileCheck %s


__enum Linear(not Deinitable):
    case open

    def finish(deinit self):
        pass


def main():
    var value: Linear = .open
    # CHECK: error: 'copy' abandoned without being explicitly destroyed: type 'Linear' does not conform to 'Deinitable' and must be explicitly destroyed
    var copy = value
    value^.finish()
