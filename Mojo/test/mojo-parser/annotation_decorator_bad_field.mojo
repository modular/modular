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

# A field's annotations resolve with its own signature, so a bad sibling field
# neither suppresses a real error in them nor causes a follow-on one.


struct BadField:
    # expected-error @+1 {{use of unknown declaration 'NoSuchType'}}
    var bad: NoSuchType

    @__annotation(1)
    var good: Int

    # expected-error @+1 {{'BadField' value has no attribute 'Missing'}}
    @__annotation(Self.Missing)
    var wrong: Int
