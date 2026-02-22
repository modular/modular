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
struct WeightsRegistry(ImplicitlyCopyable):
    """Bag of weights where names[i] names a weight with data weights[i]."""

    var names: List[String]
    var weights: List[OpaquePointer[MutAnyOrigin]]

    fn __init__(out self, *, copy: Self):
        """Copy an existing weights registry.

        Args:
            copy: The existing weights registry.
        """
        self.names = copy.names.copy()
        self.weights = copy.weights.copy()

    def __getitem__(self, name: String) -> OpaquePointer[MutAnyOrigin]:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.weights[i]

        raise Error("no weight called " + name + " in weights registry")
