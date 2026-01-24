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
"""WeightsRegistry with O(1) lookup.

Performance fix: O(n²) → O(n) for n weights.
- 14K weights: ~103M comparisons → ~14K lookups
- Expected speedup: >1000×
"""


struct WeightsRegistry(ImplicitlyCopyable):
    """Bag of weights with O(1) name-based lookup.

    Uses Dict for O(1) lookups instead of O(n) linear search.
    For 14K weights, this is the difference between minutes and milliseconds.
    """

    var _lookup: Dict[String, Int]
    var names: List[String]
    var weights: List[OpaquePointer[MutAnyOrigin]]

    fn __init__(out self):
        """Initialize an empty weights registry."""
        self._lookup = Dict[String, Int]()
        self.names = List[String]()
        self.weights = List[OpaquePointer[MutAnyOrigin]]()

    fn __init__(out self, names: List[String], weights: List[OpaquePointer[MutAnyOrigin]]):
        """Initialize with existing names and weights.

        Args:
            names: List of weight names.
            weights: List of weight pointers.
        """
        self._lookup = Dict[String, Int]()
        self.names = names.copy()
        self.weights = weights.copy()
        for i in range(len(self.names)):
            self._lookup[self.names[i]] = i

    fn __copyinit__(out self, existing: Self):
        """Copy an existing weights registry.

        Args:
            existing: The existing weights registry.
        """
        self._lookup = Dict[String, Int]()
        self.names = existing.names.copy()
        self.weights = existing.weights.copy()
        for i in range(len(self.names)):
            self._lookup[self.names[i]] = i

    def __getitem__(self, name: String) -> OpaquePointer[MutAnyOrigin]:
        """Get weight by name with O(1) lookup.

        Args:
            name: Name of the weight.

        Returns:
            Pointer to the weight data.

        Raises:
            Error: If weight name not found.
        """
        try:
            var idx = self._lookup[name]
            return self.weights[idx]
        except:
            raise Error("no weight called " + name + " in weights registry")

    fn __len__(self) -> Int:
        """Return number of weights."""
        return len(self.names)

    fn __contains__(self, name: String) -> Bool:
        """Check if name exists in registry."""
        return name in self._lookup
