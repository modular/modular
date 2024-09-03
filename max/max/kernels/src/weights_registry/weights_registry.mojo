# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from collections import List
from memory import UnsafePointer


@value
struct WeightsRegistry:
    """Bag of weights where names[i] names a weight with data weights[i]."""

    var names: List[String]
    var weights: List[UnsafePointer[NoneType]]

    def __getitem__(self, name: String) -> UnsafePointer[NoneType]:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return self.weights[i]

        raise Error("no weight called " + name + " in weights registry")
