//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// This defines types in the MAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTYPES_TD
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTYPES_TD

include "GraphCompiler/MAPDialect/Dialect/MAPDialect.td"
include "mlir/IR/AttrTypeBase.td"

// Base class for types in this dialect.
class MAP_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<MAPDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def MAP_Index : MAP_Type<"Index", "index"> {
  let summary = "A statically-ranked index into an iteration space.";
  let description = [{
    Represents an index into a `map.iter` kernel's iteration space, carrying its
    static rank.

    Example:
    ```mlir
    !map.index<2>    // a rank-2 index
    ```
  }];

  let parameters = (ins "uint64_t":$rank);
  let assemblyFormat = "`<` $rank `>`";
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPTYPES_TD
