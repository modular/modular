//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef DEPGRAPHTYPES_TD
#define DEPGRAPHTYPES_TD

include "Mojo/DepGraphDialect/DepGraphDialect.td"
include "mlir/IR/AttrTypeBase.td"

class DepGraphType<string typeName, string typeMnemonic,
                   list<Trait> traits = []>
    : TypeDef<DepGraphDialect, typeName, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// NodeType
//===----------------------------------------------------------------------===//

def DepGraph_NodeType : DepGraphType<"Node", "node"> {
  let summary = "SSA handle to a module node in the dependency graph";
  let description = [{
    The `!depgraph.node` type is an opaque handle produced by a
    `depgraph.module` operation.  A module names the handles of the modules it
    directly imports as its `deps` operands, forming the dependency edges.

    Example:

    ```mlir
    %m = depgraph.module @foo { path = "foo.mojo" }  // %m : !depgraph.node
    ```
  }];

  let assemblyFormat = "";
}

#endif // DEPGRAPHTYPES_TD
