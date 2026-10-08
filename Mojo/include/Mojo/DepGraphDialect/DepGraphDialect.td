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

#ifndef DEPGRAPHDIALECT_TD
#define DEPGRAPHDIALECT_TD

include "mlir/IR/DialectBase.td"

def DepGraphDialect : Dialect {
  let name = "depgraph";

  let summary = "dialect for representing module import dependency graphs";
  let description = [{
    The `depgraph` dialect represents the import dependency graph among Mojo
    modules and packages as an explicit graph IR.

    A `depgraph.graph` container holds `depgraph.module` node operations, each
    listing its direct dependencies as SSA operands.  Because the container
    uses a graph region (no terminator, no ordering constraints), it can
    represent cyclic dependency structures that arise during error reporting
    before the cycle is diagnosed.
  }];

  let cppNamespace = "::M::KGEN::DepGraph";
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerTypes();
    void registerOperations();
  }];
}

#endif // DEPGRAPHDIALECT_TD
