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

#ifndef DEPGRAPHOPS_TD
#define DEPGRAPHOPS_TD

include "Mojo/DepGraphDialect/DepGraphDialect.td"
include "Mojo/DepGraphDialect/DepGraphTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class DepGraphOp<string mnemonic, list<Trait> traits = []>
    : Op<DepGraphDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// GraphOp
//===----------------------------------------------------------------------===//

def DepGraph_GraphOp
    : DepGraphOp<"graph",
                 [NoRegionArguments] # GraphRegionNoTerminator.traits> {
  let summary = "container for a module import dependency graph";
  let description = [{
    A `depgraph.graph` holds a flat graph region containing `depgraph.module`
    nodes.  Each module lists its direct dependencies as SSA operands — the
    use-def graph is the dependency graph.

    The region is a *graph region*: operations may reference SSA values defined
    anywhere in the same block, including later in the block.  This allows
    cycles (e.g. mutual imports) to be represented for diagnostic purposes.

    The optional `root` symbol names the top-level module of the compilation —
    the node every other module is (transitively) a dependency of.  It is a
    reference to a `depgraph.module` symbol defined in the graph's body.

    Example:

    ```mlir
    depgraph.graph root @main {
      %std  = depgraph.module @std
      %util = depgraph.module @util (%std)
      %main = depgraph.module @main (%util, %std)
    }
    ```
  }];

  let arguments = (ins OptionalAttr<FlatSymbolRefAttr>:$root);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "(`root` $root^)? $body attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Returns the module node with the given symbol name, or null.
    ModuleNodeOp lookupModule(llvm::StringRef name);

    /// Returns the root module node, or null if no root is set or the root
    /// symbol does not resolve to a module in this graph.
    ModuleNodeOp getRootModule();
  }];
}

//===----------------------------------------------------------------------===//
// ModuleNodeOp
//===----------------------------------------------------------------------===//

def DepGraph_ModuleNodeOp : DepGraphOp<"module"> {
  let summary = "a module or package node in the dependency graph";
  let description = [{
    A `depgraph.module` represents a single Mojo compilation unit — a file
    module or a package — as a node in the dependency graph.  It yields a
    `!depgraph.node` handle and lists its direct import dependencies as
    operands.  The SSA use-def graph is the dependency graph.

    The node carries a `sym_name` for `@name` syntax and name-based lookup
    (`GraphOp::lookupModule`), but is intentionally *not* an MLIR `Symbol`:
    a symbol op may not produce results, and this node's `!depgraph.node`
    result is what forms the use-def dependency edges.

    Attributes:
    - `sym_name` (required): the fully-qualified, dotted module name
      (e.g. `std.collections.__init__`), so a package's init file is
      distinguishable from every other `__init__`.
    - `path` (optional): filesystem path of the source file or package root.
    - `content_hash` (optional): lowercase hex xxh3-64 digest of the source
      file content at the time the module was parsed.

    Operands:
    - `deps` (variadic): handles of modules that this module directly imports.

    Example:

    ```mlir
    %std  = depgraph.module @std
    %util = depgraph.module @util (%std) { path = "util.mojo" }
    %main = depgraph.module @main (%util, %std) { path = "main.mojo" }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$path,
    OptionalAttr<StrAttr>:$content_hash,
    Variadic<DepGraph_NodeType>:$deps
  );
  let results = (outs DepGraph_NodeType:$handle);

  let assemblyFormat = "$sym_name (`(` $deps^ `)`)? prop-dict attr-dict";
  let hasVerifier = 1;

  let builders = [
    /// Build a module node with full metadata and dependencies.
    OpBuilder<(ins "llvm::StringRef":$name, "llvm::StringRef":$path,
                   "mlir::ValueRange":$deps)>,
  ];
}

#endif // DEPGRAPHOPS_TD
