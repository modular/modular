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

#include "GraphCompiler/MOGGDialect/Support/ModuleHasher.h"

#include "AsyncRT/CompilerSupport/Context.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/TransformUtils/SCCUtils.h"
#include "Support/AssertStream.h"
#include "Support/HashUtils.h"

#include "mlir/Bytecode/BytecodeWriter.h"

using namespace M;

namespace {

constexpr StringLiteral kSemanticHashName = "semantic_hash";

/// Per-node state machine: parts of the state depend on each other, and
/// SCCGraph iteratively applies an update function to each node in an SCC.
enum class State {
  SetGenerator,
  SetInputs,
  SetScc,
  SetFinal,
  Done,
};

State nextState(State state) {
  return (State)std::min((int)State::Done, (int)state + 1);
}

bool isAfter(State lhs, State rhs) { return (int)lhs > (int)rhs; }

struct Node
    : public KGEN::SCCNode<Node, KGEN::GeneratorOpInterface, KGEN::CallOp> {
  using Base = KGEN::SCCNode<Node, KGEN::GeneratorOpInterface, KGEN::CallOp>;

  using Base::Base;
  Node(KGEN::GeneratorOpInterface func)
      : Base(func), functionName(func ? func.getNameAttr() : nullptr) {}

  template <typename T>
  void addHash(T &&value) {
    *hasher << std::forward<T>(value);
  }

  /// Separately allocated: the hasher state is large (~800 bytes) and only
  /// needed while hashing the current SCC.
  std::unique_ptr<raw_xxhash128_stream> hasher;

  /// Need to save the function name as we have to temporarily mutate it in
  /// setGeneratorHash while computing the bytecode hash.
  StringAttr functionName;

  /// The bytecode hash of the generator for the node.
  /// We need this state here since it can be observed by other members of
  /// the same SCC.
  StringAttr generatorHash;

  /// The full hash of the current node:
  ///
  ///     totalHash = (hash(generator) + hash(inputs) + hash(SCC members))
  StringAttr totalHash;

  State state{State::SetGenerator};
};

struct Graph : public KGEN::SCCGraph<Graph, Node> {
  void build(ModuleOp module, const SymbolTable &symtab);

  bool doAnalysis(Node *node);
  void doRewrite(const Node *node);

  bool setGeneratorHash(Node *node);
  bool setInputsHash(Node *node);
  bool setFinalHash(Node *node);
  bool setSccHash(Node *node);
};

void Graph::build(ModuleOp module, const SymbolTable &symtab) {
  // Overrides CallGraphBase::build, whose edges are call ops: before
  // elaboration, KGEN IR can refer to functions through FlatSymbolRefAttrs in
  // arbitrary attributes and types, so every such reference is an edge. The
  // result is a superset of CallGraphBase::build's graph.

  // Create a callgraph node for each GeneratorOpInterface in the module.
  // This logic is the same as in the parent class' build method.
  for (auto func : module.getOps<FuncOpT>())
    nodes.try_emplace(func, Node(func));

  // Populate the callers and callees lists for each node in the graph.
  for (auto &[caller, callerNode] : nodes) {
    caller.walk([&](Operation *op) {
      mlir::AttrTypeWalker walker;
      walker.addWalk([&](FlatSymbolRefAttr symbol) {
        auto callee =
            dyn_cast_or_null<Node::FuncOpT>(symtab.lookup(symbol.getAttr()));
        if (!callee)
          return;

        Node *calleeNode = &nodes.find(callee)->second;
        callerNode.callsites.emplace_back(dyn_cast<CallOpT>(op), calleeNode);

        calleeNode->callers.push_back(&callerNode);
      });

      walker.walk(op->getAttrDictionary());

      for (auto type : op->getResultTypes())
        walker.walk(type);

      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (BlockArgument &arg : block.getArguments()) {
            walker.walk(arg.getType());
          }
        }
      }
    });
  }
}

bool Graph::setGeneratorHash(Node *node) {
  node->hasher = std::make_unique<raw_xxhash128_stream>();

  ModuleOp module = node->func->getParentOfType<ModuleOp>();
  ASSERT_STREAM(module, "generator must live under a module");
  SymbolTable symtab(module);

  auto ctx = node->func.getContext();
  Builder builder(ctx);

  // Erasing names works for SCC inputs only because setInputsHash incorporates
  // each callsite; map references to SCC members to unique numbers instead.
  llvm::SmallDenseMap<SymbolRefAttr, SymbolRefAttr> attrMap;
  for (auto [index, sccNode] : llvm::enumerate(node->scc->nodes)) {
    auto name = FlatSymbolRefAttr::get(sccNode->functionName);
    attrMap[name] =
        FlatSymbolRefAttr::get(builder.getStringAttr(std::to_string(index)));
  }

  auto emptyString = FlatSymbolRefAttr::get(builder.getStringAttr(""));
  auto unknownLoc = UnknownLoc::get(ctx);

  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](FlatSymbolRefAttr symbol) -> SymbolRefAttr {
    // Replace SCC member references with stable indices.
    if (auto mapped = attrMap.lookup(symbol))
      return mapped;
    // Replace external GeneratorOpInterface references with empty string —
    // their hash is incorporated via setInputsHash. Non-GeneratorOpInterface
    // refs (e.g., trait symbol refs from TypeConformsToTraitAttr) must be
    // preserved so they contribute to the bytecode hash.
    if (auto *op = symtab.lookup(symbol.getAttr()))
      if (isa<KGEN::GeneratorOpInterface>(op))
        return emptyString;
    return symbol;
  });
  replacer.addReplacement(
      [&](LocationAttr loc) -> LocationAttr { return unknownLoc; });

  // Set the function name so we do not incorporate the function name into
  // the bytecode hash.
  OwningOpRef<KGEN::GeneratorOpInterface> clone = node->func.clone();
  clone->setSymbolName("");

  replacer.recursivelyReplaceElementsIn(*clone,
                                        /*replaceAttrs=*/true,
                                        /*replaceLocs=*/true,
                                        /*replaceTypes=*/true);

  // Hash the bytecode of the backing generator.
  (void)mlir::writeBytecodeToFile(*clone, *node->hasher);

  node->generatorHash = builder.getStringAttr(node->hasher->hashString());
  return true;
}

bool Graph::setInputsHash(Node *node) {
  // Incorporate the hash of each callee for calls that target functions
  // outside the current SCC.
  for (auto sccNode : node->scc->nodes) {
    for (auto edge : sccNode->callsites) {
      // Calls within the current SCC will not have their hash set yet.
      if (auto nodeHash = edge.node->totalHash)
        node->addHash(edge.node->totalHash.strref());
    }
  }

  return true;
}

bool Graph::setSccHash(Node *node) {
  // Can only proceed if all nodes in the SCC have their generator hashes set.
  if (!llvm::all_of(node->scc->nodes, [&](Node *sccNode) {
        return isAfter(sccNode->state, State::SetGenerator);
      }))
    return false;

  for (auto sccNode : node->scc->nodes)
    node->addHash(sccNode->generatorHash.strref());

  return true;
}

bool Graph::setFinalHash(Node *node) {
  Builder builder(node->func.getContext());
  node->totalHash = builder.getStringAttr(node->hasher->hashString());
  node->hasher = nullptr;
  return true;
}

bool Graph::doAnalysis(Node *node) {
  State originalState = node->state;

  while (true) {
    bool advance = false;
    switch (node->state) {
    case State::SetGenerator:
      advance = setGeneratorHash(node);
      break;
    case State::SetScc:
      advance = setSccHash(node);
      break;
    case State::SetInputs:
      advance = setInputsHash(node);
      break;
    case State::SetFinal:
      advance = setFinalHash(node);
      break;
    case State::Done:
      advance = false;
      break;
    }

    if (!advance)
      break;
    node->state = nextState(node->state);
  }

  return node->state != originalState;
}

void Graph::doRewrite(const Node *node) {
  auto func = node->func;

  Builder builder(func->getContext());

  NamedAttrList attrs(func->getDiscardableAttrDictionary());
  attrs.append(kSemanticHashName, node->totalHash);
  func->setDiscardableAttrs(attrs);
}

} // end namespace

void M::MOGG::hashModuleContents(mlir::ModuleOp module) {
  SymbolTable symtab(module);
  hashModuleContents(module, symtab);
}

void M::MOGG::hashModuleContents(mlir::ModuleOp module,
                                 const mlir::SymbolTable &symtab) {
  AsyncRT::CPUDevice &cpuDevice =
      *loadContext(module.getContext())->get<AsyncRT::CPUDevice>();

  Graph graph;
  graph.build(module, symtab);
  graph.run(cpuDevice);
}

std::string M::MOGG::getSemanticHash(KGEN::GeneratorOp generator) {
  auto hashAttr = generator->getAttrOfType<StringAttr>(kSemanticHashName);
  ASSERT_STREAM(hashAttr, "unable to find semantic hash for function");
  return hashAttr.str();
}

llvm::DenseMap<StringAttr, KGEN::GeneratorOp>
M::MOGG::applySemanticNaming(ModuleOp module,
                             ArrayRef<StringAttr> symbolsToRename,
                             const SymbolTable &symtab, StringRef namePrefix) {
  // Mapping from old (public) names to the renamed generators.  The keys are
  // the names that callers in the MGP module use (i.e. the linkage name when
  // present, otherwise the sym_name).
  llvm::DenseMap<StringAttr, KGEN::GeneratorOp> renaming;
  renaming.reserve(symbolsToRename.size());

  if (symbolsToRename.empty())
    return renaming;

  // Linkage names are only set onto symbol names during elaboration, so the
  // symbol table can't find generators by linkage name yet. This assumes the
  // linkage names we need are literals (in theory they may be parameter
  // expressions).

  // Build a map from linkageName -> GeneratorOp so we can look up generators
  // by their public name even when the sym_name is mangled.
  llvm::DenseMap<StringAttr, KGEN::GeneratorOp> linkageNameToGen;
  for (auto gen : module.getOps<KGEN::GeneratorOp>()) {
    if (auto ln = gen.getLinkageNameAttr()) {
      // Our symbols never use parametric linkage names.
      if (auto str = dyn_cast<StringAttr>(ln.getName())) {
        // Convert back from a !kgen.string-typed StringAttr to an untyped one,
        // so that symbol equality works.
        linkageNameToGen[StringAttr::get(module.getContext(), str.getValue())] =
            gen;
      }
    }
  }

  auto findGenerator = [&](StringAttr name) -> KGEN::GeneratorOp {
    if (auto gen = symtab.lookup<KGEN::GeneratorOp>(name))
      return gen;
    return linkageNameToGen.lookup(name);
  };

  for (StringAttr symbol : symbolsToRename) {
    auto generator = findGenerator(symbol);
    ASSERT_STREAM(generator, << "symbol " << symbol << " not found in module");
    // Remove it because it influences the hashing.
    generator.removeSourceNameAttr();
  }

  hashModuleContents(module, symtab);

  // The semantic hashes of the functions we have visited already. This tells
  // us if we need to rename the current generator, or if we can delete it.
  llvm::DenseMap<StringAttr, KGEN::GeneratorOp> visited;
  visited.reserve(symbolsToRename.size());

  // Track old sym_name -> new sym_name for updating references within the
  // kgen module (sym_name may differ from the public linkage name).
  llvm::DenseMap<StringAttr, StringAttr> symRenaming;

  for (StringAttr symbol : symbolsToRename) {
    auto generator = findGenerator(symbol);
    auto hash = generator->getAttrOfType<StringAttr>(kSemanticHashName);
    ASSERT_STREAM(hash, "module has not had semantic hashes computed");

    StringAttr oldSymName = generator.getSymNameAttr();

    // Semantically equivalent to a visited generator: delete it and remap its
    // uses.
    if (auto remapped = visited.lookup_or(hash, nullptr)) {
      symRenaming[oldSymName] = remapped.getSymNameAttr();
      renaming[symbol] = remapped;
      generator.erase();
    } else {
      auto newName = StringAttr::get(module.getContext(),
                                     Twine(namePrefix) + hash.strref());

      symRenaming[oldSymName] = newName;
      renaming[symbol] = generator;
      generator.setSymNameAttr(newName);
      // Remove the linkage name so that it doesn't propagate through
      // elaboration and cause the linkage name resolution to undo this rename.
      generator.removeLinkageNameAttr();
      visited.insert({hash, generator});
    }
  }

  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement([&](FlatSymbolRefAttr ref) {
    if (auto newName = symRenaming.lookup_or(ref.getAttr(), {}))
      return FlatSymbolRefAttr::get(newName);
    return ref;
  });

  replacer.recursivelyReplaceElementsIn(module,
                                        /*replaceAttrs=*/true,
                                        /*replaceLocs=*/false,
                                        /*replaceTypes=*/true);

  return renaming;
}
