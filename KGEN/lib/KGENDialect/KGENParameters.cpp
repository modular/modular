//===- KGENParameters.cpp - KGEN Parameter utilities ----------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for working with KGEN parameters.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/MetaDialect/MetaDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace M;
using namespace M::KGEN;

namespace llvm {

/// Make `ParamDeclRefAttr` behave like a `Attribute` in terms of
/// pointer-likeness.
template <>
struct PointerLikeTypeTraits<ParamDeclRefAttr>
    : PointerLikeTypeTraits<mlir::Attribute> {
  static inline void *getAsVoidPointer(ParamDeclRefAttr v) {
    return const_cast<void *>(v.getAsOpaquePointer());
  }
  static inline ParamDeclRefAttr getFromVoidPointer(void *p) {
    return ParamDeclRefAttr::getFromOpaquePointer(p);
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Parameter Verification
//===----------------------------------------------------------------------===//

namespace {
struct ParameterVerifier final {
  ParameterVerifier(ParameterDeclsAndUses &parameters)
      : parameters(parameters) {}

  /// Walk the operation and all the operations in its body to find the
  /// definitions and uses of parameters.  This diagnoses and rejects parameter
  /// definitions in invalid positions as well.
  LogicalResult collectParameterDefsAndUses(Operation *topLevelOp);

  /// Once all the defs and uses of parameters are collected, verify that the
  /// uses are correct.
  LogicalResult checkParameterUses(Operation *topLevelOp);

  /// Verify that the parameter use-def graph has a partial ordering, diagnosing
  /// any cycles that are present.
  LogicalResult checkParameterUseDefGraph(Operation *topLevelOp);

  /// Return the set of parameter uses for the specified operation.
  SmallVector<ParamDeclRefAttr> &getUsesForOperation(Operation *op) const {
    auto it = opIndexInUses.find(op);
    assert(op && it != opIndexInUses.end());
    return parameters.usersAndDeclarers[it->second].second;
  }

  /// This is the parameter information that we're building.
  ParameterDeclsAndUses &parameters;

private:
  /// Scan the specified attribute and its recursive uses, diagnosing incorrect
  /// parameter declarations and collecting parameter uses into `uses`.  This
  /// emits errors at the specified location.
  void collectParameterUsesFromAttr(Attribute attr,
                                    SmallVector<ParamDeclRefAttr> &uses,
                                    Location loc);

  /// Scan the specified type and its recursive uses, diagnosing incorrect
  /// parameter declarations and collecting parameter uses into `uses`.  This
  /// emits errors at the specified location.
  void collectParameterUsesFromType(Type type,
                                    SmallVector<ParamDeclRefAttr> &uses,
                                    Location loc);

  /// This is set to true if we find a problem during the collect phase.
  bool hadError = false;

  /// A single operation may use multiple parameter declarations, either
  /// directly or through types on attributes and SSA operands/results.  This
  /// keeps track of all of the uses that happen anywhere within an operation.
  DenseMap<Operation *, size_t> opIndexInUses;

  /// Attributes and types are memoized and exist in tree structures with reuse:
  /// naively scanning them can lead to exponential compile time behavior.  As
  /// such, we memoize the attributes and types we've already checked that we
  /// know have no parameters in them.
  llvm::SmallDenseSet<Attribute> parameterLessAttrs;
  llvm::SmallDenseSet<Type> parameterLessTypes;
};
} // end anonymous namespace.

/// Walk the operation and all the operations in its body to find the
/// definitions and uses of parameters.  This diagnoses and rejects parameter
/// definitions in invalid positions as well.
LogicalResult
ParameterVerifier::collectParameterDefsAndUses(Operation *topLevelOp) {
  // TODO: We probably shouldn't walk into IsolatedFromAbove operations.  This
  // walk may need to be adjusted if we have any.
  topLevelOp->walk<mlir::WalkOrder::PreOrder>([&](Operation *bodyOp) {
    ParamDeclArrayAttr paramDeclsAttr;
    SmallVector<ParamDeclRefAttr> paramUses;

    // Scan all the attributes and types to look for uses of parameters.  We let
    // the walker scan the region hierarchy.
    for (const NamedAttribute &namedAttr : bodyOp->getAttrs()) {
      // For normal attributes, we scan the attribute tree looking or parameter
      // uses and reject unexpected parameter definitions.
      if (namedAttr.getName().strref() != "paramDecls") {
        collectParameterUsesFromAttr(namedAttr.getValue(), paramUses,
                                     bodyOp->getLoc());
        continue;
      }

      // We handle the `paramDecls` attribute specially, remember it for below.
      paramDeclsAttr = namedAttr.getValue().dyn_cast<ParamDeclArrayAttr>();
      if (!paramDeclsAttr) {
        bodyOp->emitError("paramDecls attribute should be an array ")
            << namedAttr.getValue();
        hadError = true;
        return;
      }
    }

    // Check the types of results to find any parameters embedded in their
    // types.  We don't have to check operands because they are always checked
    // when being defined.
    for (Type type : bodyOp->getResultTypes())
      collectParameterUsesFromType(type, paramUses, bodyOp->getLoc());

    // Scan the region list if present.  The walker will automatically recurse
    // for us, but we have to check the block arguments.
    if (bodyOp->getNumRegions()) { // Microoptimization: getRegions() is slow.
      for (auto &region : bodyOp->getRegions()) {
        for (auto &block : region)
          for (Value arg : block.getArguments())
            collectParameterUsesFromType(arg.getType(), paramUses,
                                         bodyOp->getLoc());
      }
    }

    // If this operation had any parameter uses or decls, remember them.
    if (!paramUses.empty() || paramDeclsAttr)
      parameters.usersAndDeclarers.push_back({bodyOp, std::move(paramUses)});

    // Ok, check parameter declarations if present.
    if (!paramDeclsAttr)
      return;

    for (Attribute attr : paramDeclsAttr) {
      // All the members of this array must be ParamDeclAttr's.
      auto param = attr.dyn_cast<ParamDeclAttr>();
      if (!param) {
        bodyOp->emitError("unknown attribute kind in paramDecls list ") << attr;
        hadError = true;
        return;
      }

      // We cannot have any redefinitions.
      auto &opAndDeclAttr = parameters.decls[param.getName()];
      if (opAndDeclAttr.first) {
        auto diag = bodyOp->emitError("redeclaration of parameter ")
                    << param.getName();
        diag.attachNote(opAndDeclAttr.first->getLoc())
            << "previous declaration here";
        hadError = true;
        return;
      }
      opAndDeclAttr = {bodyOp, param};
    }
  });

  return failure(hadError);
}

/// Scan the specified attribute and its recursive uses, diagnosing incorrect
/// parameter declarations and collecting parameter uses.
void ParameterVerifier::collectParameterUsesFromAttr(
    Attribute attr, SmallVector<ParamDeclRefAttr> &uses, Location loc) {

  // Collect parameter references.
  if (auto paramRef = attr.dyn_cast<ParamDeclRefAttr>()) {
    uses.push_back(paramRef);
    return;
  }

  // If this attribute has no sub-attributes or we have already scanned it an
  // know that it has no parameters in it, return early.
  if (attr.isa<IntegerAttr, FloatAttr, StringAttr, SymbolRefAttr,
               DenseElementsAttr, mlir::DenseArrayBaseAttr>() ||
      parameterLessAttrs.contains(attr))
    return;

  // Reject errant parameter decls.
  if (auto paramDecl = attr.dyn_cast<ParamDeclAttr>()) {
    emitError(loc, "invalid ParamDeclAttr outside of paramDecls attribute ")
        << paramDecl;
    return;
  }

  size_t oldSize = uses.size();

  // Otherwise we haven't processed this, check the attribute's type if it has
  // one.
  // TODO: Capture types using SubElementAttrInterface.
  if (auto typedAttr = attr.dyn_cast<TypedAttr>())
    collectParameterUsesFromType(typedAttr.getType(), uses, loc);

  // Recursively check for any nested types/attributes, e.g. the elements of an
  // array attribute.
  if (auto itf = attr.dyn_cast<mlir::SubElementAttrInterface>()) {
    itf.walkSubElements(
        [&](Attribute attr) { collectParameterUsesFromAttr(attr, uses, loc); },
        [&](Type type) { collectParameterUsesFromType(type, uses, loc); });
  } else if (attr.isa<DTypeConstantAttr>()) {
    // This attribute doesn't participate with SubElementAttrInterface but we
    // know it doesn't have any subelements.
  } else {
    // Conservatively reject unknown attributes, we don't want someone to forget
    // to conform to SubElementAttrInterface.
    emitError(loc, "unknown attribute for parameterization scan: ") << attr;
  }

  // If the attribute had no uses, remember that so we don't have to re-scan it
  // in the future.
  if (oldSize == uses.size())
    parameterLessAttrs.insert(attr);
}

/// Scan the specified type and its recursive uses, diagnosing incorrect
/// parameter declarations and collecting parameter uses.
void ParameterVerifier::collectParameterUsesFromType(
    Type type, SmallVector<ParamDeclRefAttr> &uses, Location loc) {
  // Ignore types we have already scanned.
  if (parameterLessTypes.count(type))
    return;

  size_t oldSize = uses.size();

  // Recursively check for any nested types, e.g. the input/outputs of a
  // function type.  This also handles types like !meta.scalar etc.
  if (auto itf = type.dyn_cast<mlir::SubElementTypeInterface>()) {
    itf.walkSubElements(
        [&](Attribute attr) { collectParameterUsesFromAttr(attr, uses, loc); },
        [&](Type type) { collectParameterUsesFromType(type, uses, loc); });
  } else {
    // These are known leaf types that don't participate with
    // SubElementTypeInterface.
    if (!type.isa<IntegerType, FloatType, NoneType, IndexType, DTypeType>()) {
      // Conservatively reject unknown types, we don't want someone to forget to
      // conform to SubElementTypeInterface.
      emitError(loc, "unknown type for parameterization scan: ") << type;
    }
  }

  // If the attribute had no uses, remember that so we don't have to re-scan it
  // in the future.
  if (oldSize == uses.size())
    parameterLessTypes.insert(type);
}

/// Once all the defs and uses of parameters are collected, verify that the
/// uses are correct.
LogicalResult ParameterVerifier::checkParameterUses(Operation *topLevelOp) {
  // Take a look at all the parameter uses to verify they are referencing
  // defined parameters and that they are used with the correct type.
  size_t usersAndDeclarersIndex = 0;
  for (auto &[usingOp, paramRefAttrArray] : parameters.usersAndDeclarers) {
    for (auto paramRefAttr : paramRefAttrArray) {
      // Check the use is referring to a parameter that was defined.
      auto decl = parameters.decls[paramRefAttr.getName()];
      if (!decl.first) {
        usingOp->emitOpError("invalid use of parameter with no declaration ")
            << paramRefAttr.getName();
        return failure();
      }

      // Check that the types of the uses match the defs.
      if (decl.second.getType() != paramRefAttr.getType()) {
        auto diag = usingOp->emitOpError("reference to parameter ")
                    << paramRefAttr.getName() << " with incorrect type "
                    << paramRefAttr.getType();
        diag.attachNote(decl.first->getLoc())
            << "parameter defined with type " << decl.second.getType();
        return failure();
      }
    }

    // Build the `opIndexInUses` map so the graph iterator can be efficient.
    assert(usingOp && "null operations shouldn't appear here");
    opIndexInUses[usingOp] = usersAndDeclarersIndex++;
  }

  return success();
}

/// Collect information about the parameter definitions and uses in the
/// specified operation.  This emits an error and returns `None` on an IR
/// verification error.
FailureOr<ParameterDeclsAndUses>
ParameterDeclsAndUses::calculate(Operation *topLevelOp) {
  ParameterDeclsAndUses result;
  ParameterVerifier verifier(result);

  // Start by doing a pass over the operation and all the operations in its
  // body to find the definitions and uses of parameters.
  if (failed(verifier.collectParameterDefsAndUses(topLevelOp)) ||
      // Next, now that we know the set of parameters we have to process, verify
      // that the uses match up.
      failed(verifier.checkParameterUses(topLevelOp)) ||
      // Verify that there are no cycles in the graph.
      failed(verifier.checkParameterUseDefGraph(topLevelOp)))
    return failure();

  return std::move(result);
}

//===----------------------------------------------------------------------===//
// ParameterUseDefGraph Implementation
//===----------------------------------------------------------------------===//

namespace {
class ParameterUseDefGraphNodeIterator;

/// This class defines a "node iterator" in the graph of operations that use and
/// define parameters.  Each node in this graph is an operation.  Each edge
/// between the nodes is a parameter use-def edge.
///
/// This uses a null `op` as a special representation for the root node.  This
/// node acts like it points to all the using operations.
class ParameterUseDefGraphNode {
public:
  ParameterUseDefGraphNode(const ParameterVerifier *verifier, Operation *op)
      : verifier(verifier), op(op) {}

  bool operator==(const ParameterUseDefGraphNode &rhs) const {
    assert(verifier == rhs.verifier && "node from different graphs?");
    return op == rhs.op;
  }
  bool operator!=(const ParameterUseDefGraphNode &rhs) const {
    return !(*this == rhs);
  }

  /// return the operation this node corresponds to.
  Operation *getOperation() const { return op; }
  const ParameterVerifier *getVerifier() const { return verifier; }

  /// Given a normal node (not the entry node) return the parameter uses.
  ArrayRef<ParamDeclRefAttr> getUsesForOperation() const {
    assert(op && "entry node doesn't have uses");
    return verifier->getUsesForOperation(op);
  }

  ParameterUseDefGraphNodeIterator begin() const;
  ParameterUseDefGraphNodeIterator end() const;

private:
  friend class ParameterUseDefGraphNodeIterator;
  friend struct llvm::DenseMapInfo<ParameterUseDefGraphNode>;
  const ParameterVerifier *verifier;
  Operation *op;
};
} // end anonymous namespace

namespace llvm {
template <>
struct DenseMapInfo<ParameterUseDefGraphNode> {
  static inline ParameterUseDefGraphNode getEmptyKey() {
    return {nullptr, DenseMapInfo<Operation *>::getEmptyKey()};
  }
  static inline ParameterUseDefGraphNode getTombstoneKey() {
    return {nullptr, DenseMapInfo<Operation *>::getTombstoneKey()};
  }
  static unsigned getHashValue(const ParameterUseDefGraphNode &node) {
    return DenseMapInfo<Operation *>::getHashValue(node.op);
  }

  static bool isEqual(const ParameterUseDefGraphNode &lhs,
                      const ParameterUseDefGraphNode &rhs) {
    return lhs.op == rhs.op;
  }
};
} // namespace llvm

namespace {
class ParameterUseDefGraphNodeIterator
    : public llvm::iterator_facade_base<ParameterUseDefGraphNodeIterator,
                                        std::forward_iterator_tag,
                                        ParameterUseDefGraphNode> {
public:
  ParameterUseDefGraphNodeIterator(const ParameterUseDefGraphNode &node,
                                   unsigned useNumber)
      : node(node), useNumber(useNumber) {}

  bool operator==(const ParameterUseDefGraphNodeIterator &rhs) const {
    return node == rhs.node && useNumber == rhs.useNumber;
  }

  ParameterUseDefGraphNode operator*() const {
    auto *verifier = node.getVerifier();
    // The entry node of the graph is a virtual node designated with a null
    // Operation* which indexes all of the nodes in the graph.
    if (node.getOperation() == nullptr)
      return {verifier,
              verifier->parameters.usersAndDeclarers[useNumber].first};

    // Otherwise we index into the 'usesByOp' array in the verifier.
    StringAttr paramName = getParameterName();
    auto it = verifier->parameters.decls.find(paramName);
    assert(it != verifier->parameters.decls.end() &&
           "already checked that used parameters are defined");
    // Get the operation defining the parameter.
    return {verifier, it->second.first};
  }

  ParameterUseDefGraphNodeIterator operator++() {
    ++useNumber;
    return *this;
  }
  ParameterUseDefGraphNodeIterator operator++(int) {
    ParameterUseDefGraphNodeIterator tmp = *this;
    ++*this;
    return tmp;
  }

  const ParameterUseDefGraphNode &getSourceNode() const { return node; }

  StringAttr getParameterName() const {
    assert(node.getOperation() != nullptr && "entry node cannot be cyclic");
    return node.getUsesForOperation()[useNumber].getName();
  }

private:
  ParameterUseDefGraphNode node;
  unsigned useNumber;
};
} // end anonymous namespace

ParameterUseDefGraphNodeIterator ParameterUseDefGraphNode::begin() const {
  return ParameterUseDefGraphNodeIterator(*this, 0);
}

ParameterUseDefGraphNodeIterator ParameterUseDefGraphNode::end() const {
  assert(verifier && "cannot get children of invalid node");
  unsigned endIndex;
  if (op == nullptr) {
    // Handle the special case of the virtual root node: the end index is the
    // end of the array of operators using parameters.
    endIndex = verifier->parameters.usersAndDeclarers.size();
  } else {
    endIndex = getUsesForOperation().size();
  }

  return ParameterUseDefGraphNodeIterator(*this, endIndex);
}

/// The ParameterVerifier graph is defined with `ParameterUseDefGraphNode` nodes
/// and `ParameterUseDefGraphNodeIterator` iterators.
namespace llvm {
template <>
struct GraphTraits<ParameterVerifier *> {
  using NodeRef = ParameterUseDefGraphNode;
  using ChildIteratorType = ParameterUseDefGraphNodeIterator;

  /// The "entry node" is a virtual node with a null Operation* that acts like
  /// it points to all the using operations.
  static NodeRef getEntryNode(const ParameterVerifier *verifier) {
    return ParameterUseDefGraphNode(verifier, nullptr);
  }

  static ChildIteratorType child_begin(NodeRef node) { return node.begin(); }
  static ChildIteratorType child_end(NodeRef node) { return node.end(); }
};
} // namespace llvm

/// Given a cycle in the operation parameter use graph, determine if it is an
/// error and diagnose it if so.  This returns success() in cases where the
/// cycle is tolerable.
static LogicalResult diagnoseCycle(ArrayRef<ParameterUseDefGraphNode> nodes,
                                   Operation *topLevelOp) {
  // Ignore self cycle in the top level op itself, this is because it is
  // defining parameters and using those parameters in its own argument
  // types.
  if (nodes.size() == 1 && nodes[0].getOperation() == topLevelOp)
    return success();

  // Build a set of the nodes in the SCC so we can do efficient queries.
  SmallPtrSet<Operation *, 4> opsInSCC;
  for (auto node : nodes)
    opsInSCC.insert(node.getOperation());

  // Emit the error on the container operation with notes indicating the
  // problem.
  auto diag =
      topLevelOp->emitError("invalid cyclic reference between operations "
                            "defining and using parameters");

  // An SCC may contain multiple different cyclic paths.  We diagnose the first
  // one we see by walking the graph - always staying within the SCC, until we
  // reach a node we've already seen.  Given this is an SCC, we know that we
  // will eventually reach one of the nodes in the path.
  SmallVector<ParameterUseDefGraphNodeIterator> path;
  SmallPtrSet<Operation *, 4> opsInPath;
  ParameterUseDefGraphNode nextNode = nodes.front();

  // Loop until we find a backrefence.
  while (opsInPath.insert(nextNode.getOperation()).second) {
    // Find an iterator from this node to another within this SCC.
    auto it = nextNode.begin();
    while (!opsInSCC.count((*it).getOperation())) {
      // Advance past edges to nodes outside the SCC.
      ++it;
      assert(it != nextNode.end() && "SCC means we should find an edge");
    }

    path.push_back(it);
    nextNode = *it;
  }

  // Okay, we found a path through the SCC that loops back to 'nextNode'.  Note
  // that it may not be a cycle though, because we may have found a path like
  // A->B->C->D->C.  In this case, we want to just diagnose C->D->C.  Handle
  // this by trimming off the beginning of the path until we find `C`.
  while (path.front().getSourceNode() != nextNode)
    path.erase(path.begin());

  // Okay, we found a path, diagnose it.
  for (auto &edge : path) {
    const char *nextDiag = ", which is defined by:";
    if (path.size() == 1)
      nextDiag = ", which is defined by itself";
    else if (&edge == &path.back())
      nextDiag = ", which is defined by the first operation";

    diag.attachNote(edge.getSourceNode().getOperation()->getLoc())
        << "this operation uses parameter " << edge.getParameterName()
        << nextDiag;
  }
  return failure();
}

/// Verify that the parameter use-def graph has a partial ordering, diagnosing
/// any cycles that are present.
LogicalResult
ParameterVerifier::checkParameterUseDefGraph(Operation *topLevelOp) {
  // Now that we've verified simple properties, check that there is a
  // defininable partial order between operations that define an use parameters.
  // We do this by using LLVM's SCC iterator to walk the graph imposed by these
  // nodes. It naturally provides a post-order traversal, makes it easy to balk
  // at cyclic references, and is non-recursive.
  SmallVector<Operation *, 16> newOrder;
  for (auto sccIt = llvm::scc_begin(this); !sccIt.isAtEnd(); ++sccIt) {
    // If this node has a cycle detected in it, then we have an unrecoverable
    // error.  Emit the error on the containiner with notes on every problematic
    // operation.
    if (sccIt.hasCycle() && failed(diagnoseCycle(*sccIt, topLevelOp)))
      return failure();

    assert(sccIt->size() == 1 &&
           "Should only have a single node in non-cyclic regions");
    // Remember the partial ordering we have.
    newOrder.push_back(sccIt->front().getOperation());
  }

  // Build a new `usersAndDeclarers` list in the correct order defined by the
  // SCC iterators post-order traversal.
  SmallVector<std::pair<Operation *, SmallVector<ParamDeclRefAttr>>, 8>
      usersAndDeclarers;
  usersAndDeclarers.reserve(parameters.usersAndDeclarers.size());
  for (Operation *op : newOrder) {
    if (op)
      usersAndDeclarers.push_back({op, std::move(getUsesForOperation(op))});
  }
  parameters.usersAndDeclarers = std::move(usersAndDeclarers);
  return success();
}
