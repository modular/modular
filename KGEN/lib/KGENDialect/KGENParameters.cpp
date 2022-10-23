//===----------------------------------------------------------------------===//
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
#include "KGEN/KGENDialect/KGENDeclInterface.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"

using namespace M;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// ParameterCollector
//===----------------------------------------------------------------------===//

namespace {
class ParameterCollector {
public:
  virtual ~ParameterCollector() = default;

  /// Scan the specified attribute and its recursive uses, diagnosing incorrect
  /// parameter declarations and collecting parameter uses into `uses`.
  void collectUsesFromAttr(Attribute attr,
                           SmallVectorImpl<ParamDeclRefAttr> &uses,
                           bool &hasConstExpr);

  /// Scan the specified type and its recursive uses, diagnosing incorrect
  /// parameter declarations and collecting parameter uses into `uses`.
  void collectUsesFromTypes(Type type, SmallVectorImpl<ParamDeclRefAttr> &uses,
                            bool &hasConstExpr);

private:
  /// The first time we encounter a SymbolConstantAttr, check to see if the
  /// declaration it refers to agrees with the value and parameter
  /// specification.
  virtual void verifySymbolConstantAttr(SymbolConstantAttr symbolConstant) = 0;

  /// When we encounter a RefType, check that its parameter bindings match
  /// the parameter declarations on the type declaration.
  virtual void verifyRefType(RefType typeDef) {}

  /// Attributes and types are memoized and exist in tree structures with reuse:
  /// naively scanning them can lead to exponential compile time behavior.  As
  /// such, we memoize the attributes and types we've already checked that we
  /// know have no parameters in them and whether the paramless attributes are
  /// constant parameter expressions.
  llvm::SmallDenseMap<Attribute, bool> parameterLessAttrs;
  llvm::SmallDenseMap<Type, bool> parameterLessTypes;
};
} // end anonymous namespace

/// Scan the specified attribute and its recursive uses, diagnosing incorrect
/// parameter declarations and collecting parameter uses.
void ParameterCollector::collectUsesFromAttr(
    Attribute attr, SmallVectorImpl<ParamDeclRefAttr> &uses,
    bool &hasConstExpr) {
  // If we have already scanned it and know that it has no parameters in it,
  // return early.
  if (!attr)
    return;
  if (auto it = parameterLessAttrs.find(attr); it != parameterLessAttrs.end()) {
    hasConstExpr |= it->second;
    return;
  }

  // Collect parameter references.
  if (auto paramRef = dyn_cast<ParamDeclRefAttr>(attr)) {
    uses.push_back(paramRef);
    return;
  }

  // Check any SymbolConstantAttr's we encounter.
  if (auto symbolConstant = dyn_cast<SymbolConstantAttr>(attr))
    verifySymbolConstantAttr(symbolConstant);

  // Save the number of nested parameters before recursing and check whether the
  // attribute has a nested constant expression.
  size_t oldSize = uses.size();
  bool hasNestedConstExpr = false;

  // Otherwise we haven't processed this, check the attribute's type if it has
  // one.
  if (auto typedAttr = dyn_cast<TypedAttr>(attr))
    collectUsesFromTypes(typedAttr.getType(), uses, hasNestedConstExpr);

  // Recursively check for any nested types/attributes, e.g. the elements of an
  // array attribute.
  if (auto itf = dyn_cast<mlir::SubElementAttrInterface>(attr)) {
    itf.walkImmediateSubElements(
        [&](Attribute attr) {
          collectUsesFromAttr(attr, uses, hasNestedConstExpr);
        },
        [&](Type type) {
          collectUsesFromTypes(type, uses, hasNestedConstExpr);
        });
  }

  // If the attribute had no uses, remember that so we don't have to re-scan it
  // in the future.
  if (oldSize == uses.size()) {
    // Check whether this is a parameterless expression.
    hasNestedConstExpr |= isa<ParamOperatorAttr>(attr);
    parameterLessAttrs.try_emplace(attr, hasNestedConstExpr);
    hasConstExpr |= hasNestedConstExpr;
  }
}

/// Scan the specified type and its recursive uses, diagnosing incorrect
/// parameter declarations and collecting parameter uses.
void ParameterCollector::collectUsesFromTypes(
    Type type, SmallVectorImpl<ParamDeclRefAttr> &uses, bool &hasConstExpr) {
  // Signature types with input parameters are effectively "isolated from above"
  // in that they may have their own local parameter declarations that are used
  // in their type signature, but they cannot "capture" parameters from the
  // enclosing context. As such, they are always considered "parameterless".
  if (auto signature = dyn_cast_if_present<SignatureType>(type))
    if (!signature.getInputParams().empty())
      return;

  // Ignore types we have already scanned.
  if (!type)
    return;
  if (auto it = parameterLessTypes.find(type); it != parameterLessTypes.end()) {
    hasConstExpr |= it->second;
    return;
  }

  // Check any RefType's we encounter.
  if (auto typeDef = dyn_cast<RefType>(type))
    verifyRefType(typeDef);

  // Save the number of nested parameters before recursing and check whether the
  // attribute has a nested constant expression.
  size_t oldSize = uses.size();
  bool hasNestedConstExpr = false;

  // Recursively check for any nested types, e.g. the input/outputs of a
  // function type, types like !pop.scalar<ty> etc.
  if (auto itf = dyn_cast<mlir::SubElementTypeInterface>(type)) {
    itf.walkImmediateSubElements(
        [&](Attribute attr) {
          collectUsesFromAttr(attr, uses, hasNestedConstExpr);
        },
        [&](Type type) {
          collectUsesFromTypes(type, uses, hasNestedConstExpr);
        });
  }

  // If the type had parameter uses or constant expressions, don't consider it
  // "parameterless".  We want other operations using the same type to record
  // the uses as well.
  if (oldSize == uses.size()) {
    parameterLessTypes.try_emplace(type, hasNestedConstExpr);
    hasConstExpr |= hasNestedConstExpr;
  }
}

//===----------------------------------------------------------------------===//
// SignatureType Verification
//===----------------------------------------------------------------------===//

ErrorOrSuccess SignatureType::checkSelfContained() {
  // Check the input parameters for conflicts.
  SmallDenseMap<StringAttr, Type> paramsMap;
  for (auto inputParam : getInputParams()) {
    auto &entry = paramsMap[inputParam.getName()];
    if (entry)
      return Error("signature parameter \"" + inputParam.getName().strref() +
                   "\" redefined");
    entry = inputParam.getType();
  }

  // If the signature has no input parameters, then it isn't "isolated" within
  // itself, it may use contextual types.  The normal parameter scanner will
  // handle it.
  if (getInputParams().empty())
    return success();

  // Otherwise, we need to check that any input/output value types only use
  // parameters defined in the signature itself.
  bool hadSymbolConstantReferences = false;
  struct SignatureTypeCollector : public ParameterCollector {
    SignatureTypeCollector(bool &hadSymbolConstantReferences)
        : hadSymbolConstantReferences(hadSymbolConstantReferences) {}
    void verifySymbolConstantAttr(SymbolConstantAttr symbolConstant) override {
      hadSymbolConstantReferences = true;
    }
    bool &hadSymbolConstantReferences;
  } parameterCollector(hadSymbolConstantReferences);

  // Collect all the parameter references from the function type in the
  // signature.
  SmallVector<ParamDeclRefAttr> uses;
  bool hasConstExpr;
  parameterCollector.collectUsesFromTypes(getValues(), uses, hasConstExpr);

  // Reject any SymbolConstantAttr's, they cannot exist in a signature.  This
  // structurally cannot exist, but this is defensive code in case something
  // changes in the future.
  if (hadSymbolConstantReferences)
    return Error("signature type cannot use an @symbol reference");

  // Check that each of the uses is to a defined input parameter.
  for (ParamDeclRefAttr use : uses) {
    auto &entry = paramsMap[use.getName()];
    if (!entry)
      return Error("\"" + use.getName().strref() +
                   "\" parameter not defined in signature");
    if (entry != use.getType())
      return Error("use of \"" + use.getName().strref() +
                   "\" with incorrect type in signature");
  }

  // Otherwise we succeed.
  return success();
}

//===----------------------------------------------------------------------===//
// DeclParameterVerifier
//===----------------------------------------------------------------------===//

namespace {
struct DeclParameterVerifier final : public ParameterCollector {
  DeclParameterVerifier(KGENDeclInterface topLevelDeclOp,
                        ParameterDeclsAndUses &parameters,
                        SymbolTableCollection *symbolTables)
      : topLevelDeclOp(topLevelDeclOp), parameters(parameters),
        symbolTables(symbolTables) {}

  /// Walk the operation and all the operations in its body to find the
  /// definitions and uses of parameters.  This diagnoses and rejects parameter
  /// definitions in invalid positions as well.
  LogicalResult collectParameterDefsAndUses();

  /// Once all the defs and uses of parameters are collected, verify that the
  /// uses are correct.
  LogicalResult checkParameterUses();

  /// Reorder the declsAndUses list to be in correct top-down order.  This also
  /// verifies that the parameter use-def graph has a partial ordering,
  /// diagnosing any cycles that are present.
  LogicalResult checkAndReorderParameterUseDefGraph();

  /// Return the set of parameter uses for the specified operation.
  SmallVector<ParamDeclRefAttr> &getUsesForOperation(Operation *op) const {
    auto it = opIndexInUses.find(op);
    assert(op && it != opIndexInUses.end());
    return parameters.usersAndDeclarers[it->second].second;
  }

  void verifySymbolConstantAttr(SymbolConstantAttr symbolConstant) override;

  void verifyRefType(RefType typeDef) override;

  // This is the top level declaration that we're analyzing.
  KGENDeclInterface const topLevelDeclOp;

  /// This is the parameter information that we're building.
  ParameterDeclsAndUses &parameters;

  /// If non-null, this contains a set of symbol tables that we can use to
  /// verify the validity of SymbolConstantAttr's.
  SymbolTableCollection *symbolTables;

  /// This is the current operation being scanned during the attribute/type
  /// collection phase.
  Optional<Location> curLocationCollecting;

private:
  /// This is set to true if we find a problem during the collect phase.
  bool hadError = false;

  /// A single operation may use multiple parameter declarations, either
  /// directly or through types on attributes and SSA operands/results.  This
  /// keeps track of all of the uses that happen anywhere within an operation.
  DenseMap<Operation *, size_t> opIndexInUses;
};
} // namespace

/// Walk the operation and all the operations in its body to find the
/// definitions and uses of parameters.  This diagnoses and rejects parameter
/// definitions in invalid positions as well.
LogicalResult DeclParameterVerifier::collectParameterDefsAndUses() {
  topLevelDeclOp->walk<mlir::WalkOrder::PreOrder>([&](Operation *bodyOp) {
    // Walk over nested operations isolated from above.
    if (bodyOp != topLevelDeclOp &&
        bodyOp->hasTrait<OpTrait::IsIsolatedFromAbove>())
      return WalkResult::skip();

    ParamDeclArrayAttr paramDeclsAttr;
    SmallVector<ParamDeclRefAttr> uses;
    bool hasConstExpr;

    curLocationCollecting = bodyOp->getLoc();

    // Scan all the attributes and types to look for uses of parameters.  We let
    // the walker scan the region hierarchy.
    for (const NamedAttribute &namedAttr : bodyOp->getAttrs()) {
      // Scan the attribute tree looking or parameter uses.
      collectUsesFromAttr(namedAttr.getValue(), uses, hasConstExpr);

      // We handle the `paramDecls` attribute specially, remember it for
      // below.
      if (namedAttr.getName().strref() == "paramDecls") {
        paramDeclsAttr = dyn_cast<ParamDeclArrayAttr>(namedAttr.getValue());
        if (!paramDeclsAttr) {
          bodyOp->emitError("paramDecls attribute should be an array ")
              << namedAttr.getValue();
          hadError = true;
          return WalkResult::advance();
        }
      }
    }

    // Check the types of results to find any parameters embedded in their
    // types.  We don't have to check operands because they are always checked
    // when being defined.
    for (Type type : bodyOp->getResultTypes())
      collectUsesFromTypes(type, uses, hasConstExpr);

    // Scan the region list if present.  The walker will automatically recurse
    // for us, but we have to check the block arguments.
    if (bodyOp->getNumRegions()) { // Microoptimization: getRegions() is slow.
      for (auto &region : bodyOp->getRegions()) {
        for (auto &block : region)
          for (Value arg : block.getArguments())
            collectUsesFromTypes(arg.getType(), uses, hasConstExpr);
      }
    }

    // We're done collecting from this operation.
    curLocationCollecting = None;

    // If this operation had any parameter uses or decls, remember them.
    if (!uses.empty() || paramDeclsAttr) {
      parameters.usersAndDeclarers.push_back({bodyOp, std::move(uses)});
    } else if (hasConstExpr) {
      // If this operation contains only constant expressions, remember it.
      parameters.constExprOps.push_back(bodyOp);
    }

    // Ok, check parameter declarations if present.
    if (!paramDeclsAttr)
      return WalkResult::advance();

    for (ParamDeclAttr param : paramDeclsAttr) {
      // We cannot have any redefinitions.
      auto &opAndDeclAttr = parameters.decls[param.getName()];
      if (opAndDeclAttr.first) {
        auto diag = bodyOp->emitError("redeclaration of parameter ")
                    << param.getName();
        diag.attachNote(opAndDeclAttr.first->getLoc())
            << "previous declaration here";
        hadError = true;
        return WalkResult::advance();
      }

      opAndDeclAttr = {bodyOp, param};
    }

    return WalkResult::advance();
  });
  return failure(hadError);
}

/// Once all the defs and uses of parameters are collected, verify that the
/// uses are correct.
LogicalResult DeclParameterVerifier::checkParameterUses() {
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

/// The first time we encounter a SymbolConstantAttr, check to see if the
/// declaration it refers to agrees with the value and parameter
/// specification.
void DeclParameterVerifier::verifySymbolConstantAttr(
    SymbolConstantAttr symbolConstant) {
  // We only check this during the op verification phase.
  if (!symbolTables)
    return;

  auto symbol = symbolConstant.getSymbol();
  auto decl = dyn_cast_if_present<KGENDeclInterface>(
      symbolTables->lookupNearestSymbolFrom(topLevelDeclOp, symbol));

  if (!decl) {
    hadError = true;
    emitError(curLocationCollecting.value(), "'")
        << symbol << "' does not reference a KGEN declaration";
    return;
  }

  auto declSignature = decl.getSignature();

  // If this SymbolConstant binds the parameters for the symbol, then remap its
  // signature to include the substitutions.
  if (!symbolConstant.getParamValues().empty()) {
    auto result = declSignature.getSpecializedSignature(
        symbolConstant.getParamValues(), [&]() {
          hadError = true;
          return emitError(curLocationCollecting.value());
        });
    if (!result)
      return;

    // The signature we just got back has all the parameter we just substituted
    // in as part of the signature.  These are now fully bound, so we don't need
    // them anymore.
    declSignature =
        SignatureType::get(ParamDeclArrayAttr::get(result.getContext(), {}),
                           result.getResultParamTypes(), result.getValues());
  }

  auto symbolSignature = cast<SignatureType>(symbolConstant.getType());

  // Parameter types match exactly.  We could support higher order rebinding
  // if there is a need.
  SmallString<32> paramName("@");
  paramName.append(symbol.getLeafReference());
  if (failed(verifyDeclSignaturesMatch(
          "symbol use", symbolSignature, curLocationCollecting.value(),
          paramName.c_str(), declSignature, decl->getLoc())))
    hadError = true;
}

/// The first time we encounter a RefType, check to see if its parameter
/// bindings agrees with the parameter declarations of the referred type
/// dedclaration.
void DeclParameterVerifier::verifyRefType(RefType typeDef) {
  // We only check this during the op verification phase.
  if (!symbolTables)
    return;

  auto decl = dyn_cast_if_present<KGENDeclInterface>(
      symbolTables->lookupNearestSymbolFrom(topLevelDeclOp, typeDef.getName()));
  if (!decl) {
    hadError = true;
    emitError(curLocationCollecting.value())
        << typeDef.getName() << " does not reference a KGEN type declaration";
    return;
  }

  SmallString<32> paramName("@");
  paramName.append(typeDef.getName().getLeafReference());
  if (failed(verifyParamDeclsMatch(
          "typedef symbol use",
          llvm::to_vector(llvm::map_range(
              typeDef.getParamValues(),
              [](ParamBindAttr value) { return value.getDecl(); })),
          curLocationCollecting.value(), paramName.c_str(),
          decl.getParamDeclsAttr(), decl.getLoc())))
    hadError = true;
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
  ParameterUseDefGraphNode(const DeclParameterVerifier *verifier, Operation *op)
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
  const DeclParameterVerifier *getVerifier() const { return verifier; }

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
  const DeclParameterVerifier *verifier;
  Operation *op;
};
} // namespace

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
} // namespace

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

/// The DeclParameterVerifier graph is defined with `ParameterUseDefGraphNode`
/// nodes and `ParameterUseDefGraphNodeIterator` iterators.
namespace llvm {
template <>
struct GraphTraits<DeclParameterVerifier *> {
  using NodeRef = ParameterUseDefGraphNode;
  using ChildIteratorType = ParameterUseDefGraphNodeIterator;

  /// The "entry node" is a virtual node with a null Operation* that acts like
  /// it points to all the using operations.
  static NodeRef getEntryNode(const DeclParameterVerifier *verifier) {
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

/// Reorder the declsAndUses list to be in correct top-down order.  This also
/// verifies that the parameter use-def graph has a partial ordering,
/// diagnosing any cycles that are present.
LogicalResult DeclParameterVerifier::checkAndReorderParameterUseDefGraph() {
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
    if (sccIt.hasCycle() && failed(diagnoseCycle(*sccIt, topLevelDeclOp)))
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

//===----------------------------------------------------------------------===//
// Main Entrypoint
//===----------------------------------------------------------------------===//

/// Collect information about the parameter definitions and uses in the
/// specified operation.  This assumes the IR is in a valid state.
ParameterDeclsAndUses ParameterDeclsAndUses::calculate(KGENDeclInterface op) {
  auto result = calculateAndPotentiallyVerify(op, nullptr);
  assert(succeeded(result) && "IR should be legal here!");
  return std::move(result.value());
}

/// Check deep invariants for a func/generator decl body, used by the
/// verifiers for these operations.  If a problem is detected, this emits an
/// error and returns failure.
FailureOr<ParameterDeclsAndUses>
ParameterDeclsAndUses::calculateAndVerify(KGENDeclInterface op,
                                          SymbolTableCollection &symbolTables) {
  return calculateAndPotentiallyVerify(op, &symbolTables);
}

/// Collect information about the parameter definitions and uses in the
/// specified operation.
///
/// If the SymbolTableCollection is non-null, check deep invariants for a
/// func/generator decl body, used by the verifiers for these operations.  If a
/// problem is detected, this emits an error and returns failure.
FailureOr<ParameterDeclsAndUses>
ParameterDeclsAndUses::calculateAndPotentiallyVerify(
    KGENDeclInterface topLevelOp, SymbolTableCollection *symbolTables) {
  ParameterDeclsAndUses result;
  DeclParameterVerifier verifier(topLevelOp, result, symbolTables);

  // Start by doing a pass over the operation and all the operations in its
  // body to find the definitions and uses of parameters.
  if (failed(verifier.collectParameterDefsAndUses()) ||
      // Next, now that we know the set of parameters we have to process,
      // verify that the uses match up.
      failed(verifier.checkParameterUses()) ||
      // Verify that there are no cycles in the graph.
      failed(verifier.checkAndReorderParameterUseDefGraph()))
    return failure();

  return std::move(result);
}

SmallVector<Operation *> ParameterDeclsAndUses::getParametricOps() const {
  SmallVector<Operation *> result;
  result.reserve(usersAndDeclarers.size() + constExprOps.size());
  for (const auto &elt : usersAndDeclarers)
    result.push_back(elt.first);
  llvm::append_range(result, constExprOps);
  return result;
}
