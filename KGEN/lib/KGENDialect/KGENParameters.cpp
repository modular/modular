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
#include "KGEN/KGENDialect/KGENInterfaces.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
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
  void collectUsesFromType(Type type, SmallVectorImpl<ParamDeclRefAttr> &uses,
                           bool &hasConstExpr);

private:
  void collectUsesFromTypesImpl(Type type,
                                SmallVectorImpl<ParamDeclRefAttr> &uses,
                                bool &hasConstExpr);

  /// The first time we encounter an attribute with a reference to an
  /// out-of-line declaration, verify it.
  virtual void verifyRefAttr(DeclRefAttrInterface refAttr) {}

  /// When we encounter a DeclRefType, check that its parameter bindings match
  /// the parameter declarations on the type declaration.
  virtual void verifyRefType(DeclRefType refType) {}

  /// Verify a use of a parameter declared in a nested scope.
  virtual void verifyNestedParameterUse(ParamDeclAttr decl,
                                        ParamDeclRefAttr use) {}

  /// Report that a nested parameter declaration is a duplicate.
  virtual void reportDuplicateNestedDecl(ParamDeclAttr decl) {}

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
    collectUsesFromType(paramRef.getType(), uses, hasConstExpr);
    uses.push_back(paramRef);
    return;
  }

  // Check any SymbolConstantAttr's we encounter.
  if (auto ref = dyn_cast<DeclRefAttrInterface>(attr))
    verifyRefAttr(ref);

  // Save the number of nested parameters before recursing and check whether the
  // attribute has a nested constant expression.
  size_t oldSize = uses.size();
  bool hasNestedConstExpr = false;

  // Otherwise we haven't processed this, check the attribute's type if it has
  // one.
  if (auto typedAttr = dyn_cast<TypedAttr>(attr))
    collectUsesFromType(typedAttr.getType(), uses, hasNestedConstExpr);

  // Recursively check for any nested types/attributes, e.g. the elements of an
  // array attribute.
  attr.walkImmediateSubElements(
      [&](Attribute attr) {
        collectUsesFromAttr(attr, uses, hasNestedConstExpr);
      },
      [&](Type type) { collectUsesFromType(type, uses, hasNestedConstExpr); });

  // If the attribute had no uses, remember that so we don't have to re-scan it
  // in the future.
  if (oldSize == uses.size()) {
    // Check whether this is a parameterless expression.
    hasNestedConstExpr |= isa<ParamOperatorAttr>(attr);
    parameterLessAttrs.try_emplace(attr, hasNestedConstExpr);
    hasConstExpr |= hasNestedConstExpr;
  }
}

void ParameterCollector::collectUsesFromType(
    Type type, SmallVectorImpl<ParamDeclRefAttr> &uses, bool &hasConstExpr) {
  // Signature types define nested parameters.
  if (auto sig = dyn_cast<SignatureType>(type)) {
    SmallPtrSet<StringAttr, 4> nestedParams;
    for (ParamDeclAttr param : llvm::concat<const ParamDeclAttr>(
             sig.getInputParams(), sig.getResultParams()))
      if (!nestedParams.insert(param.getName()).second)
        return reportDuplicateNestedDecl(param);
    SmallVector<ParamDeclRefAttr> nestedUses;
    collectUsesFromTypesImpl(type, nestedUses, hasConstExpr);
    // Filter the nested uses and determine which belong to the higher scope.
    for (ParamDeclRefAttr nestedUse : nestedUses)
      if (!nestedParams.contains(nestedUse.getName()))
        uses.push_back(nestedUse);
    return;
  }
  return collectUsesFromTypesImpl(type, uses, hasConstExpr);
}

void ParameterCollector::collectUsesFromTypesImpl(
    Type type, SmallVectorImpl<ParamDeclRefAttr> &uses, bool &hasConstExpr) {
  // Ignore types we have already scanned.
  if (!type)
    return;
  if (auto it = parameterLessTypes.find(type); it != parameterLessTypes.end()) {
    hasConstExpr |= it->second;
    return;
  }

  // Check any DeclRefType's we encounter.
  if (auto refType = dyn_cast<DeclRefType>(type))
    verifyRefType(refType);

  // Save the number of nested parameters before recursing and check whether the
  // attribute has a nested constant expression.
  size_t oldSize = uses.size();
  bool hasNestedConstExpr = false;

  // Recursively check for any nested types, e.g. the input/outputs of a
  // function type, types like !pop.scalar<ty> etc.
  type.walkImmediateSubElements(
      [&](Attribute attr) {
        collectUsesFromAttr(attr, uses, hasNestedConstExpr);
      },
      [&](Type type) { collectUsesFromType(type, uses, hasNestedConstExpr); });

  // If the type had parameter uses or constant expressions, don't consider it
  // "parameterless".  We want other operations using the same type to record
  // the uses as well.
  if (oldSize == uses.size()) {
    parameterLessTypes.try_emplace(type, hasNestedConstExpr);
    hasConstExpr |= hasNestedConstExpr;
  }
}

//===----------------------------------------------------------------------===//
// VerifyingParameterCollector
//===----------------------------------------------------------------------===//

namespace {
class VerifyingParameterCollector : public ParameterCollector {
public:
  VerifyingParameterCollector(ModuleOp module, SymbolTableCollection *symtab)
      : module(module), symtab(symtab) {}

  /// The first time we encounter an attribute with a reference to an
  /// out-of-line declaration, verify it.
  void verifyRefAttr(DeclRefAttrInterface refAttr) override;

  /// The first time we encounter a DeclRefType, check to see if its parameter
  /// bindings agrees with the parameter declarations of the referred type
  /// dedclaration.
  void verifyRefType(DeclRefType refType) override;

  /// Verify use of a nested parameter declaration. Emit an error if it fails.
  void verifyNestedParameterUse(ParamDeclAttr decl,
                                ParamDeclRefAttr use) override;

  /// Report a duplicate nested parameter declaration.
  void reportDuplicateNestedDecl(ParamDeclAttr decl) override;

  /// Whether a verification error occurred.
  bool hadError = false;
  /// The current operation where we are collecting parameters.
  Operation *op;

private:
  /// The module in which to lookup symbol references.
  ModuleOp module;
  /// The symbol to use to verify symbol references.
  SymbolTableCollection *symtab;
};
} // namespace

void VerifyingParameterCollector::verifyRefAttr(DeclRefAttrInterface refAttr) {
  // We only check this during the op verification phase.
  if (!symtab)
    return;
  if (failed(refAttr.verifySymbolUses(module, *symtab, op->getLoc())))
    hadError = true;
}

void VerifyingParameterCollector::verifyRefType(DeclRefType refType) {
  // We only check this during the op verification phase.
  if (!symtab)
    return;

  auto decl = dyn_cast_or_null<DeclInterface>(
      symtab->lookupSymbolIn(module, refType.getSymbol()));
  if (!decl) {
    hadError = true;
    emitError(op->getLoc())
        << refType.getSymbol() << " does not reference a KGEN type declaration";
    return;
  }

  // We have to specialize the type's parameter decls.
  ParameterEvaluator evaluator;
  for (auto [value, decl] :
       llvm::zip(refType.getParamValues(), decl.getInputParamDecls()))
    evaluator.setParameterValue(decl, value.getValue());
  SmallVector<ParamDeclAttr> specializedDecls;
  specializedDecls.reserve(refType.getParamValues().size());
  for (ParamDeclAttr decl : decl.getInputParamDecls())
    specializedDecls.push_back(
        cast<ParamDeclAttr>(evaluator.getReboundAttribute(decl)));

  SmallString<32> paramName("@");
  paramName.append(refType.getSymbol().getLeafReference());
  if (failed(verifyParamDeclsMatch("input parameter",
                                   "!kgen.declref symbol use",
                                   refType.getParamValues(), op->getLoc(),
                                   paramName, specializedDecls, decl.getLoc())))
    hadError = true;
}

void VerifyingParameterCollector::verifyNestedParameterUse(
    ParamDeclAttr decl, ParamDeclRefAttr use) {
  if (decl.getType() == use.getType())
    return;
  (mlir::emitError(op->getLoc(), "use of nested parameter ")
   << decl.getName() << " with incorrect type " << use.getType())
          .attachNote()
      << "parameter defined with type " << decl.getType();
  hadError = true;
}

void VerifyingParameterCollector::reportDuplicateNestedDecl(
    ParamDeclAttr decl) {
  mlir::emitError(op->getLoc(), "nested parameter ")
      << decl.getName() << " redefined";
  hadError = true;
}

//===----------------------------------------------------------------------===//
// ParameterUseDefGraph Implementation
//===----------------------------------------------------------------------===//

namespace {
struct ParameterUseDefGraphNodeIterator;

/// Each node in the parameter use-def graph is a parameter definition. An
/// outgoing edge represents a use of another parameter in the definition of the
/// parameter, and incoming edges are from other parameters that use this one in
/// their declarations.
///
/// A null parameter indicates a virtual root node that points to all other
/// nodes.
struct ParameterUseDefGraphNode {
  ParameterUseDefGraph *g;
  StringAttr param;

  /// Enable nodes to be check for equality.
  bool operator==(const ParameterUseDefGraphNode &rhs) const {
    return param == rhs.param;
  }
  bool operator!=(const ParameterUseDefGraphNode &rhs) const {
    return param != rhs.param;
  }

  ParameterUseDefGraphNodeIterator begin() const;
  ParameterUseDefGraphNodeIterator end() const;
};

/// An iterator for the parameter use-def graph. This class iterates through
/// the uses of a parameter.
struct ParameterUseDefGraphNodeIterator
    : public llvm::iterator_facade_base<ParameterUseDefGraphNodeIterator,
                                        std::forward_iterator_tag,
                                        ParameterUseDefGraphNode> {
  ParameterUseDefGraphNodeIterator(ParameterUseDefGraphNode node,
                                   size_t useNumber)
      : node(node), useNumber(useNumber) {}

  ParameterUseDefGraphNode node;
  size_t useNumber;

  /// Enable iterators to be checked for equality.
  bool operator==(const ParameterUseDefGraphNodeIterator &rhs) const {
    return node == rhs.node && useNumber == rhs.useNumber;
  }

  /// Enable iterators to be incremented.
  ParameterUseDefGraphNodeIterator operator++() {
    ++useNumber;
    return *this;
  }
  ParameterUseDefGraphNodeIterator operator++(int) {
    ParameterUseDefGraphNodeIterator tmp = *this;
    ++*this;
    return tmp;
  }

  /// For the virtual root node, deference into the parameter definition. For
  /// regular nodes, deference to the node that defines the used parameter.
  ParameterUseDefGraphNode operator*() const {
    if (!node.param)
      return {node.g, node.g->params[useNumber]};
    return {node.g, node.g->defs[node.param].uses[useNumber].getName()};
  }
};

ParameterUseDefGraphNodeIterator ParameterUseDefGraphNode::begin() const {
  return {*this, 0};
}

ParameterUseDefGraphNodeIterator ParameterUseDefGraphNode::end() const {
  // For the virtual root node, the end iterator is the last parameter.
  if (!param)
    return {*this, g->params.size()};
  // Do not traverse through to parameters in higher scopes.
  if (!g->scope->isAncestor(g->decls[param].scope))
    return begin();
  // If the used parameter has no definition, this is a leaf node.
  auto it = g->defs.find(param);
  if (it == g->defs.end())
    return begin();
  // The end iterator is the last use.
  return {*this, it->second.uses.size()};
}
} // namespace

namespace llvm {
template <>
struct DenseMapInfo<ParameterUseDefGraphNode> {
  static inline ParameterUseDefGraphNode getEmptyKey() {
    return {nullptr, DenseMapInfo<StringAttr>::getEmptyKey()};
  }
  static inline ParameterUseDefGraphNode getTombstoneKey() {
    return {nullptr, DenseMapInfo<StringAttr>::getTombstoneKey()};
  }
  static unsigned getHashValue(const ParameterUseDefGraphNode &node) {
    return DenseMapInfo<StringAttr>::getHashValue(node.param);
  }
  static bool isEqual(const ParameterUseDefGraphNode &lhs,
                      const ParameterUseDefGraphNode &rhs) {
    return lhs == rhs;
  }
};

template <>
struct GraphTraits<ParameterUseDefGraph *> {
  using NodeRef = ParameterUseDefGraphNode;
  using ChildIteratorType = ParameterUseDefGraphNodeIterator;

  static NodeRef getEntryNode(ParameterUseDefGraph *g) { return {g, nullptr}; }

  static ChildIteratorType child_begin(NodeRef node) { return node.begin(); }
  static ChildIteratorType child_end(NodeRef node) { return node.end(); }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// ParameterUseDefGraph
//===----------------------------------------------------------------------===//

void impl::scanAllAttrsAndTypes(Operation *op,
                                function_ref<void(Attribute)> scanAttr,
                                function_ref<void(Type)> scanType) {
  llvm::for_each(op->getOperandTypes(), scanType);
  llvm::for_each(op->getResultTypes(), scanType);
  for (Region &region : op->getRegions())
    for (Block &block : region)
      llvm::for_each(block.getArgumentTypes(), scanType);

  // FIXME(#7743): Scan locations too when the elaborator has been updated to
  // handle the new parameter use-def graph.
  scanAttr(op->getAttrDictionary());
}

/// Collect parameter uses from the operation. If there are any uses or
/// otherwise unresolved parameter operators, indicate that the operation is
/// parametric.
static void collectUses(ParameterUseDefGraph &g, VerifyingParameterCollector &c,
                        Operation *op, bool isDecl) {
  // Track whether parameter uses or expressions were found.
  bool hasConstExpr = false;
  SmallVector<ParamDeclRefAttr> uses;

  auto scanAttr = [&](Attribute attr) {
    c.collectUsesFromAttr(attr, uses, hasConstExpr);
  };
  auto scanType = [&](Type type) {
    c.collectUsesFromType(type, uses, hasConstExpr);
  };

  auto itf = dyn_cast<ParamOpInterface>(op);
  if (itf) {
    // If the parameter operation is the containing declaration, collect only
    // uses below the defined scope.
    if (op == g.scope->getParentOp())
      itf.collectParameterUsesBelow(scanAttr, scanType);
    else
      itf.collectParameterUses(scanAttr, scanType);

    // Otherwise, scan all attributes and types if the operation is not a
    // declaration or it is the containing declaration.
  } else if (!isa<DeclInterface>(op) || op == g.scope->getParentOp()) {
    impl::scanAllAttrsAndTypes(op, scanAttr, scanType);
  }

  // If the operation is parametric, add it to the list.
  if (hasConstExpr || !uses.empty()) {
    if (!isDecl)
      g.paramOps.push_back(op);
    g.opUses[op] = std::move(uses);
  } else if (!isDecl && itf && itf.isImplicitlyParametric()) {
    // Track implicitly parametric operations only when they don't already
    // declare parameters.
    g.paramOps.push_back(op);
  }
}

static LogicalResult recordDecl(ParameterUseDefGraph &g, ParamDeclAttr decl,
                                Operation *op, Region &scope) {
  ParamDeclaration &paramDecl = g.decls[decl.getName()];

  // If this parameter has already been declared in an operation in the same
  // scope, we have an error.
  if (paramDecl.scope && scope.isAncestor(paramDecl.scope)) {
    return (op->emitError("redeclaration of parameter ") << decl.getName())
               .attachNote(paramDecl.declOp->getLoc())
           << "previous declaration here";
  }

  // Record the new declaration.
  paramDecl.declOp = op;
  paramDecl.type = decl.getType();
  paramDecl.scope = &scope;
  return success();
}

static ParamDefinition &recordDef(ParameterUseDefGraph &g, ParamDeclAttr decl,
                                  Operation *op) {
  ParamDefinition &paramDef = g.defs[decl.getName()];
  assert(!paramDef.defOp && "parameter redefinitions are not possible");
  g.params.push_back(decl.getName());
  paramDef.defOp = op;
  return paramDef;
}

/// Cycles detected in the definition of a parameter are always forbidden. When
/// that occurs, emit a nice error detailing the cycle.
static void emitCycleError(ParameterUseDefGraph &g,
                           ArrayRef<ParameterUseDefGraphNode> nodes) {
  // Build a set of the nodes in the SCC so we can do efficient queries.
  SmallPtrSet<StringAttr, 4> paramsInSCC;
  for (const ParameterUseDefGraphNode &node : nodes)
    paramsInSCC.insert(node.param);

  // Emit the error on the container operation with notes indicating the
  // problem.
  InFlightDiagnostic diag = g.scope->getParentOp()->emitError(
      "cyclic reference between expressions defining and using parameters");

  // An SCC may contain multiple different cyclic paths.  We diagnose the first
  // one we see by walking the graph - always staying within the SCC, until we
  // reach a node we've already seen.  Given this is an SCC, we know that we
  // will eventually reach one of the nodes in the path.
  SmallVector<ParameterUseDefGraphNodeIterator> path;
  SmallPtrSet<StringAttr, 4> paramsInPath;
  ParameterUseDefGraphNode nextNode = nodes.front();

  // Loop until we find a backrefence.
  while (paramsInPath.insert(nextNode.param).second) {
    // Find an iterator from this node to another within this SCC.
    ParameterUseDefGraphNodeIterator it = nextNode.begin();
    while (!paramsInSCC.contains((*it).param)) {
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
  while (path.front().node != nextNode)
    path.erase(path.begin());

  // Okay, we found a path, diagnose it.
  for (ParameterUseDefGraphNodeIterator &edge : path) {
    const char *nextDiag = ", which references the expression:";
    if (path.size() == 1)
      nextDiag = ", which references itself";
    else if (&edge == &path.back())
      nextDiag = ", which references the first expression";

    StringAttr defParam = edge.node.param;
    diag.attachNote(g.defs[defParam].defOp->getLoc())
        << "parameter " << defParam << " is defined here" << nextDiag;
  }
}

LogicalResult
ParameterUseDefGraph::calculateOrVerify(ModuleOp module,
                                        SymbolTableCollection *symtab) {
  // Defer the processing of the use-def node for region declarations until
  // after nested scopes have been analyzed.
  SmallVector<std::pair<ParamDeclAttr, SmallVector<Region *, 0>>> regionValues;
  // The parameter collector to use.
  VerifyingParameterCollector c(module, symtab);

  auto processOp = [&](Operation *op) -> WalkResult {
    // Set the operation for which we are collecting parameters. It will be used
    // to report errors.
    c.op = op;

    // Track whether the operation declares parameters. Operations that declare
    // parameters are treated differently than those that simply use parameters.
    bool isDecl = false;

    // Check if this operation defines a parameter scope.
    auto result = WalkResult::advance();
    if (auto decl = dyn_cast<DeclInterface>(op)) {
      // Check if this is a nested scope.
      if (scope->getParentOp() != decl) {
        // Walk over nested scopes. Defer processing of nested scopes until
        // after this scope has been analyzed.
        for (Region &r : decl->getRegions())
          nestedDecls.push_back(&r);
        result = WalkResult::skip();
      } else {
        // Record parameter declarations for the top-level declaration.
        auto recordDeclWrapper = [&](ParamDeclAttr decl) -> LogicalResult {
          isDecl = true;
          return recordDecl(*this, decl, op, *scope);
        };
        // A declaration declares input parameters but does not define them.
        for (ParamDeclAttr decl : decl.getInputParamDecls())
          if (failed(recordDeclWrapper(decl)))
            return failure();
        if (auto func = dyn_cast<FuncInterface>(op)) {
          // A function declares result parameters but does not define them.
          for (ParamDeclAttr decl : func.getResultParams())
            if (failed(recordDeclWrapper(decl)))
              return failure();
        }
      }
    }

    // Check if this operation implements the parametric operation interface.
    if (auto itf = dyn_cast<ParamOpInterface>(op);
        itf && itf != scope->getParentOp()) {
      // Check declarations.
      bool hadError = false;
      itf.walkDeclarations([&](ParamDeclAttr decl) {
        isDecl = true;
        if (failed(recordDecl(*this, decl, op, *scope)))
          hadError = true;
      });
      if (hadError)
        return WalkResult::interrupt();

      // Check definitions.
      ssize_t index = 0;
      itf.walkDefinitions([&](ParamDeclAttr decl, const ParamDefValue &value) {
        ParamDefinition &def = recordDef(*this, decl, op);
        def.index = index++;
        bool unused;
        for (Attribute expr : value.exprs)
          c.collectUsesFromAttr(expr, def.uses, unused);
        // If the definition of this parameter depends on a region, defer
        // processing of the nested region uses.
        if (!value.regions.empty())
          regionValues.emplace_back(decl, value.regions);
      });
    }

    // Collect parameter uses from this operation.
    collectUses(*this, c, op, isDecl);
    return result;
  };

  // Process the scope's parent op - don't recurse because the parent op might
  // have multiple regions.
  processOp(scope->getParentOp());

  // Now walk the scope and not sibling regions!
  WalkResult result = scope->walk<mlir::WalkOrder::PreOrder>(processOp);
  if (result.wasInterrupted())
    return failure();

  // Check the validity of all parameter references.
  for (auto &[op, uses] : opUses) {
    for (ParamDeclRefAttr use : uses) {
      auto it = decls.find(use.getName());
      // Ensure that the use refers to a parameter that was declared.
      if (it == decls.end())
        return op->emitOpError("invalid use of parameter with no declaration ")
               << use.getName();

      // Check that the type of the parameter references matches the type of its
      // declaration.
      if (symtab && it->second.type != use.getType()) {
        return (op->emitOpError("reference to parameter ")
                << use.getName() << " with incorrect type " << use.getType())
                   .attachNote(it->second.declOp->getLoc())
               << "parameter defined with type " << it->second.type;
      }

      // If the declaration of the parameter is outside the current scope,
      // indicate this as a parameter use from above.
      if (!scope->isAncestor(it->second.scope))
        usesFromAbove.insert(use);
    }
  }

  // If an error was encountered while collecting parameters, bail out here.
  if (c.hadError)
    return failure();

  // Process all nested scopes.
  for (Region *nestedScope : nestedDecls) {
    ParameterUseDefGraph nested(*nestedScope);
    // Propagate the current declarations into the nested scope.
    nested.decls = decls;
    if (failed(nested.calculateOrVerify(module, symtab)))
      return failure();

    // If there were no uses from above, notify the nested declaration that it
    // is isolated. Do not do this during verification.
    if (nested.usesFromAbove.empty()) {
      auto decl = cast<DeclInterface>(nestedScope->getParentOp());
      decl.notifyKnownIsolatedFromAbove(nestedScope->getRegionNumber());
    }

    // Bubble up the nested scopes and all nested uses from above.
    for (auto &[scope, g] : nested.nestedScopes)
      nestedScopes.try_emplace(scope, std::move(g));
    for (ParamDeclRefAttr use : nested.usesFromAbove) {
      auto it = decls.find(use.getName());
      assert(it != decls.end() && "nested use has no declaration?");
      if (!scope->isAncestor(it->second.scope))
        usesFromAbove.insert(use);
    }
    nested.nestedScopes.clear();
    nestedScopes.try_emplace(nestedScope, std::move(nested));
  }

  // The parameter uses that a region parameter declaration depend on are
  // computed after the walk, since the walk is performed pre-order. Now that
  // we have the uses in the nested scopes, compute their dependent parameters.
  for (auto &[decl, regions] : regionValues) {
    ParamDefinition &def = defs.find(decl.getName())->second;
    for (Region *region : regions) {
      auto it = nestedScopes.find(region);
      assert(it != nestedScopes.end() && "didn't visit nested body?");
      llvm::append_range(def.uses, it->second.usesFromAbove);
    }
  }

  // Check that there is a definite partial ordering between parameters and emit
  // errors for any encountered cycles. Compute the new order.
  SmallVector<StringAttr> paramSolveOrder;
  for (auto sccIt = llvm::scc_begin(this); !sccIt.isAtEnd(); ++sccIt) {
    if (sccIt.hasCycle()) {
      emitCycleError(*this, *sccIt);
      return failure();
    }

    assert(sccIt->size() == 1 && "non-cyclic regions should have one node");
    StringAttr param = sccIt->front().param;
    if (param && scope->isAncestor(decls.find(param)->second.scope))
      paramSolveOrder.push_back(param);
  }
  params = std::move(paramSolveOrder);

  return success(!result.wasInterrupted());
}

void ParameterUseDefGraph::calculate() {
  LogicalResult result = calculateOrVerify({}, nullptr);
  assert(succeeded(result) && "IR should be legal here!");
}

LogicalResult ParameterUseDefGraph::verify(SymbolTableCollection &symtab) {
  return calculateOrVerify(scope->getParentOp()->getParentOfType<ModuleOp>(),
                           &symtab);
}

ParameterUseDefGraph ParameterUseDefGraph::copy(const IRMapping &map) {
  // Note that we use map.lookupOrDefault here because only a subgraph might
  // have been copied, so we don't necessarily have the op/block in the
  // IRMapping.

  auto remapRegion = [&](Region *region) {
    // Look up the first remapped block in the region, and return that region.
    Block *remappedBlock = map.lookupOrDefault(&region->front());
    return remappedBlock->getParent();
  };

  ParameterUseDefGraph out(*remapRegion(scope));

  // Copy over decls and defs.
  for (auto [name, decl] : decls)
    out.decls[name] = ParamDeclaration{
        decl.type, map.lookupOrDefault(decl.declOp), remapRegion(decl.scope)};
  for (auto [name, def] : defs)
    out.defs[name] = ParamDefinition{def.value, def.index,
                                     map.lookupOrDefault(def.defOp), def.uses};

  // Copy over param ops.
  for (auto paramOp : paramOps)
    out.paramOps.push_back(map.lookupOrDefault(paramOp));

  // These are trivial to copy over.
  out.params = params;
  out.usesFromAbove = usesFromAbove;

  // Copy over the op uses.
  for (auto [op, useVector] : opUses)
    out.opUses[map.lookupOrDefault(op)] = useVector;

  // Copy the remapped nested decls.
  for (auto nestedDecl : nestedDecls)
    out.nestedDecls.push_back(remapRegion(nestedDecl));

  // And finally, for each nested scope, we'll have to do the same thing.
  for (auto &[decl, graph] : nestedScopes)
    out.nestedScopes.try_emplace(remapRegion(decl), graph.copy(map));

  return out;
}
