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
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/ScopedHashTable.h"
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

  /// The first time we encounter a SymbolConstantAttr, check to see if the
  /// declaration it refers to agrees with the value and parameter
  /// specification.
  virtual void verifySymbolConstantAttr(SymbolConstantAttr symbolConstant) {}

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
  if (auto symbolConstant = dyn_cast<SymbolConstantAttr>(attr))
    verifySymbolConstantAttr(symbolConstant);

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

  /// The first time we encounter a SymbolConstantAttr, check to see if the
  /// declaration it refers to agrees with the value and parameter
  /// specification.
  void verifySymbolConstantAttr(SymbolConstantAttr symbolConstant) override;

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

void VerifyingParameterCollector::verifySymbolConstantAttr(
    SymbolConstantAttr symbolConstant) {
  // We only check this during the op verification phase.
  if (!symtab)
    return;

  // Build the signature of the referenced symbol.
  SymbolRefAttr symbol = symbolConstant.getSymbol();
  SmallVector<Operation *> symbolOps;
  if (failed(symtab->lookupSymbolIn(module, symbol, symbolOps))) {
    hadError = true;
    emitError(op->getLoc())
        << symbol << " does not reference a KGEN declaration";
    return;
  }

  // The leaf symbol must refer to a function.
  auto func = dyn_cast<FuncInterface>(symbolOps.back());
  if (!func) {
    hadError = true;
    emitError(op->getLoc()) << symbol << " does not reference a KGEN function";
    return;
  }
  // Everything else must be a declaration.
  for (Operation *op : llvm::drop_end(symbolOps)) {
    if (!isa<DeclInterface>(op)) {
      emitError(op->getLoc())
          << "symbol @" << cast<mlir::SymbolOpInterface>(op).getName()
          << " does not reference a KGEN declaration";
      return;
    }
  }

  SmallVector<ParamDeclAttr> inputParams;
  auto startIt = std::prev(symbolOps.end());
  while (startIt != symbolOps.begin() &&
         !isa<FuncInterface>(*std::prev(startIt)))
    --startIt;
  for (Operation *op : llvm::make_range(startIt, symbolOps.end()))
    llvm::append_range(inputParams,
                       cast<DeclInterface>(op).getInputParamDecls());

  auto declSignature = SignatureType::get(
      ParamDeclArrayAttr::get(func.getContext(), inputParams),
      ParamDeclArrayAttr::get(func.getContext(), func.getResultParams()),
      func.getSignature().getValues(), func.getMetadata());

  // If this SymbolConstant binds the parameters for the symbol, then remap its
  // signature to include the substitutions.
  if (!symbolConstant.getParamValues().empty()) {
    auto result = declSignature.getSpecializedSignature(
        symbolConstant.getParamValues(), [&]() {
          hadError = true;
          return emitError(op->getLoc());
        });
    if (!result)
      return;

    // The signature we just got back has all the parameters we just substituted
    // in as part of the signature and handles the unbound case correctly.
    declSignature = result;
  }

  auto symbolSignature = symbolConstant.getType();

  // Parameter types match exactly.  We could support higher order rebinding
  // if there is a need.
  SmallString<32> paramName("@");
  paramName.append(symbol.getLeafReference());
  if (failed(verifyDeclSignaturesMatch("symbol use", symbolSignature,
                                       op->getLoc(), paramName.c_str(),
                                       declSignature, func->getLoc())))
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
  if (!g->scope->getParentOp()->isAncestor(g->decls[param].declOp))
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

/// Collect parameter uses from the operation. If there are any uses or
/// otherwise unresolved parameter operators, indicate that the operation is
/// parametric.
static void collectUses(ParameterUseDefGraph &g, VerifyingParameterCollector &c,
                        Operation *op, bool isDecl) {
  // Track whether parameter uses or expressions were found.
  bool hasConstExpr = false;
  SmallVector<ParamDeclRefAttr> uses;

  // Scan the operation's operand, result, and block argument types, location,
  // and attributes.
  auto scanTypes = [&](TypeRange types) {
    for (Type type : types)
      c.collectUsesFromType(type, uses, hasConstExpr);
  };
  scanTypes(op->getOperandTypes());
  scanTypes(op->getResultTypes());
  for (Region &region : op->getRegions())
    for (Block &block : region)
      scanTypes(block.getArgumentTypes());
  // FIXME: This doesnt at the moment, because locations may contain parameters
  // that violate the use-def graph.
  c.collectUsesFromAttr(op->getAttrDictionary(), uses, hasConstExpr);

  // If the operation is parametric, add it to the list.
  if (hasConstExpr || !uses.empty()) {
    if (!isDecl)
      g.paramOps.push_back(op);
    g.opUses[op] = std::move(uses);
  } else if (isa<KGENCallOpInterface>(op) && !isDecl) {
    // Call operations are implicitly parametric.
    g.paramOps.push_back(op);
  }
}

static LogicalResult recordDecl(ParameterUseDefGraph &g, ParamDeclAttr decl,
                                Operation *op, Region &scope) {
  ParamDeclaration &paramDecl = g.decls[decl.getName()];

  // If this parameter has already been declared in an operation in the same
  // scope, we have an error.
  if (paramDecl.declOp && scope.getParentOp()->isAncestor(paramDecl.declOp)) {
    return (op->emitError("redeclaration of parameter ") << decl.getName())
               .attachNote(paramDecl.declOp->getLoc())
           << "previous declaration here";
  }

  // Record the new declaration.
  paramDecl.declOp = op;
  paramDecl.type = decl.getType();
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

/// Visit the operation to collect its uses and record any parameter
/// declarations. This function ascertains the nature of how parameters are
/// defined based on the type of the operation.
static LogicalResult visit(ParameterUseDefGraph &g,
                           VerifyingParameterCollector &c, Region &scope,
                           Operation *op) {
  bool isDecl = false;
  auto recordDeclWrapper = [&](ParamDeclAttr decl) -> LogicalResult {
    isDecl = true;
    return recordDecl(g, decl, op, scope);
  };

  auto recordDefWrapper = [&](ParamDeclAttr decl) -> ParamDefinition & {
    return recordDef(g, decl, op);
  };

  // Check for an operation that may declare parameters.
  c.op = op;
  if (auto declare = dyn_cast<ParamDeclareOp>(op)) {
    // A `kgen.param.declare` declares a parameter and defines it with a
    // parameter expression.
    if (failed(recordDeclWrapper(declare.getParamDecl())))
      return failure();
    ParamDefinition &def = recordDefWrapper(declare.getParamDecl());
    def.value = declare.getValue();
    // The definition depends on uses in the value.
    bool unused;
    c.collectUsesFromAttr(def.value, def.uses, unused);

  } else if (auto region = dyn_cast<ParamDeclareRegionOp>(op)) {
    // A `kgen.param.declare.region` declares a parameter, but the definition
    // is a region, which the elaborator will process into a symbol constant.
    if (failed(recordDeclWrapper(region.getParamDecl())))
      return failure();
    recordDefWrapper(region.getParamDecl());

  } else if (auto search = dyn_cast<ParamSearchOp>(op)) {
    // A `kgen.param.search` declares a parameter that can have one of many
    // possible values.
    if (failed(recordDeclWrapper(search.getParamDecl())))
      return failure();
    ParamDefinition &def = recordDefWrapper(search.getParamDecl());
    def.value = search.getValuesAttr();
    // The definition depends on all possible values.
    bool unused;
    c.collectUsesFromAttr(def.value, def.uses, unused);

  } else if (auto call = dyn_cast<KGENCallOpInterface>(op);
             call && !isa<GeneratorInterfaceOp>(call)) {
    // A `kgen.call` or other call operation declares parameters that bind to
    // the result parameters of the callee. All definitions depend on uses in
    // the callee expression.
    SmallVector<ParamDeclRefAttr> uses;
    bool unused;
    c.collectUsesFromAttr(call.getCallee(), uses, unused);
    for (auto [index, decl] : llvm::enumerate(call.getParamDecls())) {
      if (failed(recordDeclWrapper(decl)))
        return failure();
      ParamDefinition &def = recordDefWrapper(decl);
      def.index = index;
      def.uses = uses;
    }

  } else if (auto decl = dyn_cast<DeclInterface>(op)) {
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

  } else if (auto returnOp = dyn_cast<ReturnOp>(op)) {
    // A return operation defines the result parameters of the enclosing
    // function.
    auto func = cast<FuncInterface>(returnOp->getParentOp());
    assert(&func->getRegion(0) == &scope && "unknown return operation");
    for (auto [index, decl] : llvm::enumerate(func.getResultParams())) {
      assert(g.decls.find(decl.getName()) != g.decls.end());
      ParamDefinition &def = recordDefWrapper(decl);
      def.value = returnOp.getParameters()[index];
      def.index = index;
      // The return parameter depends on its value.
      bool unused;
      c.collectUsesFromAttr(def.value, def.uses, unused);
    }
  } else if (auto yieldOp = dyn_cast<ParamYieldOp>(op)) {
    // A yield operation defines the result parameters of the enclosing
    // param.if.
    auto ifOp = cast<ParamIfOp>(yieldOp->getParentOp());
    for (auto [index, decl] : llvm::enumerate(ifOp.getParamDecls())) {
      assert(g.decls.find(decl.getName()) != g.decls.end());
      ParamDefinition &def = recordDefWrapper(decl);
      def.value = yieldOp.getParameters()[index];
      def.index = index;
      // The return parameter depends on its value.
      bool unused;
      c.collectUsesFromAttr(def.value, def.uses, unused);
    }
  } else if (auto decls = op->getAttrOfType<ParamDeclArrayAttr>("paramDecls")) {
    // If the operation otherwise has opaque parameter declarations, include
    // them here.
    for (ParamDeclAttr decl : decls)
      if (failed(recordDeclWrapper(decl)))
        return failure();
  }

  // Collect parameter uses from the operation, if any.
  collectUses(g, c, op, isDecl);

  return success();
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

/// `param.if` ops should record itself as the decl and the def for its result
/// parameters. This will ensure the elaborator processes the `param.if` before
/// anything inside it.
static LogicalResult processParamIfOp(ParamIfOp ifOp,
                                      ParameterUseDefGraph &graph,
                                      VerifyingParameterCollector &c,
                                      Region *scope) {
  // `param.if` ops are inherently parametric. If we don't have any param decls
  // and we collected no parameter refs from the condition, mark it as
  // parametric anyway.
  SmallVector<ParamDeclRefAttr> condRefs;
  collectParameterReferences(ifOp.getCond(), condRefs);
  if (ifOp.getParamDecls().empty() && condRefs.empty())
    graph.paramOps.push_back(ifOp);

  // Record all the defs/decls.
  for (auto [idx, decl] : llvm::enumerate(ifOp.getParamDecls())) {
    if (failed(recordDecl(graph, decl, ifOp, *scope)))
      return failure();

    // And record the new definition. Its defining op is the if op.
    ParamDefinition &paramDef = recordDef(graph, decl, ifOp);
    paramDef.index = idx;
  }

  return success();
}

LogicalResult
ParameterUseDefGraph::calculateOrVerify(ModuleOp module,
                                        SymbolTableCollection *symtab) {
  // Defer the processing of the use-def node for region declarations until
  // after nested scopes have been analyzed.
  SmallVector<StringAttr> regionParams;
  SmallVector<std::pair<ParamDeclArrayAttr, Region *>> ifRegions;
  // The parameter collector to use.
  VerifyingParameterCollector c(module, symtab);

  auto processOp = [&](Operation *op) -> WalkResult {
    // Walk over nested scopes. Defer processing of nested scopes until
    // after this scope has been analyzed.
    if (auto decl = dyn_cast<DeclInterface>(op);
        decl && scope->getParentOp() != decl) {
      // Process the param.if op's result parameters in this scope.
      if (auto ifOp = dyn_cast<ParamIfOp>(op)) {
        if (failed(processParamIfOp(ifOp, *this, c, scope)))
          return WalkResult::interrupt();

        // Add then/else regions to the upper scope, we'll handle uses of the
        // condition attr later.
        ifRegions.emplace_back(ifOp.getParamDeclsAttr(), &ifOp.getThen());
        ifRegions.emplace_back(ifOp.getParamDeclsAttr(), &ifOp.getElse());
      }
      // Then, push the regions into the nested decls.
      for (Region &r : decl->getRegions())
        if (&r != scope)
          nestedDecls.push_back(&r);
      return WalkResult::skip();
    }

    if (auto region = dyn_cast<ParamDeclareRegionOp>(op))
      regionParams.push_back(region.getParamDecl().getName());

    // Visit the operation inside this scope.
    if (failed(visit(*this, c, *scope, op)))
      return WalkResult::interrupt();
    return WalkResult::advance();
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
      if (!scope->getParentOp()->isAncestor(it->second.declOp))
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
      if (!scope->getParentOp()->isAncestor(it->second.declOp))
        usesFromAbove.insert(use);
    }
    nested.nestedScopes.clear();
    nestedScopes.try_emplace(nestedScope, std::move(nested));
  }

  // The parameter uses that a region parameter declaration depend on are
  // computed after the walk, since the walk is performed pre-order. Now that
  // we have the uses in the nested scopes, compute their dependent parameters.
  for (StringAttr regionParam : regionParams) {
    ParamDefinition &def = defs[regionParam];
    auto region = cast<ParamDeclareRegionOp>(def.defOp);
    auto it = nestedScopes.find(&region.getBody().front().front().getRegion(0));
    assert(it != nestedScopes.end() && "didn't visit nested body?");
    def.uses = llvm::to_vector(it->second.usesFromAbove);
  }

  // Do the same as the region parameter decl for the if op. Results are 'used
  // by' every use from above.
  for (auto [paramDecls, region] : ifRegions) {
    for (ParamDeclAttr decl : paramDecls) {
      // Get the definition for the ref.
      ParamDefinition &def = defs[decl.getName()];
      // Add the nested uses from above to the uses of the condition's def.
      auto it = nestedScopes.find(region);
      assert(it != nestedScopes.end() && "didn't visit nested if/else?");
      def.uses.append(it->second.usesFromAbove.begin(),
                      it->second.usesFromAbove.end());
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
    if (param &&
        scope->getParentOp()->isAncestor(decls.find(param)->second.declOp))
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
    out.decls[name] =
        ParamDeclaration{decl.type, map.lookupOrDefault(decl.declOp)};
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
