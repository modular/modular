//===- KernelElaborator.cpp - Core kernel elaborator algorithm ------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains logic to lower a file full of kernel into concrete
// implementations of the kernels.
//
//===----------------------------------------------------------------------===//

#include "Internals.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "Support/DType.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"

using namespace M;
using namespace KGEN;

/// We expect all parameter expressions to simplify down to concrete constants,
/// we don't want anything left as a ParamOperatorAttr or ParamDeclRefAttr.
static bool isSimpleConstant(Attribute attr) {
  return attr.isa<FloatAttr, IntegerAttr, DTypeConstantAttr>();
}

//===----------------------------------------------------------------------===//
// Elaborator class definition
//===----------------------------------------------------------------------===//

namespace {
class Elaborator {
public:
  Elaborator(ModuleOp primary, ModuleOp library)
      : primaryModule(primary), libraryModule(library), symbolTable(primary) {}

  /// Scan the primary and library module to collect all the interfaces,
  /// verifying that any common interfaces are the same.
  ParseResult collectInterfaces();

  /// Return the operation that defines the specified symbol.
  Operation *lookupCallee(StringAttr symbolName) const {
    return symbolTable.lookup(symbolName);
  }

  /// Return all instantiations of the specified declaration (a kernel,
  /// generator, or interface) with teh specified input parameter values.
  /// `insertionPoint` is always a point in the primary module where a new
  /// kernel should be placed if necessary.
  const SmallVector<KernelOp> &
  getAllInstantiations(Operation *decl, ArrayRef<Attribute> inputParamValues,
                       Operation *insertionPoint);

  /// Insert a variant of an existing kernel into the primary file.
  void insertKernelVariant(KernelOp existing, KernelOp newKernel);

private:
  /// Specialize a kernel body, generating one variant or each viable
  /// instantiation of that body.  Kernels do not have parameters, but they can
  /// invoke interfaces etc which can cause them to produce multiple variants.
  SmallVector<KernelOp> specializeKernel(KernelOp kernel);

  /// Specialize a kernel generator with the specified input parameters and
  /// return the generated kernel.  `insertionPoint` is always a point in the
  /// primary module where a new kernel should be placed if necessary.
  SmallVector<KernelOp>
  specializeGenerator(GeneratorOp generator,
                      ArrayRef<Attribute> inputParamValues,
                      Operation *insertionPoint);

  /// Specialize a kernel interface with the specified input parameters and
  /// return the generated kernel.  `insertionPoint` is always a point in the
  /// primary module where a new kernel should be placed if necessary.
  SmallVector<KernelOp>
  specializeInterface(GeneratorInterfaceOp itf,
                      ArrayRef<Attribute> inputParamValues,
                      Operation *insertionPoint);

  ArrayRef<GeneratorOp> getGeneratorsImplementing(GeneratorInterfaceOp itf) {
    auto it = interfaceImpls.find(itf.getNameAttr());
    return it == interfaceImpls.end() ? ArrayRef<GeneratorOp>() : it->second;
  }

private:
  /// These are the two modules we start with.  The primary module is mutated by
  /// our algorithm, the library module is immutable.
  ModuleOp primaryModule, libraryModule;

  /// This symbol table allows efficient lookups in the primary module.
  SymbolTable symbolTable;

  /// This collects all of the generator implementations of generator
  /// interfaces, across both the primary module and the library.
  DenseMap<StringAttr, SmallVector<GeneratorOp, 4>> interfaceImpls;

  /// This is a cache of already-instantiated declarations.  The key is the
  /// kernel/generator/interface and input parameters, the result are
  /// all-possible kernels that could be generated from this.
  DenseMap<std::pair<Operation *, ArrayAttr>, SmallVector<KernelOp>>
      generatedKernels;
};
} // end anonymous namespace

/// Insert a variant of an existing kernel into the primary file.
void Elaborator::insertKernelVariant(KernelOp existing, KernelOp newKernel) {
  auto insertPt = Block::iterator(existing.getOperation());
  symbolTable.insert(newKernel, /*insertionPoint*/ ++insertPt);
}

//===----------------------------------------------------------------------===//
// collectInterfaces and cleanup helpers
//===----------------------------------------------------------------------===//

/// Scan the primary and library module to collect all the interfaces,
/// verifying that any common interfaces are the same.
ParseResult Elaborator::collectInterfaces() {
  // Collect all the generator interfaces in the library module, which will
  // allow cross checking them below.
  DenseMap<StringAttr, GeneratorInterfaceOp> libraryInterfaces;
  for (auto itf : libraryModule.getOps<GeneratorInterfaceOp>())
    libraryInterfaces[itf.getNameAttr()] = itf;

  // Collect all the kernel generators that implement a given interface,
  // starting with the library.  These will already have been type checked
  // within the library.
  for (auto generator : libraryModule.getOps<GeneratorOp>()) {
    if (auto interface = generator.getImplementsAttr())
      interfaceImpls[interface.getAttr()].push_back(generator);
  }

  // Collect the kernel generators from the primary module.  Start by checking
  // that any generator implementations that exist in both modules match in
  // signature exactly.
  for (auto itf : primaryModule.getOps<GeneratorInterfaceOp>()) {
    auto it = libraryInterfaces.find(itf.getNameAttr());
    if (it == libraryInterfaces.end())
      continue;
    if (failed(verifyDeclMatchesInterface("interface", itf, "library interface",
                                          it->second)))
      return failure();
  }

  // If they all match up, collect the generator implementations from the
  // primary module.
  for (auto generator : primaryModule.getOps<GeneratorOp>())
    if (auto interface = generator.getImplementsAttr())
      interfaceImpls[interface.getAttr()].push_back(generator);

  return success();
}

//===----------------------------------------------------------------------===//
// Elaborator Algorithm for one Kernel
//===----------------------------------------------------------------------===//

namespace {
/// This class keeps a set of defined parameter values and is used to evaluate
/// and simplify operations based on those values.
class ParameterRewriter {
public:
  ParameterRewriter(Elaborator &Elaborator, KernelOp kernel,
                    SmallVector<Operation *> opsToRewrite)
      : Elaborator(Elaborator), kernel(kernel),
        opsToRewrite(std::move(opsToRewrite)) {}

  /// Create a clone of this rewriter, but refer with a clone of the kernel.
  /// This uses operationMap to remap our state onto the newly created kernel.
  ParameterRewriter(const ParameterRewriter &existing,
                    DenseMap<Operation *, Operation *> &operationMap);

  /// Process all the `opsToRewrite`, simplifying this kernel.  If new variants
  /// of this kernel are necessary, they are added to rewriterWorklist.
  LogicalResult rewriteOps(SmallVector<ParameterRewriter, 2> &rewriterWorklist);

  /// Return the kernel we're generating into.
  KernelOp getKernel() const { return kernel; }

private:
  /// Set a value for the specified parameter declaration to the specified
  /// simplified value.
  void setParameterValue(ParamDeclAttr decl, Attribute value) {
    assert(!paramValues.count(decl.getName()) && "parameter already declared!");
    assert(isSimpleConstant(value) && "expression isn't simplified");
    paramValues[decl.getName()] = value;
  }

  /// Given a generic parameter expression, simplify it by folding the
  /// expression according to known parameter values.  This returns an error if
  /// the expression cannot be folded for one reason or another.
  ErrorOr<Attribute> simplifyParameterExpr(Attribute expr);

private:
  void processParamBindOp(ParamBindOp op);
  void processParamValueOp(ParamValueOp op);
  void processCallOp(CallOp call,
                     SmallVector<ParameterRewriter, 2> &rewriterWorklist);
  void completeCallOpProcessing(CallOp call, KernelOp newCallee);
  void spawnNewKernelClone(CallOp call, KernelOp callee,
                           SmallVector<ParameterRewriter, 2> &rewriterWorklist);
  void processGenericOp(Operation *op);

  /// Get the specified attribute with any nested parameter expressions
  /// rewritten.
  Attribute getReboundAttribute(Attribute attr, Location loc);

  /// Get the specified type with any nested parameter expressions rewritten.
  Type getReboundType(Type type, Location loc);

  // This is maintains global information about the file we're generating into.
  Elaborator &Elaborator;

  /// This is the kernel we're working on.
  KernelOp kernel;

  /// These are the operations we still need to visit to complete our rewrite.
  SmallVector<Operation *> opsToRewrite;

  /// These are the bound parameter values, captured in simplified form.
  DenseMap<StringAttr, Attribute> paramValues;

  /// This caches attributes and Types with parameter references rebound, and
  /// remembers complex attributes that don't have parameter subexprs (noted as
  /// being rebound to themselves).
  DenseMap<Attribute, Attribute> rewrittenAttrs;
  DenseMap<Type, Type> rewrittenTypes;
};
} // end anonymous namespace

/// Create a clone of this rewriter, but refer with a clone of the kernel.
/// This uses operationMap to remap our state onto the newly created kernel.
ParameterRewriter::ParameterRewriter(
    const ParameterRewriter &existing,
    DenseMap<Operation *, Operation *> &operationMap)
    : Elaborator(existing.Elaborator), paramValues(existing.paramValues),
      rewrittenAttrs(existing.rewrittenAttrs),
      rewrittenTypes(existing.rewrittenTypes) {
  // Remap the kernel itself.
  kernel = cast<KernelOp>(operationMap[existing.kernel]);
  assert(kernel && "didn't remap kernel correctly");

  // Remap the operation worklist.
  opsToRewrite.reserve(existing.opsToRewrite.size());
  for (Operation *op : existing.opsToRewrite) {
    opsToRewrite.push_back(operationMap[op]);
    assert(opsToRewrite.back() && "didn't clone operation correctly?");
  }
}

/// Work the `opsToRewrite` worklist.
LogicalResult ParameterRewriter::rewriteOps(
    SmallVector<ParameterRewriter, 2> &rewriterWorklist) {
  /// We use a worklist for this so cloned versions of ParameterRewriter can
  /// be created and known where to pick up from.
  while (!opsToRewrite.empty()) {
    Operation *op = opsToRewrite.pop_back_val();

    /// Process an operation that needs to be rewritten/lowered based on the
    /// context of the parameter values we know are defined.
    if (auto bind = dyn_cast<ParamBindOp>(op))
      processParamBindOp(bind);
    else if (auto value = dyn_cast<ParamValueOp>(op))
      processParamValueOp(value);
    else if (auto call = dyn_cast<CallOp>(op))
      processCallOp(call, rewriterWorklist);
    else if (isa<KernelOp>(op))
      /*kernels can define parameters, nothing need be done with them*/;
    else
      processGenericOp(op);
  }

  // Check that the thing we just built is correct!
  return verify(kernel);
}

static std::string getString(Attribute attr) {
  std::string str;
  llvm::raw_string_ostream(str) << attr;
  return str;
}

/// Given a generic parameter expression, simplify it by folding the
/// expression according to known parameter values.  This returns an error if
/// the expression cannot be folded for one reason or another.
ErrorOr<Attribute> ParameterRewriter::simplifyParameterExpr(Attribute expr) {
  // Simple constants don't need simplification.
  if (isSimpleConstant(expr))
    return expr;

  // We can directly substitute declaration references given our known table of
  // bindings.
  if (auto declRef = expr.dyn_cast<ParamDeclRefAttr>()) {
    auto value = paramValues[declRef.getName()];
    assert(value && "Verifier should check that all parameters are defined");
    return value;
  }

  // Simplify operators by recursively simplifying their operands, then
  // refolding the expression.
  if (auto oper = expr.dyn_cast<ParamOperatorAttr>()) {
    SmallVector<Attribute> simplifiedOperands;
    for (auto value : oper.getOperands()) {
      auto simplified = simplifyParameterExpr(value);
      if (simplified.isError())
        return simplified;
      simplifiedOperands.push_back(simplified.takeValue());
    }

    // FIXME: 'index' folding should require target information to simplify
    // things like div.
    auto result = ParamOperatorAttr::get(oper.getOpcode(), simplifiedOperands);
    if (!isSimpleConstant(result))
      return Error("could not simplify operator " + getString(expr));
    return result;
  }

  // Otherwise, we don't know how to simplify this attribute, it's an error.
  return Error("unknown expression to fold: " + getString(expr));
}

void ParameterRewriter::processParamBindOp(ParamBindOp op) {
  // Simplify the input expression.
  auto errorOrValue = simplifyParameterExpr(op.getValue());
  if (errorOrValue.isError()) {
    op->emitError(errorOrValue.getError());
    return;
  }

  // Bind it to the parameter declaration it is setting.
  setParameterValue(op.getParamDecl(), errorOrValue.takeValue());

  // The param.bind operation serves no other purpose, so we can remove it.
  op->erase();
}

void ParameterRewriter::processParamValueOp(ParamValueOp op) {
  // ParamValueOp projects a parameter expression into an SSA value.  We can
  // eventually lower this into lower level operators in the target set, but
  // for now we just simplify their operand.
  auto errorOrValue = simplifyParameterExpr(op.getValue());
  if (errorOrValue.isError()) {
    op->emitError(errorOrValue.getError());
    return;
  }

  op.setValueAttr(errorOrValue.takeValue());
}

void ParameterRewriter::processCallOp(
    CallOp call, SmallVector<ParameterRewriter, 2> &rewriterWorklist) {
  // Evaluating any input parameters.
  SmallVector<Attribute> boundInputParams;
  for (auto param : call.getParamValues()) {
    auto value = simplifyParameterExpr(param.cast<ParamBindAttr>().getValue());
    if (value.isError()) {
      call->emitError(value.getError());
      return;
    }
    boundInputParams.push_back(value.takeValue());
  }

  // Instantiate the callee into one or more KernelOp's, depending on what the
  // callee is.
  auto callee = Elaborator.lookupCallee(call.getCalleeAttr().getAttr());
  SmallVector<KernelOp> newCallees =
      Elaborator.getAllInstantiations(callee, boundInputParams, kernel);

  // If kernel instantiation failed for some reason, bail out.  The error will
  // already be reported.
  if (newCallees.empty())
    return;

  // If we found more than one callee to produce then we need to spawn multiple
  // versions of the kernel we are currently constructing, each which get a
  // different callee.
  for (KernelOp callee : llvm::drop_begin(newCallees))
    spawnNewKernelClone(call, callee, rewriterWorklist);

  // Finally, we can handle the first one as our continued progress here.
  completeCallOpProcessing(call, newCallees[0]);
}

void ParameterRewriter::completeCallOpProcessing(CallOp call,
                                                 KernelOp newCallee) {
  // If we resolved the call to a new thing, build a new call to replace the old
  // one.
  OpBuilder b(call);
  auto newCall = b.create<CallOp>(
      call.getLoc(), call.getResultTypes(), newCallee.getNameAttr(),
      /*input params*/ ArrayRef<Attribute>(),
      /*output params*/ call.getParamDecls().getValue(), call.getOperands());

  // The SSA results of the old call go directly to the new call and remove it.
  call->getResults().replaceAllUsesWith(newCall);
  call->erase();

  // Bind the result parameters to the output parameter decls.
  for (auto [decl, bindValue] : llvm::zip(
           newCall.getParamDecls(), newCallee.getReturnOp().getParameters()))
    setParameterValue(decl.cast<ParamDeclAttr>(),
                      bindValue.cast<ParamBindAttr>().getValue());
}

/// Sometimes when we expand a call, we find that there are multiple viable
/// callees that we can generate.  We handle this by spawning new parameter
/// rewriters with state copied from the current one, but which resolve the call
/// to different callees.  This spawns a new rewriter with the specified call
/// resolving to the specified callee.
void ParameterRewriter::spawnNewKernelClone(
    CallOp call, KernelOp callee,
    SmallVector<ParameterRewriter, 2> &rewriterWorklist) {

  // Start by cloning the current WIP kernel to a new copy of it.
  BlockAndValueMapping blocksAndValues;
  DenseMap<Operation *, Operation *> operationMap;
  auto newKernel =
      cast<KernelOp>(cloneOperation(kernel, blocksAndValues, operationMap));

  // Insert the kernel into the output file and auto-unique the symbol.
  Elaborator.insertKernelVariant(kernel, newKernel);

  // Generate the new rewriter which will process this.
  auto &newRewriter = rewriterWorklist.emplace_back(*this, operationMap);

  // Change the future of this kernel by resolving the call in the new kernel to
  // the specifed callee.
  auto newCall = cast<CallOp>(operationMap[call]);
  newRewriter.completeCallOpProcessing(newCall, callee);
}

/// Get the specified attribute with any nested parameter expressions
/// rewritten.
Attribute ParameterRewriter::getReboundAttribute(Attribute attr, Location loc) {
  // These are common leaf attributes that we know are never parameterized.
  if (attr.isa<IntegerAttr, FloatAttr, StringAttr, SymbolRefAttr,
               DTypeConstantAttr>())
    return attr;

  // If we've already processed this attribute, just reuse the memoized result.
  auto iter = rewrittenAttrs.find(attr);
  if (iter != rewrittenAttrs.end())
    return iter->second;

  // TODO(jeff): MLIR attribute should not carry types!
  if (getReboundType(attr.getType(), loc) != attr.getType()) {
    emitError(loc, "unsupported parameterized type in attribute ") << attr;
    return rewrittenAttrs[attr] = attr;
  }

  // If this is a foldable parameter expression, do it.
  Attribute result = attr;
  if (attr.isa<ParamDeclRefAttr, ParamOperatorAttr>()) {
    auto newVal = simplifyParameterExpr(attr);
    if (!newVal.isError())
      result = newVal.takeValue();

  } else if (auto itf = attr.dyn_cast<mlir::SubElementAttrInterface>()) {
    SmallVector<std::pair<size_t, Attribute>> newAttrs;
    bool changedType = false;
    size_t attrNo = 0;
    itf.walkImmediateSubElements(
        [&](Attribute attr) {
          auto newAttr = getReboundAttribute(attr, loc);
          if (newAttr != attr)
            newAttrs.push_back(std::make_pair(attrNo, newAttr));
          ++attrNo;
        },
        [&](Type type) { changedType = type != getReboundType(type, loc); });
    if (changedType) {
      // TODO: Improve SubElementTypeInterface:
      // https://github.com/llvm/llvm-project/issues/56355
      emitError(loc, "don't know how to rebind parameterized subtypes in ")
          << attr;
    } else if (!newAttrs.empty()) {
      result = itf.replaceImmediateSubAttribute(newAttrs);
    }
  } else {
    emitError(loc, "unknown attribute in parameterized operation ") << attr;
  }

  return rewrittenAttrs[attr] = result;
}

/// Get the specified type with any nested parameter expressions rewritten.
Type ParameterRewriter::getReboundType(Type type, Location loc) {
  // These are known leaf types that don't participate with
  // SubElementTypeInterface and have no attributes or types within them.
  if (type.isa<IntegerType, FloatType, NoneType, IndexType, DTypeType>())
    return type;

  // If we've already processed this type, just reuse the memoized result.
  auto iter = rewrittenTypes.find(type);
  if (iter != rewrittenTypes.end())
    return iter->second;

  Type result = type;
  if (auto itf = type.dyn_cast<mlir::SubElementTypeInterface>()) {
    SmallVector<std::pair<size_t, Attribute>> newAttrs;
    bool changedType = false;
    size_t attrNo = 0;
    itf.walkImmediateSubElements(
        [&](Attribute attr) {
          auto newAttr = getReboundAttribute(attr, loc);
          if (newAttr != attr)
            newAttrs.push_back(std::make_pair(attrNo, newAttr));
          ++attrNo;
        },
        [&](Type type) { changedType = type != getReboundType(type, loc); });
    if (changedType) {
      // TODO: Improve SubElementTypeInterface:
      // https://github.com/llvm/llvm-project/issues/56355
      emitError(loc, "don't know how to rebind parameterized subtypes in ")
          << type;
    } else if (!newAttrs.empty()) {
      result = itf.replaceImmediateSubAttribute(newAttrs);
    }
  } else {
    emitError(loc, "unknown type in parameterized operation ") << type;
  }

  return rewrittenTypes[type] = result;
}

/// Unknown operations are allowed to use types and attributes with parameter
/// references.  Substitute in concrete values for their references.
void ParameterRewriter::processGenericOp(Operation *op) {
  // We can rewrite generic references and /uses/ of parameters, but we don't
  // don't know how to calculate the new value for a defined parameter.  If
  // there is a reason to allow open extension of operations that define
  // parameters, we could genericize this into a op interface.
  if (!getParamDecls(op).empty()) {
    op->emitError("unknown parameter-defining operator in GenerateKernels");
    return;
  }

  // Scan all the attributes and types to look for uses of parameters.  We let
  // the walker scan the region hierarchy.
  SmallVector<NamedAttribute> newAttrs;
  bool changedAttrs = false;
  for (const NamedAttribute &namedAttr : op->getAttrs()) {
    newAttrs.push_back(NamedAttribute(
        namedAttr.getName(),
        getReboundAttribute(namedAttr.getValue(), op->getLoc())));
    changedAttrs |= namedAttr.getValue() != newAttrs.back().getValue();
  }
  if (changedAttrs)
    op->setAttrs(newAttrs);

  // Check the types of results to find any parameters embedded in their
  // types.  We don't have to check operands because they are always checked
  // when being defined.
  for (OpResult result : op->getResults())
    result.setType(getReboundType(result.getType(), op->getLoc()));

  // Scan the region list if present.  The walker will automatically recurse
  // for us, but we have to check the block arguments.
  if (op->getNumRegions()) { // Microoptimization: getRegions() is slow.
    for (auto &region : op->getRegions())
      for (auto &block : region)
        for (Value arg : block.getArguments())
          arg.setType(getReboundType(arg.getType(), op->getLoc()));
  }
}

//===----------------------------------------------------------------------===//
// Elaborator::getAllInstantiations
//===----------------------------------------------------------------------===//

/// This returns a name to use when the specified generator is specialized
/// with the specified input parameters.
static StringAttr mangleParameterValues(GeneratorOp generator,
                                        ArrayRef<Attribute> inputParamValues) {
  Builder b(generator.getContext());
  if (inputParamValues.empty())
    return b.getStringAttr(generator.getName() + "_kernel");

  std::string result;
  llvm::raw_string_ostream os(result);
  os << generator.getName();

  auto inputParamDecls = generator.getParameterInfo().first;
  for (auto [inputDecl, value] : llvm::zip(inputParamDecls, inputParamValues)) {
    os << ',' << inputDecl.cast<ParamDeclAttr>().getName().str() << '=';

    if (auto intAttr = value.dyn_cast<IntegerAttr>()) {
      os << intAttr.getValue();
    } else if (auto floatAttr = value.dyn_cast<FloatAttr>()) {
      SmallString<32> str;
      floatAttr.getValue().toString(str);
      os << str;
    } else if (auto dtypeAttr = value.dyn_cast<DTypeConstantAttr>()) {
      os << dtypeAttr.getDType();
    } else {
      assert(!isSimpleConstant(value) && "not handling all simple constants");
      os << "??";
    }
  }
  return b.getStringAttr(result);
}

/// Specialize a kernel body, generating one variant or each viable
/// instantiation of that body.  Kernels do not have parameters, but they can
/// invoke interfaces etc which can cause them to produce multiple variants.
SmallVector<KernelOp> Elaborator::specializeKernel(KernelOp kernel) {
  /// Get a partial ordering of parameter definitions and uses that is listed
  /// "top down" in our evaluation order.
  SmallVector<Operation *> opsToRewrite;
  {
    auto paramInfo = ParameterDeclsAndUses::calculate(kernel);
    if (failed(paramInfo)) {
      kernel->emitError("verification error for kernel");
      return {};
    }
    opsToRewrite = paramInfo->getUsingAndDeclaringOps();
  }

  // We are going to use opsToRewrite as a worklist, so reverse it for efficient
  // pop_back.
  std::reverse(opsToRewrite.begin(), opsToRewrite.end());

  // Start by rewriting this kernel.
  SmallVector<ParameterRewriter, 2> rewriterWorklist;
  rewriterWorklist.emplace_back(*this, kernel, std::move(opsToRewrite));

  // Rewriting kernels may generate other kernel clones.  If so, rewrite them,
  // until we converge.
  SmallVector<KernelOp> results;
  while (!rewriterWorklist.empty()) {
    auto rewriter = rewriterWorklist.pop_back_val();
    if (succeeded(rewriter.rewriteOps(rewriterWorklist)))
      results.push_back(rewriter.getKernel());
  }
  return results;
}

/// Specialize a kernel generator with the specified input parameters and
/// return the symbol name to use for the result, along with an array of
/// ParamBindAttrs for the result attributes.  `insertionPoint` is always a
/// point in the primary module where a new kernel should be placed if
/// necessary.
SmallVector<KernelOp>
Elaborator::specializeGenerator(GeneratorOp generator,
                                ArrayRef<Attribute> inputParamValues,
                                Operation *insertionPoint) {
  // We insert specializations of the generator immediately before the generator
  // if it is defined in the primary module.  Otherwise if it is from the
  // library, it would be better to insert it before the first client that
  // needed it (make tests easier to write).
  if (generator->getParentOp() == primaryModule) {
    insertionPoint = generator;
  } else {
    assert(insertionPoint && insertionPoint->getParentOp() == primaryModule);
  }

  auto [inputParamDecls, resultParamDecls] = generator.getParameterInfo();
  assert(inputParamValues.size() == inputParamDecls.size() &&
         "incorrect # input parameter values");

  // TODO (low prio): Some day we could mangle "instantiated from here"
  // information into the location.
  OpBuilder b(insertionPoint);
  auto newKernel = b.create<KernelOp>(
      generator.getLoc(), mangleParameterValues(generator, inputParamValues),
      generator.getFunctionType(), resultParamDecls);

  // Insert the newKernel into the symbol table which will then know about it,
  // but it will also auto-rename the symbol for us in the case of conflicts.
  symbolTable.insert(newKernel);

  // Clone the body of the generator over.
  BlockAndValueMapping mapper;
  generator.getBody().cloneInto(&newKernel.getBody(), mapper);

  // Provide definitions of the input parameters in the body block as bound
  // constants.
  b.setInsertionPoint(&newKernel.getBodyBlock()->front());
  for (auto [inputDecl, inputValue] :
       llvm::zip(inputParamDecls, inputParamValues)) {
    b.create<ParamBindOp>(generator.getLoc(), inputDecl.cast<ParamDeclAttr>(),
                          inputValue);
  }

  // Now that we have a new synthesized generic kernel, run the rewriter over it
  // to specialize its body.
  return specializeKernel(newKernel);
}

/// Specialize a kernel interface with the specified input parameters and
/// return the generated kernel.  `insertionPoint` is always a point in the
/// primary module where a new kernel should be placed if necessary.
SmallVector<KernelOp>
Elaborator::specializeInterface(GeneratorInterfaceOp itf,
                                ArrayRef<Attribute> inputParamValues,
                                Operation *insertionPoint) {
  // An interface is an abstraction over multiple generators.  Invoke each of
  // them, collecting the results together into a single result.
  ArrayRef<GeneratorOp> interfaceImpls = getGeneratorsImplementing(itf);
  if (interfaceImpls.empty())
    return {};

  SmallVector<KernelOp> result;
  for (GeneratorOp gen : interfaceImpls) {
    // Make sure to go through getAllInstantiations so generators are cached.
    result.append(getAllInstantiations(gen, inputParamValues, insertionPoint));
  }
  return result;
}

/// Return all instantiations of the specified declaration (a kernel,
/// generator, or interface) with teh specified input parameter values.
/// `insertionPoint` is always a point in the primary module where a new
/// kernel should be placed if necessary.
const SmallVector<KernelOp> &
Elaborator::getAllInstantiations(Operation *decl,
                                 ArrayRef<Attribute> inputParamValues,
                                 Operation *insertionPoint) {

  // Check the cache these so multiple uses of the same kernel don't get
  // separate instantiations.
  auto inputParamKey = ArrayAttr::get(decl->getContext(), inputParamValues);
  auto cacheKey = std::make_pair(decl, inputParamKey);
  auto cacheIt = generatedKernels.find(cacheKey);
  if (cacheIt != generatedKernels.end())
    return cacheIt->second;

  SmallVector<KernelOp> newCallees;
  if (auto kernel = dyn_cast<KernelOp>(decl))
    newCallees = specializeKernel(kernel);
  else if (auto generator = dyn_cast<GeneratorOp>(decl))
    newCallees = specializeGenerator(generator, inputParamValues, kernel);
  else if (auto interface = dyn_cast<GeneratorInterfaceOp>(decl))
    newCallees = specializeInterface(interface, inputParamValues, kernel);
  else
    decl->emitError("cannot handle this yet");

  auto &result = generatedKernels[cacheKey];
  result = std::move(newCallees);
  return result;
}

//===----------------------------------------------------------------------===//
// generateKernels Driver
//===----------------------------------------------------------------------===//

/// Elaborate kernels in the specified module, incorporating implementation
/// logic from the specified library.
LogicalResult M::elaborateKernels(ModuleOp primary, ModuleOp library) {
  // We currently rely on pointer equivalence between attributes etc when
  // matching across modules, so the modules must be in the same context.  We
  // could relax this restriction in the future if there were a reason to.
  if (primary.getContext() != library.getContext())
    return primary.emitError() << "Cannot generate kernels when primary and "
                                  "library are in different MLIR contexts";
  Elaborator generator(primary, library);

  // Scan the primary and library module to collect all the interfaces,
  // verifying that any common interfaces are the same.
  if (generator.collectInterfaces())
    return failure();

  // Concretize all the kernels at the top-level.
  bool didFail = false;
  SmallVector<KernelOp, 16> kernelsToGenerate;

  // Collect all the kernels to generate in a prepass, because we will be
  // creating new kernels in the primary file that are already concretized and
  // we don't want to reprocess them.
  // FIXME: This isn't correct at all: kernels can call kernels which can lead
  // to them getting deleted.

  // TODO: When expanding a kernel we need to pass in history of prior expansion
  // bindings, which constrains/defines future expansions of the same thing, and
  // we need to return up novel bindings that are done.  For each multi-version
  // we need to track /which way/ we're resolving an ambiguity.  For something
  // like this:
  //    kernel @foo() {
  //      call @someInterface()        // has 5 implementations
  //    }
  //
  //    kernel @bar() { call @foo() }  // has 5 implementations
  //
  //    kernel @baz() {
  //      call @foo()
  //      call @foo()
  //      call @bar()
  //    }
  //
  // We should process @bar before recursing down to @foo.  We should only
  // generate 5 copies of @bar, each of which resolves the call to
  // foo->someInterface in the same direction.  We should not generate 5*5*5
  // copies of @bar that has all pairs of foo/someInterface resolutions.
  for (auto kernel : primary.getOps<KernelOp>())
    kernelsToGenerate.push_back(kernel);

  // Process each kernel.
  for (auto kernel : kernelsToGenerate) {
    SmallVector<KernelOp> results =
        generator.getAllInstantiations(kernel, {}, kernel);
    didFail |= results.empty();
  }

  // If we failed to expand any kernel, propagate that failure.
  if (didFail)
    return failure();

  // On success, we remove generators and generator interfaces from the file to
  // clean it up.
  for (Operation &op : llvm::make_early_inc_range(primary.getOps())) {
    if (isa<GeneratorOp, GeneratorInterfaceOp>(op))
      op.erase();
  }

  return success();
}
