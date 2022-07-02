//===- GenerateKernels.cpp - Kernel generator driver ----------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file contains logic to lower a file full of kernel generators into
//
//===----------------------------------------------------------------------===//

#include "Internals.h"

#include "GenericML/Support/TensorEltType.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
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
// KernelGenerator class definition
//===----------------------------------------------------------------------===//

namespace {
class KernelGenerator {
public:
  KernelGenerator(ModuleOp primary, ModuleOp library)
      : primaryModule(primary), libraryModule(library), symbolTable(primary) {}

  /// Scan the primary and library module to collect all the interfaces,
  /// verifying that any common interfaces are the same.
  ParseResult collectInterfaces();

  /// Concretize a kernel in the primary file.
  ParseResult processKernel(KernelOp kernel);

  /// Concretize all kernels in the primary file.
  ParseResult processKernels();

  /// Remove generators and generator interfaces from the file to clean it up.
  void removeGenerators();

  /// Return the operation that defines the specified symbol.
  Operation *lookupCallee(StringAttr symbolName) const {
    return symbolTable.lookup(symbolName);
  }

  /// Specialize a kernel generator with the specified input parameters and
  /// return the symbol name to use for the result.  `insertionPoint` is always
  /// a point in the primary module where a new kernel should be placed if
  /// necessary.
  std::pair<StringAttr, SmallVector<Attribute>>
  getSpecializedGenerator(GeneratorOp generator,
                          ArrayRef<Attribute> inputParamValues,
                          Operation *insertionPoint);

private:
  /// These are the two modules we start with.  The primary module is mutated by
  /// our algorithm, the library module is immutable.
  ModuleOp primaryModule, libraryModule;

  /// This symbol table allows efficient lookups in the primary module.
  SymbolTable symbolTable;

  /// This collects all of the generator implementations of generator
  /// interfaces, across both the primary module and the library.
  DenseMap<StringAttr, SmallVector<GeneratorOp, 4>> interfaceImpls;
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// collectInterfaces and cleanup helpers
//===----------------------------------------------------------------------===//

/// Scan the primary and library module to collect all the interfaces,
/// verifying that any common interfaces are the same.
ParseResult KernelGenerator::collectInterfaces() {
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

/// Remove generators and generator interfaces from the file to clean it up.
void KernelGenerator::removeGenerators() {
  for (Operation &op : llvm::make_early_inc_range(primaryModule.getOps())) {
    if (isa<GeneratorOp, GeneratorInterfaceOp>(op))
      op.erase();
  }
}

//===----------------------------------------------------------------------===//
// Core Kernel Generator Algorithm
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
      os << dtypeAttr.getTensorEltType();
    } else {
      assert(!isSimpleConstant(value) && "not handling all simple constants");
      os << "??";
    }
  }
  return b.getStringAttr(result);
}

//===----------------------------------------------------------------------===//
// Core Kernel Generator Algorithm
//===----------------------------------------------------------------------===//

namespace {
/// This class keeps a set of defined parameter values and is used to evaluate
/// and simplify operations based on those values.
class ParameterRewriter {
public:
  ParameterRewriter(KernelGenerator &kernelGenerator)
      : kernelGenerator(kernelGenerator) {}

  /// Process the body of a kernel.
  LogicalResult processKernelBody(KernelOp kernel);

  /// Process an operation that needs to be rewritten/lowered based on the
  /// context of the parameter values we know are defined.
  LogicalResult processOp(Operation *op);

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
  void processCallOp(CallOp op);

  // This is maintains global information about the file we're generating into.
  KernelGenerator &kernelGenerator;

  /// These are the bound parameter values, captured in simplified form.
  DenseMap<StringAttr, Attribute> paramValues;
};
} // end anonymous namespace

/// Process the body of a kernel.
LogicalResult ParameterRewriter::processKernelBody(KernelOp kernel) {
  /// Get a partial ordering of parameter definitions and uses that is listed
  /// "top down" in our evaluation order.
  auto paramInfo = ParameterDeclsAndUses::calculate(kernel);
  if (failed(paramInfo))
    return kernel->emitError("verification error for kernel");

  // Process each def/use in order.
  for (auto &user : paramInfo->usersAndDeclarers) {
    if (failed(processOp(user.first)))
      return failure();
  }

  return success();
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

LogicalResult ParameterRewriter::processOp(Operation *op) {
  if (auto bind = dyn_cast<ParamBindOp>(op))
    processParamBindOp(bind);
  else if (auto value = dyn_cast<ParamValueOp>(op))
    processParamValueOp(value);
  else if (auto call = dyn_cast<CallOp>(op))
    processCallOp(call);
  else if (getParamDecls(op).empty())
    return op->emitError("unknown parameter-using operator in GenerateKernels");
  else
    return op->emitError(
        "unknown parameter defining operator in GenerateKernels");

  return success();
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

void ParameterRewriter::processCallOp(CallOp call) {
  // If this is a direct call to an existing kernel (not a generator or
  // interface) then we have nothing to do.  It cannot be parameterized, and it
  // must already be in the primary module.
  auto callee = kernelGenerator.lookupCallee(call.getCalleeAttr().getAttr());
  if (isa<KernelOp>(callee))
    return;

  // Otherwise, if it is a generator or generator interface we need to
  // specialize it if it isn't already.  Start by evaluating the input
  // parameters.
  SmallVector<Attribute> boundInputParameters;
  for (auto param : call.getParamValues()) {
    auto value = simplifyParameterExpr(param.cast<ParamBindAttr>().getValue());
    if (value.isError()) {
      call->emitError(value.getError());
      return;
    }
    boundInputParameters.push_back(value.takeValue());
  }

  auto generator = dyn_cast<GeneratorOp>(callee);
  if (!generator) {
    // TODO: Handle generator interfaces.
    call->emitError("cannot handle this yet");
    return;
  }

  auto [newCallee, outputParams] = kernelGenerator.getSpecializedGenerator(
      generator, boundInputParameters, call->getParentOfType<KernelOp>());

  // If kernel generation failed for some reason, bail out.  The error will
  // already be reported.
  if (!newCallee)
    return;

  OpBuilder b(call);
  auto newCall = b.create<CallOp>(
      call.getLoc(), call.getResultTypes(), newCallee,
      /*input params*/ ArrayRef<Attribute>(),
      /*output params*/ ArrayRef<Attribute>(), call.getOperands());

  // The SSA results of the old call go directly to the new call.
  call->getResults().replaceAllUsesWith(newCall);

  // Bind the result parameters to the output parameter decls.
  for (auto [decl, value] : llvm::zip(call.getParamDecls(), outputParams))
    setParameterValue(decl.cast<ParamDeclAttr>(), value);

  // The old call is resolved and dead now.
  call->erase();
}

/// Concretize a kernel in the primary file.
ParseResult KernelGenerator::processKernel(KernelOp kernel) {
  ParameterRewriter rewriter(*this);
  return rewriter.processKernelBody(kernel);
}

ParseResult KernelGenerator::processKernels() {
  bool didFail = false;
  SmallVector<KernelOp, 16> kernelsToGenerate;

  // Collect all the kernels to generate in a prepass, because we will be
  // creating new kernels in the primary file that are already concretized and
  // we don't want to reprocess them.
  for (auto kernel : primaryModule.getOps<KernelOp>())
    kernelsToGenerate.push_back(kernel);

  // Process each kernel.
  for (auto kernel : kernelsToGenerate)
    didFail |= failed(processKernel(kernel));

  return failure(didFail);
}

//===----------------------------------------------------------------------===//
// KernelGenerator::getSpecializedGenerator
//===----------------------------------------------------------------------===//

/// Specialize a kernel generator with the specified input parameters and
/// return the symbol name to use for the result.
std::pair<StringAttr, SmallVector<Attribute>>
KernelGenerator::getSpecializedGenerator(GeneratorOp generator,
                                         ArrayRef<Attribute> inputParamValues,
                                         Operation *insertionPoint) {
  // TODO: Cache these so multiple uses of the same kernel don't get separate
  // instantiations.

  // We insert specializations of the generator immediately before the generator
  // if it is defined in the primary module.  Otherwise if it is from the
  // library, it would be better to insert it before the first client that
  // needed it (make tests easier to write).
  if (generator->getParentOp() == primaryModule) {
    insertionPoint = generator;
  } else {
    assert(insertionPoint && insertionPoint->getParentOp() == primaryModule);
  }

  // TODO (low prio): Some day we could mangle "instantiated from here"
  // information into the location.
  OpBuilder b(insertionPoint);
  auto newKernel = b.create<KernelOp>(
      generator.getLoc(), mangleParameterValues(generator, inputParamValues),
      generator.getFunctionType());

  // Insert the newKernel into the symbol table which will then know about it,
  // but it will also auto-rename the symbol for us in the case of conflicts.
  symbolTable.insert(newKernel);

  // Clone the body of the generator over.
  BlockAndValueMapping mapper;
  generator.getBody().cloneInto(&newKernel.getBody(), mapper);

  // Provide definitions of the input parameters in the body block as bound
  // constants.
  auto [inputParamDecls, resultParamDecls] = generator.getParameterInfo();
  assert(inputParamValues.size() == inputParamDecls.size() &&
         "incorrect # input parameter values");

  b.setInsertionPoint(&newKernel.getBodyBlock()->front());
  for (auto [inputDecl, inputValue] :
       llvm::zip(inputParamDecls, inputParamValues)) {
    b.create<ParamBindOp>(generator.getLoc(), inputDecl.cast<ParamDeclAttr>(),
                          inputValue);
  }

  // Now that we have a new synthesized generic kernel, run the rewriter over it
  // to specialize its body.
  ParameterRewriter rewriter(*this);
  if (failed(rewriter.processKernelBody(newKernel)))
    return {};

  // TODO: Handle output parameters.

  // Check that the thing we just built is correct!
  if (failed(verify(newKernel)))
    return {};

  return std::make_pair(newKernel.getNameAttr(), SmallVector<Attribute>());
}

//===----------------------------------------------------------------------===//
// generateKernels Driver
//===----------------------------------------------------------------------===//

/// Generate kernels in the specified module, incorporating implementation logic
/// from the specified library.
LogicalResult M::generateKernels(ModuleOp primary, ModuleOp library) {
  // We currently rely on pointer equivalence between attributes etc when
  // matching across modules, so the modules must be in the same context.  We
  // could relax this restriction in the future if there were a reason to.
  if (primary.getContext() != library.getContext())
    return primary.emitError() << "Cannot generate kernels when primary and "
                                  "library are in different MLIR contexts";
  KernelGenerator generator(primary, library);

  // Scan the primary and library module to collect all the interfaces,
  // verifying that any common interfaces are the same.
  if (generator.collectInterfaces() || generator.processKernels())
    return failure();

  generator.removeGenerators();
  return success();
}
