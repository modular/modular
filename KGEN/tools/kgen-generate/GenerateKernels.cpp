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
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/BuiltinOps.h"
using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// KernelGenerator class definition
//===----------------------------------------------------------------------===//

namespace {
class KernelGenerator {
public:
  KernelGenerator(ModuleOp primary, ModuleOp library)
      : primaryModule(primary), libraryModule(library) {}

  /// Scan the primary and library module to collect all the interfaces,
  /// verifying that any common interfaces are the same.
  ParseResult collectInterfaces();

  /// Concretize a kernel in the primary file.
  ParseResult processKernel(KernelOp kernel);

  /// Concretize all kernels in the primary file.
  ParseResult processKernels();

  /// Remove generators and generator interfaces from the file to clean it up.
  void removeGenerators();

private:
  /// These are the two modules we start with.  The primary module is mutated by
  /// our algorithm, the library module is immutable.
  ModuleOp primaryModule, libraryModule;

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

/// We expect all parameter expressions to simplify down to concrete constants,
/// we don't want anything left as a ParamOperatorAttr or ParamDeclRefAttr.
static bool isSimpleConstant(Attribute attr) {
  return attr.isa<FloatAttr, IntegerAttr, DTypeConstantAttr>();
}

namespace {
/// This class keeps a set of defined parameter values and is used to evaluate
/// and simplify operations based on those values.
class ParameterRewriter {
public:
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

  /// These are the b
  DenseMap<StringAttr, Attribute> paramValues;
};
} // end anonymous namespace

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

  // TODO: Generalize :-)
  // FIXME: 'index' folding should require target information to simplify things
  // like div.

  std::string attrStr;
  llvm::raw_string_ostream(attrStr) << expr;

  return Error("unknown expression to fold: " + attrStr);
}

LogicalResult ParameterRewriter::processOp(Operation *op) {
  if (auto bind = dyn_cast<ParamBindOp>(op))
    processParamBindOp(bind);
  else if (auto value = dyn_cast<ParamValueOp>(op))
    processParamValueOp(value);
  else
    return op->emitError("unknown operator in GenerateKernels");
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

/// Concretize a kernel in the primary file.
ParseResult KernelGenerator::processKernel(KernelOp kernel) {
  /// Get a partial ordering of parameter definitions and uses that is listed
  /// "top down" in our evaluation order.
  auto paramInfo = *ParameterDeclsAndUses::calculate(kernel);

  // TODO: When handling generators, we should bind input parameter values.
  ParameterRewriter rewriter;

  // Process each def/use in order.
  for (auto &user : paramInfo.usersAndDeclarers) {
    if (failed(rewriter.processOp(user.first)))
      return failure();
  }

  // TODO: Scan for operations that we want to lower that use parameter
  // expression that happen to not be DeclRefAttrs.  For example a call that
  // passes 42 as an argument.

  return success();
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
