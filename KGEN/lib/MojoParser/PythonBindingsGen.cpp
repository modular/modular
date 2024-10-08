//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file defines the logic for automatically generating Python bindings for
// Mojo functions and types.
//
//===----------------------------------------------------------------------===//

#include "PythonBindingsGen.h"
#include "CallEmission.h"
#include "ExprEmitter.h"
#include "ExprNodes.h"
#include "StructEmitter.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/SharedState.h"

#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

namespace {
class BindingGenerator : public SharedStateUser {
public:
  BindingGenerator(SharedState &shared, ASTDecl &moduleDecl);

  LogicalResult genPyInitImplFunc();
  LogicalResult genPyInitHook();
  void genModuleBinding();

private:
  ErrorOrSuccess genFunctionBinding(ASTDecl &funcDecl, LIT::FuncOp func);
  ErrorOrSuccess genTypeBinding(ASTDecl &typeDecl, StructDeclOp type);

  OverloadSet lookupPyBindFunction(StringRef name, ASTDecl &scope,
                                   const SyntheticNode &node);
  StringAttr getMLIRString(const Twine &value);
  std::pair<LIT::FuncOp, ASTDecl *>
  createFunction(const Twine &name, ASTDecl &parent,
                 ArrayRef<ASTType> argRValueTypes,
                 ArrayRef<ArgConvention> convs, ASTType resultRValueType,
                 FnEffects effects);

  /// Shorthand to access the MLIR context.
  MLIRContext *ctx;
  /// The location of the module.
  SMLoc moduleLoc;
  /// The ASTDecl of the module for which bindings are being generated.
  ASTDecl &moduleDecl;
  /// The module operation for which bindings are being generated.
  FileModuleOp moduleOp;
  /// The `_pybind` module, where all the stubs live.
  ASTDecl *pyBindDecl;
  /// The builtin `Error` type.
  ASTType errorType;
  /// The `python` module.
  ASTDecl &pythonModule;
  /// The builtin `PythonObject` type.
  ASTType pyObjType;

  /// The `PyInit_impl_*` function where function and type declarations are
  /// added.
  LIT::FuncOp pyInitFunc;
  /// The ASTDecl of the init function.
  ASTDecl *pyInitDecl;
};
} // namespace

BindingGenerator::BindingGenerator(SharedState &shared, ASTDecl &moduleDecl)
    : SharedStateUser(shared), ctx(getContext()),
      moduleLoc(moduleDecl.getLoc()), moduleDecl(moduleDecl),
      moduleOp(cast<FileModuleOp>(moduleDecl)),
      pyBindDecl(&shared.importModule("builtin._pybind",
                                      /*currentPackage=*/nullptr, moduleLoc)),
      errorType(shared.getBuiltinErrorType(moduleDecl, moduleLoc)),
      pythonModule(shared.importModule("stdlib.python.python_object",
                                       /*currentPackage=*/nullptr, moduleLoc)),
      pyObjType(
          shared.lookupNamedType("PythonObject", pythonModule, moduleLoc)) {}

LogicalResult BindingGenerator::genPyInitImplFunc() {
  ImplicitLocOpBuilder b(moduleOp.getLoc(), moduleDecl.getDeclEndBuilder());
  SMLoc loc = moduleLoc;

  // Create a function in the form:
  //
  //   fn PyInit_impl_<MODULE_NAME>() raises -> PythonObject:
  //       module = create_pybind_module["<MODULE_NAME>"]()
  //

  StringRef moduleName = moduleOp.getSymName();

  // Declare the error and result slots.
  SmallVector<PogMetadataAttr> argList = PogListAttr::toPogs(
      {b.getStringAttr("__error__"), b.getStringAttr("__result__")},
      {PassingKind::Implicit, PassingKind::Implicit}, {});

  std::tie(pyInitFunc, pyInitDecl) = createFunction(
      "PyInit_impl_" + moduleName, moduleDecl, /*argTypes=*/{}, /*convs=*/{},
      /*resultType=*/pyObjType, FnEffects().setThrows());

  // Generate the module object.
  ExprEmitter emitter(shared, *pyInitDecl,
                      OpBuilder::atBlockEnd(pyInitFunc.getBody()));
  SyntheticNode node(loc);
  OverloadSet createModuleOv =
      lookupPyBindFunction("create_pybind_module", *pyInitDecl, node);

  // Emit the call into the result slot.
  createModuleOv.paramBindings.add(node, getMLIRString(moduleName));
  ValueDest moduleDest(MLValue(pyInitFunc.getArgument(1)), EC_PyBindGen);
  CValue moduleValue =
      createModuleOv.emitCall(CallOperands(), moduleDest, emitter);
  if (!moduleValue)
    return failure();

  // Emit the terminator for the function.
  b.setInsertionPointToEnd(pyInitFunc.getBody());
  b.create<KGEN::ReturnOp>(
      Value(b.create<ParamConstantOp>(b.getBoolAttr(false))));

  return success();
}

LogicalResult BindingGenerator::genPyInitHook() {
  SMLoc loc = moduleLoc;

  // Now create the real `PyInit_<MODULE_NAME>` function. This one needs to have
  // a precise signature.
  if (!pyObjType.isRegisterPassable(loc, shared)) {
    emitError(
        loc, "internal error: expected 'PythonObject' to be register-passable");
    return failure();
  }

  // Emit the outer function, which propagates any Mojo errors into a Python
  // exception:
  //
  //   fn PyInit_<MODULE_NAME>() -> PythonObject:
  //       var module: PythonObject
  //       try:
  //           module = PyInit_impl_<MODULE_NAME>()
  //       except e:
  //           TODO: Propagate the error.
  //       return module^
  //

  // FIXME(MOCO-1296): The module name has to be verified against allowed C
  // linkage names, and we might consider 's/-/_/g'.
  auto [pyInitHook, pyInitHookDecl] =
      createFunction("PyInit_" + moduleOp.getSymName(), moduleDecl,
                     /*argTypes=*/{}, /*convs=*/{}, pyObjType, FnEffects());

  ImplicitLocOpBuilder b(moduleOp.getLoc(),
                         OpBuilder::atBlockBegin(pyInitHook.getBody()));

  // Export the function.
  NamedAttrList attrs = pyInitHook->getAttrDictionary();
  attrs.set(pyInitHook.getExportKindAttrName(),
            ExportKindAttr::get(ctx, ExportKind::CExported));
  attrs.set(pyInitHook.getLinkageNameAttrName(),
            b.getStringAttr("PyInit_" + moduleOp.getSymName()));
  pyInitHook->setAttrs(attrs.getDictionary(ctx));

  ExprEmitter emitter(shared, *pyInitHookDecl,
                      OpBuilder::atBlockBegin(pyInitHook.getBody()));
  VarDeclOp moduleDecl = emitter.emitVarDecl(
      "module", UnresolvedType::get(ctx), b.getLoc(), VarDeclKind::Synthesized);
  VarDeclOp errDecl = emitter.emitVarDecl("error", errorType, b.getLoc(),
                                          VarDeclKind::Synthesized);

  // Generate empty 'else' and 'finally' regions.
  b.setInsertionPointToEnd(pyInitHook.getBody());
  auto tryOp = b.create<TryOp>(errDecl);
  b.createBlock(&tryOp.getElseRegion());
  b.create<TryYieldOp>();
  b.createBlock(&tryOp.getFinallyRegion());
  b.create<TryYieldOp>();

  // In the try region, emit the call to `PyInit_impl_*`.
  Block *tryBlock = b.createBlock(&tryOp.getTryRegion());
  SyntheticNode node(loc);
  OverloadSet initOv(pyInitFunc.getDeclName(), pyInitDecl,
                     ParamBindings({*pyInitHookDecl, shared}), &node,
                     CallSyntax::kDirectCall);
  ValueDest moduleDest(moduleDecl, EC_PyBindGen);
  emitter.builder = b;
  initOv.emitCall(CallOperands(), moduleDest, emitter);
  b.setInsertionPointToEnd(tryBlock);
  b.create<TryYieldOp>();

  // Emit the error handling logic.
  Block *exceptBlock = b.createBlock(&tryOp.getExceptRegion());
  OverloadSet errOv =
      lookupPyBindFunction("fail_initialization", *pyInitHookDecl, node);
  ValueDest errDest(EC_PyBindGen);
  emitter.builder = b;
  CValue nullResult = errOv.emitCall(CallOperands({{MRValue(errDecl), &node}}),
                                     errDest, emitter);
  if (!nullResult)
    return failure();
  SRValue nullResultSR = emitter.emitSRValue({nullResult, &node}, EC_PyBindGen);
  if (!nullResultSR)
    return failure();
  b.setInsertionPointToEnd(exceptBlock);
  b.create<KGEN::ReturnOp>(nullResultSR);

  b.setInsertionPointAfter(tryOp);
  emitter.builder = b;
  SRValue result =
      emitter.emitSRValue({MRValue(moduleDecl), node}, EC_PyBindGen);
  if (!result)
    return failure();
  b.setInsertionPointToEnd(pyInitHook.getBody());
  b.create<KGEN::ReturnOp>(result);
  return success();
}

void BindingGenerator::genModuleBinding() {
  // Scan all the functions in the module. Find functions for which generation
  // is currently supported. Keep track of all decls that were rejected and for
  // what reason. The binding generator will add an erroneous decl to the
  // generated module so that users accessing them from Python will receive the
  // error message.
  SmallVector<std::pair<StringAttr, StringRef>> rejectedDecls;
  for (auto [name, decls] : moduleDecl.getDeclsInScope()) {
    assert(!decls.empty() && "name mapped to empty decl?");
    auto func = dyn_cast<LIT::FuncOp>(decls.front());
    if (!func) {
      rejectedDecls.emplace_back(name, "TODO: Python binding generation is only"
                                       " supported for functions");
      continue;
    }
    if (decls.size() != 1) {
      rejectedDecls.emplace_back(name, "TODO: Python binding generation is not "
                                       "supported for overloaded functions");
      continue;
    }

    // TODO(MOCO-1298, MOCO-1299): Reject any functions with non-default
    // effects.
    LITSignatureType sig = func.getSignature();
    if (sig.getFnEffects() != FnEffects()) {
      rejectedDecls.emplace_back(name,
                                 "TODO: Python binding generation is not "
                                 "supported for raising or async functions");
      continue;
    }

    // Reject parametric functions. Supporting these is far more complicated!
    if (sig.getNumParams()) {
      rejectedDecls.emplace_back(name, "Python binding generation is not "
                                       "supported for parameter functions");
      continue;
    }

    if (auto err = genFunctionBinding(*decls.front(), func))
      rejectedDecls.emplace_back(name, err.getError());
  }
}

ErrorOrSuccess BindingGenerator::genFunctionBinding(ASTDecl &funcDecl,
                                                    LIT::FuncOp func) {
  // First, generate type bindings for each of the function argument and result
  // types.
  return success();
}

ErrorOrSuccess BindingGenerator::genTypeBinding(ASTDecl &typeDecl,
                                                StructDeclOp type) {
  // TODO(MOCO-1301, MOCO-1302, MOCO-1303, MOCO-1304, MOCO-1305): Only basic
  // wrappers are generated for the types right now.
  // FIXME(MOCO-1300): Generating Python typedefs for all transitively used
  // types in this module will create duplicate type definitions if the type is
  // used in two different binding modules. Consider:
  //
  // # foo.mojo
  // fn make_int() -> Int: ...
  //
  // # bar.mojo
  // fn use_int(x: Int): ...
  //
  // # main.py
  // from foo import make_int
  // from bar import use_int
  //
  // val = make_int()
  // bar = use_int(val) # might type error, crash, or something worse!
  //
  // Binding generation needs to be on a per-module basis, and (as an
  // optimiation) not perform duplicate work.
  return success();
}

OverloadSet BindingGenerator::lookupPyBindFunction(StringRef name,
                                                   ASTDecl &scope,
                                                   const SyntheticNode &node) {
  ArrayRef<ASTDecl *> fnDecls =
      shared.getBuiltinFunction(*pyBindDecl, name, scope.getLoc());
  ParamBindings bindings(TypeCheckScopeInfo{scope, shared});
  return OverloadSet(name, fnDecls, std::move(bindings), &node,
                     CallSyntax::kDirectCall);
}

StringAttr BindingGenerator::getMLIRString(const Twine &value) {
  return StringAttr::get(value, StringType::get(ctx));
}

std::pair<LIT::FuncOp, ASTDecl *>
BindingGenerator::createFunction(const Twine &name, ASTDecl &parent,
                                 ArrayRef<ASTType> argRValueTypes,
                                 ArrayRef<ArgConvention> convs,
                                 ASTType resultRValueType, FnEffects effects) {
  SmallVector<Type> argTypes;
  SmallVector<StringAttr> argNames;
  ImplicitLocOpBuilder b(translateLocation(parent.getLoc()),
                         parent.getDeclEndBuilder());

  // Helper to make a `!lit.ref` type. `synthesizeFunction` will make the
  // implicit lifetime declarations.
  auto makeRefType = [&](ASTType type, const Twine &name, bool isMut) {
    return RefType::get(type,
                        ParamDeclRefAttr::get(b.getStringAttr(name),
                                              LifetimeType::get(ctx, isMut)));
  };

  for (auto [idx, rvType, conv] : llvm::enumerate(argRValueTypes, convs)) {
    Type type = rvType;
    argNames.push_back(b.getStringAttr("arg" + Twine(idx)));
    if (SignatureType::hasImplicitLifetime(conv)) {
      bool isMut =
          llvm::is_contained({ArgConvention::OwnedInMem, ArgConvention::InOut,
                              ArgConvention::MutRef, ArgConvention::InitSelf},
                             conv);
      type = makeRefType(type, "arg`" + Twine(idx), isMut);
    }
    argTypes.push_back(type);
  }

  SmallVector<ArgConvention> adjConvs = llvm::to_vector(convs);
  SmallVector<PassingKind> passingKinds(argTypes.size(), PassingKind::PosOrKw);

  Type resultType = resultRValueType;
  if (effects.isThrows()) {
    argTypes.push_back(makeRefType(errorType, "error`", true));
    argNames.push_back(b.getStringAttr("error"));
    adjConvs.push_back(ArgConvention::ByRefError);
    passingKinds.push_back(PassingKind::Implicit);
    resultType = b.getI1Type();
  }

  if (effects.isThrows() ||
      !resultRValueType.isRegisterPassable(parent.getLoc(), shared)) {
    argTypes.push_back(makeRefType(resultRValueType, "result", true));
    argNames.push_back(b.getStringAttr("result"));
    adjConvs.push_back(ArgConvention::ByRefResult);
    passingKinds.push_back(PassingKind::Implicit);
    if (!effects.isThrows())
      resultType = shared.getNoneType();
  }

  SmallVector<PogMetadataAttr> argList =
      PogListAttr::toPogs(argNames, passingKinds, /*variadicIndices=*/{});

  StructEmitter emitter(shared);
  return emitter.synthesizeFunction(
      parent, name.str(), /*params=*/{}, /*paramList=*/PogListAttr::get(ctx),
      argTypes, adjConvs, PogListAttr::get(ctx, argList), resultType,
      SpecialFunctionKind::kNormal, parent.getLoc(), b, effects);
}

LogicalResult LIT::generatePythonBindings(SharedState &shared,
                                          ASTDecl &moduleDecl) {
  // Don't generate debuginfo.
  DebugInfo::DIBuilder::ScopeGuard scopeGuard;
  if (shared.diBuilder)
    scopeGuard = shared.diBuilder->pushScopeGuard(/*scope=*/nullptr);

  BindingGenerator gen(shared, moduleDecl);

  if (failed(gen.genPyInitImplFunc()))
    return failure();
  if (failed(gen.genPyInitHook()))
    return failure();
  gen.genModuleBinding();

  return success();
}
