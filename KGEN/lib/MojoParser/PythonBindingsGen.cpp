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
#include "MojoUtils.h"
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
  BindingGenerator(ASTDecl &moduleDecl);

  LogicalResult genPyInitImplFunc();
  void finalizePyInit();
  LogicalResult genPyInitHook();
  void genModuleBinding(
      const ArrayRef<std::pair<StringAttr, TinyPtrVector<ASTDecl *>>>
          &userFuncs);

private:
  ErrorOrSuccess genFunctionBinding(ASTDecl &funcDecl, FnOp func);
  ErrorOrSuccess genTypeBinding(ASTType type);

  OverloadSet lookupPyBindFunction(StringRef name, ASTDecl &scope,
                                   const SyntheticNode &node);
  std::pair<FnOp, ASTDecl *> createFunction(const Twine &name, ASTDecl &parent,
                                            ArrayRef<ASTType> argRValueTypes,
                                            ArrayRef<ArgConvention> convs,
                                            ASTType resultRValueType,
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
  /// The builtin `TypedPythonObject["Tuple"]` type.
  ASTType tupleTypedPyObjType;

  /// The `PyInit_impl_*` function where function and type declarations are
  /// added.
  FnOp pyInitFunc;
  /// The ASTDecl of the init function.
  ASTDecl *pyInitDecl;
  /// The mutable Python module reference.
  MLValue pyModule;

  /// The type decls for which Python bindings have been generated so far.
  DenseMap<Type, ErrorOrSuccess> generatedTypeBindings;
};
} // namespace

static StringAttr getTypeName(SharedState &shared, const ASTType &type) {
  // TODO: Figure out if the name needs to be globally unique.
  auto typeName = StringAttr::get(
      type.getAsString(/*forDiag=*/&shared, /*demangleParam=*/true),
      StringType::get(shared.getContext()));
  return typeName;
}

/// Instantiates TypedPythonObject["Tuple"]
static ASTType makeTupleTypedPythonObj(ASTDecl &moduleDecl,
                                       ASTDecl &pythonModule) {
  // Form the AST we want to emit.
  SMLoc moduleLoc = moduleDecl.getLoc();
  DeclRefNode typePythonObject("TypedPythonObject");
  StringRef tupleStr = "\"Tuple\""; // Avoid dangling pointer.
  StringLiteralNode tupleString(tupleStr);
  Operand subscriptOperand(&tupleString, moduleLoc, Operand::kPositional);
  SubscriptNode subscript(&typePythonObject, moduleLoc, subscriptOperand,
                          moduleLoc);
  // Emit it as a type.
  ExprEmitter emitter(pythonModule, ExprContext::EC_PyBindGen);
  return emitter.emitExprType(&subscript);
}

BindingGenerator::BindingGenerator(ASTDecl &moduleDecl)
    : SharedStateUser(moduleDecl.getShared()), ctx(getContext()),
      moduleLoc(moduleDecl.getLoc()), moduleDecl(moduleDecl),
      moduleOp(cast<FileModuleOp>(moduleDecl)),
      pyBindDecl(&shared.importModule("builtin._pybind",
                                      /*currentPackage=*/nullptr, moduleLoc)),
      errorType(shared.getBuiltinErrorType(moduleDecl, moduleLoc)),
      pythonModule(shared.importModule("stdlib.python.python_object",
                                       /*currentPackage=*/nullptr, moduleLoc)),
      pyObjType(
          shared.lookupNamedType("PythonObject", pythonModule, moduleLoc)),
      tupleTypedPyObjType(makeTupleTypedPythonObj(moduleDecl, pythonModule)) {}

LogicalResult BindingGenerator::genPyInitImplFunc() {
  ImplicitLocOpBuilder b(moduleOp.getLoc(), moduleDecl.getDeclEndBuilder());

  // Create a function in the form:
  //
  //   fn PyInit_impl_<MODULE_NAME>() raises -> PythonObject as module:
  //       module = create_pybind_module["<MODULE_NAME>"]()
  //
  StringRef moduleName = moduleOp.getSymName();

  std::tie(pyInitFunc, pyInitDecl) = createFunction(
      "PyInit_impl_" + moduleName, moduleDecl, /*argTypes=*/{}, /*convs=*/{},
      /*resultType=*/pyObjType, FnEffects().setThrows());

  // Generate the module object. Form the AST we want to emit.
  DeclRefNode typePythonObject("create_pybind_module");
  SyntheticNode moduleNameVal(
      moduleLoc, StringAttr::get(moduleName, StringType::get(ctx)));
  Operand subscriptOperand(&moduleNameVal, moduleLoc, Operand::kPositional);
  SubscriptNode subscript(&typePythonObject, moduleLoc, subscriptOperand,
                          moduleLoc);
  CallNode call(&subscript, moduleLoc, {}, moduleLoc);

  // Emit it.
  ExprEmitter emitter(*pyBindDecl, OpBuilder::atBlockEnd(pyInitFunc.getBody()));
  pyModule = MLValue(pyInitFunc.getArgument(1));
  ValueDest moduleDest(pyModule, EC_PyBindGen);
  if (!emitter.emitExpr(&call, moduleDest))
    return failure();
  return success();
}

void BindingGenerator::finalizePyInit() {
  // Emit the terminator for the function.
  ImplicitLocOpBuilder b(pyInitFunc.getLoc(),
                         OpBuilder::atBlockEnd(pyInitFunc.getBody()));
  // Return none.
  ExprEmitter::emitNormalReturn(b);
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

  ExprEmitter emitter(*pyInitHookDecl,
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
                     ParamBindings(*pyInitHookDecl), &node,
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

  // Note that genFunctionBinding and genTypeBinding later append to this init
  // function too.

  return success();
}

void BindingGenerator::genModuleBinding(
    const ArrayRef<std::pair<StringAttr, TinyPtrVector<ASTDecl *>>>
        &userFuncs) {
  // Scan all the functions in the module. Find functions for which generation
  // is currently supported. Keep track of all decls that were rejected and for
  // what reason. The binding generator will add an erroneous decl to the
  // generated module so that users accessing them from Python will receive the
  // error message.
  SmallVector<std::pair<StringAttr, StringRef>> rejectedDecls;
  for (auto [name, decls] : userFuncs) {
    assert(!decls.empty() && "name mapped to empty decl?");
    auto func = dyn_cast<FnOp>(decls.front());
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
    FnTypeGeneratorType sig = func.getFuncTypeGenerator();
    if (sig.getFnEffects() != FnEffects()) {
      rejectedDecls.emplace_back(name,
                                 "TODO: Python binding generation is not "
                                 "supported for raising or async functions");
      continue;
    }

    // Reject parametric functions. Supporting these is far more complicated!
    if (sig.getInputParamTypes().size()) {
      rejectedDecls.emplace_back(name, "Python binding generation is not "
                                       "supported for parameter functions");
      continue;
    }

    if (auto err = genFunctionBinding(*decls.front(), func))
      rejectedDecls.emplace_back(name, err.getError());
  }
}

ErrorOrSuccess BindingGenerator::genFunctionBinding(ASTDecl &funcDecl,
                                                    FnOp func) {
  // First, generate type bindings for each of the function argument and result
  // types.
  FnTypeGeneratorType sig = func.getFuncTypeGenerator();
  for (auto [type, conv] :
       llvm::zip(func.getArgumentTypes(), sig.getArgConventions())) {
    ASTType rvType = getFunctionArgumentRValueType(type, conv);

    // Check if bindings or errors were already generated for this type.
    auto it = generatedTypeBindings.find(rvType);
    if (it == generatedTypeBindings.end()) {
      ErrorOrSuccess err = genTypeBinding(rvType);
      it = generatedTypeBindings.insert({rvType, std::move(err)}).first;
    }
    if (auto err = it->second.copy())
      return err.takeError();
  }

  //------------------------------------
  // Next, generate the wrapper function
  //------------------------------------

  //
  // NOTE:
  //  This code needs to emit the equivalent to the Mojo code in
  //  feature-overview/mojo_module.mojo#incr_int__wrapper(), i.e.
  //  for a Mojo function with a signature like:
  //
  //      fn incr_int(mut value: Int):
  //          value += 1
  //
  //  we must emit a wrapper that exposes that Mojo function to CPython:
  //
  //    fn incr_int__wrapper(
  //        py_self: PythonObject,
  //        py_args: TypedPythonObject["Tuple"],
  //    ) raises -> PythonObject:
  //        check_arguments_arity("incr_int", 1, py_args)
  //
  //        var arg_0: UnsafePointer[Int] = check_and_get_arg[Int](
  //            "incr_int",
  //            "Int",
  //            py_args,
  //            0
  //        )
  //
  //        incr_int(arg_0[])
  //
  //        return PythonObject(None)
  //
  //  TODO(MSTDL-997): Generate bindings for non-`None` returning Mojo functions
  //    The current outline of this logic will only work for wrapping Mojo
  //    functions that return None. Returning new values will require
  //    implementing lookup of the runtime generated PyTypeBinding values
  //    (not impossible, just not as easy).
  //

  auto originalFuncName = func.getDeclName().str();
  auto wrapperFuncName = originalFuncName + "__wrapper";

  auto [wrapperFunc, wrapperDecl] = createFunction(
      wrapperFuncName, moduleDecl,
      /*argTypes=*/{pyObjType, tupleTypedPyObjType},    /*convs=*/
      {ArgConvention::ReadMem, ArgConvention::ReadMem}, // DO NOT SUBMIT
      /*resultType=*/pyObjType, FnEffects().setThrows());

  mlir::ImplicitLocOpBuilder builder(
      shared.translateLocation(funcDecl.getLoc()),
      wrapperDecl->getDeclEndBuilder());

  // Export the function.
  NamedAttrList attrs = wrapperFunc->getAttrDictionary();
  attrs.set(wrapperFunc.getExportKindAttrName(),
            ExportKindAttr::get(ctx, ExportKind::CExported));
  attrs.set(wrapperFunc.getLinkageNameAttrName(),
            builder.getStringAttr(wrapperFuncName));
  wrapperFunc->setAttrs(attrs.getDictionary(ctx));

  if (sig.getResultType() != shared.getNoneType()) {
    return Error("functions that return values are not supported");
  }

  AnyValue pyArgsTuple = MBValue(wrapperFunc.getArgument(1));

  ExprEmitter emitter(*pyBindDecl, builder);

  SyntheticNode synth(funcDecl.getLoc());

  // Bind the type and name parameters. Since parametric functions are
  // forbidden, the type and its parameters will be concrete.
  // FIXME(MOCO-1306): The name parameter should not be required.
  // TODO: Figure out if the name needs to be globally unique.
  PValue originalFuncNameStrAttr =
      StringAttr::get(originalFuncName, StringType::get(ctx));

  //
  // Emit `check_arguments_arity("<func name>", <arg count>, <2nd arg>)
  //

  {
    OverloadSet checkArgumentsArityOv =
        lookupPyBindFunction("check_arguments_arity", *wrapperDecl, synth);

    AnyValue funcArityAttr =
        IntegerAttr::get(IndexType::get(ctx), func.getArgumentTypes().size());

    ValueDest noneDest(EC_PyBindGen);
    checkArgumentsArityOv.emitCall(CallOperands({
                                       {originalFuncNameStrAttr, synth},
                                       {funcArityAttr, synth},
                                       {pyArgsTuple, synth},
                                   }),
                                   noneDest, emitter);
  }

  //
  // For each arg, emit:
  //     var arg_<N>: UnsafePointer[<N arg type>] =
  //         check_and_get_arg[<N arg type>](
  //             "<func name>",
  //             "<arg type name>",
  //             py_args,
  //             <N>
  //         )
  //

  std::vector<VarDeclOp> argUnsafePointerVars;
  for (auto [argIndex, type, conv] :
       llvm::enumerate(func.getArgumentTypes(), sig.getArgConventions())) {
    ASTType rvType = getFunctionArgumentRValueType(type, conv);

    // We emit into a vardecl and infer the type of it.
    std::string varName = std::string("arg_") + std::to_string(argIndex);
    VarDeclOp argUnsafePointerVar =
        emitter.emitVarDecl(varName, UnresolvedType::get(shared.getContext()),
                            builder.getLoc(), VarDeclKind::Synthesized);

    DeclRefNode nameDRE("check_and_get_arg");
    SyntheticNode argTypeVal(moduleLoc, rvType);
    Operand argTypeOp(&argTypeVal, moduleLoc, Operand::kPositional);
    SubscriptNode subscript(&nameDRE, moduleLoc, argTypeOp, moduleLoc);

    SyntheticNode callOp0(moduleLoc, originalFuncNameStrAttr);
    SyntheticNode callOp1(moduleLoc, getTypeName(shared, rvType));
    SyntheticNode callOp2(moduleLoc, pyArgsTuple);
    SyntheticNode callOp3(moduleLoc,
                          IntegerAttr::get(IndexType::get(ctx), argIndex));
    Operand callOps[] = {
        Operand(&callOp0, moduleLoc, Operand::kPositional),
        Operand(&callOp1, moduleLoc, Operand::kPositional),
        Operand(&callOp2, moduleLoc, Operand::kPositional),
        Operand(&callOp3, moduleLoc, Operand::kPositional),
    };
    CallNode call(&subscript, moduleLoc, callOps, moduleLoc);

    // Emit the call into the vardecl.
    ValueDest argUnsafePointerDest(argUnsafePointerVar, EC_PyBindGen);
    if (!emitter.emitExpr(&call, argUnsafePointerDest))
      return Error("Error emitting 'check_and_get_arg' call");

    argUnsafePointerVars.emplace_back(std::move(argUnsafePointerVar));
  }

  //
  // For each arg, call __getitem__ on it to dereference the UnsafePointer into
  // a reference.
  //

  std::vector<ASTExprAnd<AnyValue>> funcCallArgs;
  for (auto &argUnsafePointerVar : argUnsafePointerVars) {
    ValueDest argUnsafeRefDest(EC_PyBindGen);
    CValue cvalue = emitter.emitNamedMethodCall(
        "__getitem__", CallOperands({{MLValue(argUnsafePointerVar), synth}}),
        argUnsafeRefDest, CallSyntax::kMethodCall, synth);
    funcCallArgs.push_back({cvalue, synth});
  }

  //
  // Finally, call the user's function with all those dereferenced args.
  //

  OverloadSet funcOv(func.getDeclName(), {&funcDecl}, ParamBindings(funcDecl),
                     synth, CallSyntax::kDirectCall);
  ValueDest funcCallNoneDest(EC_PyBindGen);
  funcOv.emitCall(CallOperands(funcCallArgs), funcCallNoneDest, emitter);

  // TODO: get this and below in sync with createFunction, this only supports
  // memory only results so far.
  assert(wrapperFunc.getFuncTypeGenerator().hasMemoryOnlyResult());

  //
  // Emit a return None
  //

  ValueDest returnDest(pyObjType, EC_PyBindGen);
  returnDest =
      ValueDest(MLValue(wrapperFunc.getArguments().back()), EC_ReturnValue);
  AnyValue noneThing =
      emitter.emitPValue({shared.getNoneAttr(), synth}, EC_PyBindGen);
  CValue ctorResult = emitter.emitConstructorCall(
      pyObjType, CallOperands{ASTExprAnd<AnyValue>{noneThing, synth}}, synth,
      CallSyntax::kTypeCall, returnDest);
  if (!ctorResult)
    return {};

  emitter.emitNormalReturn(func.getLoc());

  //
  // Into PyInit_my_module, emit:
  //     add_wrapper_to_module[wrapperFunc, "incr_int"](module)
  //
  // Above here we were emitting into the wrapper function for the user's
  // function, but we're emitting this call into e.g. PyInit_my_module.
  DeclRefNode typePythonObject("add_wrapper_to_module");
  SyntheticNode funcPointerVal(moduleLoc, wrapperDecl->getFuncAsPValue());
  SyntheticNode funcNameVal(moduleLoc, originalFuncNameStrAttr);

  Operand subscriptOperands[] = {
      Operand(&funcPointerVal, moduleLoc, Operand::kPositional),
      Operand(&funcNameVal, moduleLoc, Operand::kPositional)};

  SubscriptNode subscript(&typePythonObject, moduleLoc, subscriptOperands,
                          moduleLoc);
  SyntheticNode moduleVal(moduleLoc, pyModule);
  Operand moduleOp(&moduleVal, moduleLoc, Operand::kPositional);
  CallNode call(&subscript, moduleLoc, moduleOp, moduleLoc);
  ExprEmitter pyInitEmitter(*pyBindDecl,
                            OpBuilder::atBlockEnd(pyInitFunc.getBody()));
  pyInitEmitter.emitExpr(&call, EC_PyBindGen);
  return success();
}

ErrorOrSuccess BindingGenerator::genTypeBinding(ASTType type) {
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
  // optimization) not perform duplicate work.
  ASTDecl *typeDecl = type.getDecl(shared);
  auto structDecl = dyn_cast_or_null<StructDeclOp>(typeDecl);
  if (!structDecl)
    return Error("TODO: type binding generation only supported for structs");

  // Use the location of the current type for better error reporting.
  SMLoc loc = typeDecl->getLoc();

  // Generate the module object. Form the AST we want to emit.
  DeclRefNode typePythonObject("gen_pytype_wrapper");
  SyntheticNode moduleNameVal(loc, type);
  SyntheticNode typeNameVal(loc, getTypeName(shared, type));
  Operand subscriptOps[] = {
      Operand(&moduleNameVal, loc, Operand::kPositional),
      Operand(&typeNameVal, loc, Operand::kPositional),
  };
  SubscriptNode subscript(&typePythonObject, loc, subscriptOps, loc);

  SyntheticNode moduleVal(loc, pyModule);
  Operand moduleOp(&moduleVal, loc, Operand::kPositional);
  CallNode call(&subscript, loc, moduleOp, loc);

  // Emit it.
  ExprEmitter emitter(*pyBindDecl, OpBuilder::atBlockEnd(pyInitFunc.getBody()));
  if (!emitter.emitExpr(&call, ExprContext::EC_PyBindGen))
    return Error("Error emitting 'gen_pytype_wrapper' call");
  return success();
}

OverloadSet BindingGenerator::lookupPyBindFunction(StringRef name,
                                                   ASTDecl &scope,
                                                   const SyntheticNode &node) {
  ArrayRef<ASTDecl *> fnDecls =
      shared.getBuiltinFunction(*pyBindDecl, name, scope.getLoc());
  ParamBindings bindings(scope);
  return OverloadSet(name, fnDecls, std::move(bindings), &node,
                     CallSyntax::kDirectCall);
}

std::pair<FnOp, ASTDecl *>
BindingGenerator::createFunction(const Twine &name, ASTDecl &parent,
                                 ArrayRef<ASTType> argRValueTypes,
                                 ArrayRef<ArgConvention> convs,
                                 ASTType resultRValueType, FnEffects effects) {
  SmallVector<Type> argTypes;
  SmallVector<StringAttr> argNames;
  ImplicitLocOpBuilder b(translateLocation(parent.getLoc()),
                         parent.getDeclEndBuilder());

  // Helper to make a `!lit.ref` type. `synthesizeFunction` will make the
  // implicit origin declarations.
  auto makeRefType = [&](ASTType type, const Twine &name, bool isMut) {
    return RefType::get(type,
                        ParamDeclRefAttr::get(b.getStringAttr(name),
                                              OriginType::get(ctx, isMut)));
  };

  for (auto [idx, rvType, conv] : llvm::enumerate(argRValueTypes, convs)) {
    Type type = rvType;
    argNames.push_back(b.getStringAttr("arg" + Twine(idx)));
    if (hasImplicitOrigin(conv)) {
      bool isMut = llvm::is_contained(
          {ArgConvention::OwnedMem, ArgConvention::Mut, ArgConvention::MutRef,
           ArgConvention::ByRefResult},
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
      PogListAttr::toPogs(argNames, passingKinds, /*no variadics*/ {});

  StructEmitter emitter(shared);
  return emitter.synthesizeFunction(
      parent, name.str(), /*params=*/{}, /*paramList=*/PogListAttr::get(ctx),
      argTypes, adjConvs, PogListAttr::get(ctx, argList), resultType,
      SpecialFunctionKind::kNormal, parent.getLoc(), b, effects);
}

LogicalResult LIT::generatePythonBindings(ASTDecl &moduleDecl) {
  // Don't generate debuginfo.
  DebugInfo::DIBuilder::ScopeGuard scopeGuard;
  if (auto &dibuilder = moduleDecl.getShared().diBuilder)
    scopeGuard = dibuilder->pushScopeGuard(/*scope=*/nullptr);

  BindingGenerator gen(moduleDecl);

  ArrayRef<std::pair<StringAttr, TinyPtrVector<ASTDecl *>>> userFuncs =
      moduleDecl.getDeclsInScope();

  if (failed(gen.genPyInitImplFunc()))
    return failure();
  if (failed(gen.genPyInitHook()))
    return failure();
  gen.genModuleBinding(userFuncs);
  gen.finalizePyInit();

  return success();
}
