//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LowerToObject.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/Compiler/OperationUtils.h"
#include "Support/ML/DType.h"
#include "Support/STLExtras.h"

using namespace M;
using namespace KGEN;

/// Get the C type string for the dtype.
static LogicalResult getCTypeForDType(FuncOp func, KGENDType dt,
                                      SmallVectorImpl<std::string> &types) {
  if (dt.isFloat()) {
    switch (dt.getValue()) {
    case DType::f32:
      types.push_back("float");
      return success();
    case DType::f64:
      types.push_back("double");
      return success();
    }
    return func.emitError("unhandled floating point dtype: ")
           << dt.getAsString();
  }
  if (dt.isInt()) {
    types.push_back(((dt.isUInt() ? "u" : "") + StringRef("int") +
                     Twine(dt.getWidthInBits()) + "_t")
                        .str());
    return success();
  }
  if (dt == DType::invalid) {
    types.push_back("void");
    return success();
  }
  if (dt.isBool()) {
    types.push_back("bool");
    return success();
  }
  if (dt.isIndex()) {
    types.push_back("ssize_t");
    return success();
  }
  return func.emitError("unhandled dtype for header generation ")
         << dt.getAsString();
};

/// Get the C type for an elementary scalar type.
static ErrorOr<std::string> getCTypeForElementary(Type t) {
  if (t.isa<IntegerType>()) {
    if (auto bitWidth = t.getIntOrFloatBitWidth(); bitWidth > 1)
      return ("int" + Twine(bitWidth) + "_t").str();
    return "bool";
  }
  if (t.isa<IndexType>())
    return "ssize_t";
  if (t.isF16())
    llvm::report_fatal_error("no support for fp16 yet");
  if (t.isF32())
    return "float";
  if (t.isF64())
    return "double";

  SmallString<128> str;
  llvm::raw_svector_ostream os(str);
  t.print(os);
  return Error("unhandled elementary type: '" + str + "'");
}

/// Get the C types for the given type.
static LogicalResult getCTypeForType(FuncOp func, Type t,
                                     SmallVectorImpl<std::string> &types) {
  if (auto simd = dyn_cast<POP::SIMDType>(t)) {
    // Since the vector_size attribute only works on GNU and CLANG compilers,
    // we pass in an array.
    if (failed(getCTypeForDType(func, *simd.getResolvedDType(), types)))
      return failure();
    auto size = *simd.getResolvedSize();
    // size == 1 is a scalar
    if (size != 1)
      types.back() += ("[" + Twine(size) + "]").str();
    return success();
  }

  if (auto ptr = dyn_cast<POP::PointerType>(t)) {
    ErrorOr<std::string> elementaryType =
        getCTypeForElementary(ptr.getElementAsType());
    // If the type is not elementary, then pass it as an opaque pointer.
    if (elementaryType.isError())
      types.push_back("void *");
    else
      types.push_back(elementaryType.takeValue() + "*");
    return success();
  }

  if (auto array = dyn_cast<POP::ArrayType>(t)) {
    if (!*array.getResolvedSize())
      return success();

    if (failed(getCTypeForType(func, array.getElementAsType(), types)))
      return failure();
    types.back() += ("[" + Twine(*array.getResolvedSize()) + "]").str();
    return success();
  }

  if (auto structType = dyn_cast<POP::StructType>(t)) {
    SmallVector<Type> elementTypes;
    elementTypes.reserve(structType.getElementTypes().size());
    if (failed(structType.resolveElementTypes(elementTypes)))
      return failure();
    for (Type elTy : elementTypes)
      if (failed(getCTypeForType(func, elTy, types)))
        return failure();
    return success();
  }

  if (auto variadic = dyn_cast<VariadicType>(t)) {
    types.push_back("void *");
    types.push_back("ssize_t");
    return success();
  }

  if (t.isa<DTypeType>()) {
    types.push_back("uint8_t");
    return success();
  }

  if (!t.isa<IndexType, IntegerType, FloatType>())
    return func.emitError("unsupported argument type: ") << t;
  if (!t.isIndex() && !llvm::isPowerOf2_64(t.getIntOrFloatBitWidth()))
    return func.emitError("integer or float bitwidth must be a power of 2");

  ErrorOr<std::string> elementaryTypeName = getCTypeForElementary(t);
  if (elementaryTypeName.isError())
    return func->emitError(elementaryTypeName.takeError().get());
  types.push_back(elementaryTypeName.takeValue());
  return success();
}

/// Emit the C signature of a KGEN func.
static LogicalResult emitSignature(raw_ostream &os, FuncOp func) {
  SmallVector<std::string> argTys, resTys;
  for (Type type : func.getArgumentTypes())
    if (failed(getCTypeForType(func, type, argTys)))
      return failure();
  for (Type type : func.getResultTypes())
    if (failed(getCTypeForType(func, type, resTys)))
      return failure();

  // Print the function declaration.
  os << "extern ";

  // If there is exactly one result type, return it. If there are multiple,
  // return them as pointers.
  if (resTys.size() == 1)
    os << resTys.front();
  else
    os << "void";

  // FIXME: This assumes the C wrapper that eventually gets generated is not
  // renamed due to a symbol name conflict. Header emission happens too early
  // in the pipeline.
  os << ' ' << func.getSymName() << '(';
  llvm::interleaveComma(argTys, os);
  if (resTys.size() > 1) {
    if (!argTys.empty())
      os << ", ";
    llvm::interleaveComma(resTys, os,
                          [&](StringRef type) { os << type << " *"; });
  }
  os << ");";
  return success();
}

LogicalResult
ObjectCompiler::produceFunctionDecls(const SymbolTable &symtab,
                                     const ExportMap &exportedSymbols,
                                     llvm::raw_ostream &os) {
  auto module = cast<ModuleOp>(symtab.getOp());
  for (auto f : module.getOps<FuncOp>()) {
    auto itExported = exportedSymbols.find(f.getNameAttr());
    if (itExported == exportedSymbols.end() || !itExported->second.isCExport)
      continue;
    // The symbol was exported, use its alias name.
    if (failed(emitSignature(os, f)))
      return mlir::emitError(f.getLoc(),
                             "during header emission for this function");
    os << "\n";
  }
  return success();
}
