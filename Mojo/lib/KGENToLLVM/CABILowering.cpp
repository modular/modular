//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "CABILowering.h"
#include "CABIAAPCS.h"
#include "CABISystemV.h"
#include "LLVMLoweringUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

using namespace M;
using namespace M::KGEN;

namespace {
/// Command-line options for C ABI lowering.
struct CABIOptions {
  llvm::cl::opt<bool> skipCoercion{
      "skip-c-abi-coercion",
      llvm::cl::desc("Disable C ABI struct coercion (for debugging)"),
      llvm::cl::init(false), llvm::cl::Hidden};
};
} // namespace

static llvm::ManagedStatic<CABIOptions> CABIOpts;

//===----------------------------------------------------------------------===//
// CABIInfo base implementation
//===----------------------------------------------------------------------===//

SignatureClassification
CABIInfo::computeSignatureInfo(mlir::TypeRange argTypes, mlir::Type retType,
                               mlir::Location loc, CABIRole role,
                               size_t numFixedArgs) const {
  SignatureClassification result =
      classifySignature(argTypes, retType, loc, numFixedArgs);
  // FIXME(MOCO-5129) - Here be dragons!
  // Because we flatten single-element structs early on during LowerLIT, we
  // can't distinguish between the C equivalents of passing 'char' vs 'struct {
  // char; }'. These need to be lowered differently - 'i8 signext' vs 'i8'.
  // As such we must be conservatively correct and strip extension attributes
  // differently depending on Mojo's role:
  if (role.isCaller) {
    // If we were to set an extension attr on the *return* we get back from C,
    // our codegen would be able to assume it's extended when it may not have
    // been:
    //   %a = call signext i8 @some_c_fn()
    //   %b = sext i8 to i64
    // If, in C, we have defined struct { char c; } some_c_fn() { ... }
    // then we have not in fact received a sign-extended integer! The code
    // generator would essentially be able to elide the sext instruction.
    for (auto [info, isSigned] : llvm::zip_equal(result.args, role.isArgSigned))
      info.applySignedness(isSigned);
    result.ret.ext = IntExtKind::None;
  } else {
    // If we were to set extension attrs on the *arguments* of an exported
    // function (i.e., we are defining the callee), our codegen could assume
    // that arguments have been extended by the C-side caller when they may not
    // have been:
    //
    //   define i32 @some_exported_fn(signext i8 %arg) {
    //     %a = sext i8 to i32
    //     ret i32 %a
    //   }
    //
    // If a C program was in fact:
    //   extern int some_exported_fn(struct { char; });
    //
    // Then it would not extend arguments before calling into Mojo. Our codegen
    // would then be able to elide the extension incorrectly!
    for (CoercionInfo &info : result.args)
      info.ext = IntExtKind::None;
    result.ret.applySignedness(role.isResultSigned);
  }
  return result;
}

SignatureClassification CABIInfo::classifySignature(mlir::TypeRange argTypes,
                                                    mlir::Type retType,
                                                    mlir::Location loc,
                                                    size_t numFixedArgs) const {
  SignatureClassification result;
  if (retType)
    result.ret = classifyReturnType(retType, loc);
  result.args.reserve(argTypes.size());
  for (auto [idx, type] : llvm::enumerate(argTypes)) {
    bool isVariadicArg = idx >= numFixedArgs;
    result.args.push_back(classifyArgumentType(type, loc, isVariadicArg));
  }
  return result;
}

//===----------------------------------------------------------------------===//
// SignatureClassification
//===----------------------------------------------------------------------===//

unsigned SignatureClassification::getLLVMParamIndex(size_t argIdx) const {
  assert(argIdx <= args.size() && "argument index out of range");
  unsigned paramIdx = ret.useSRet ? 1 : 0;
  for (const CoercionInfo &coercion : llvm::ArrayRef(args).take_front(argIdx))
    paramIdx += coercion.getNumLLVMParams();
  return paramIdx;
}

mlir::LLVM::LLVMFunctionType SignatureClassification::getFunctionType(
    mlir::MLIRContext *ctx, mlir::TypeRange argTypes, mlir::Type retType,
    bool isVarArg, size_t numFixedArgs) const {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(ctx);
  llvm::SmallVector<mlir::Type> paramTypes;
  if (ret.useSRet)
    paramTypes.push_back(ptrType);
  for (auto [coercion, type] :
       llvm::zip(llvm::ArrayRef(args).take_front(numFixedArgs), argTypes)) {
    if (coercion.isIdentity()) {
      paramTypes.push_back(type);
    } else if (coercion.useIndirect) {
      paramTypes.push_back(ptrType);
    } else {
      assert(coercion.coercedType &&
             "non-identity, non-indirect coercion must have a coercedType");
      paramTypes.push_back(coercion.coercedType);
      if (coercion.isTwoRegister()) {
        assert(coercion.coercedSecondType &&
               "two-register classification must have coercedSecondType");
        paramTypes.push_back(coercion.coercedSecondType);
      }
    }
  }

  mlir::Type resultType;
  if (ret.useSRet) {
    resultType = mlir::LLVM::LLVMVoidType::get(ctx);
  } else if (ret.isTwoRegister()) {
    assert(ret.coercedType && ret.coercedSecondType &&
           "two-register classification must have both coerced types");
    resultType = mlir::LLVM::LLVMStructType::getLiteral(
        ctx, {ret.coercedType, ret.coercedSecondType});
  } else if (ret.coercedType) {
    resultType = ret.coercedType;
  } else if (retType) {
    resultType = retType;
  } else {
    resultType = mlir::LLVM::LLVMVoidType::get(ctx);
  }
  return mlir::LLVM::LLVMFunctionType::get(resultType, paramTypes, isVarArg);
}

static llvm::StringRef getExtAttrName(IntExtKind kind) {
  switch (kind) {
  case IntExtKind::None:
    return {};
  case IntExtKind::ZExt:
    return mlir::LLVM::LLVMDialect::getZExtAttrName();
  case IntExtKind::SExt:
    return mlir::LLVM::LLVMDialect::getSExtAttrName();
  }
  llvm_unreachable("unknown IntExtKind");
}

llvm::SmallVector<mlir::NamedAttrList> SignatureClassification::getParamAttrs(
    mlir::TypeRange argTypes, mlir::Type retType, size_t numArgs) const {
  numArgs = std::min(numArgs, args.size());
  llvm::SmallVector<mlir::NamedAttrList> attrs(getLLVMParamIndex(numArgs));
  if (ret.useSRet)
    attrs[0].set(mlir::LLVM::LLVMDialect::getStructRetAttrName(),
                 mlir::TypeAttr::get(retType));
  unsigned paramIdx = getLLVMParamIndex(0);
  for (auto [coercion, type] :
       llvm::zip(llvm::ArrayRef(args).take_front(numArgs), argTypes)) {
    if (coercion.useIndirect && coercion.useByval)
      attrs[paramIdx].set(mlir::LLVM::LLVMDialect::getByValAttrName(),
                          mlir::TypeAttr::get(type));
    if (llvm::StringRef ext = getExtAttrName(coercion.ext); !ext.empty())
      attrs[paramIdx].set(ext, mlir::UnitAttr::get(type.getContext()));
    paramIdx += coercion.getNumLLVMParams();
  }
  return attrs;
}

mlir::NamedAttrList
SignatureClassification::getResultAttrs(mlir::MLIRContext *ctx) const {
  mlir::NamedAttrList attrs;
  if (llvm::StringRef ext = getExtAttrName(ret.ext); !ext.empty())
    attrs.set(ext, mlir::UnitAttr::get(ctx));
  return attrs;
}

//===----------------------------------------------------------------------===//
// Factory Function
//===----------------------------------------------------------------------===//

std::unique_ptr<CABIInfo>
M::KGEN::createCABIInfo(const llvm::Triple &triple, mlir::MLIRContext *ctx,
                        const LLVMDataLayout &dataLayout) {

  // Check if C ABI functionality is enabled
  if (!CabiUtils::isCABIEnabled()) {
    // C ABI disabled - use default pass-through ABI
    return std::make_unique<DefaultCABIInfo>(ctx, dataLayout);
  }

  // Detect architecture and select appropriate ABI
  llvm::Triple::ArchType arch = triple.getArch();

  switch (arch) {
  case llvm::Triple::x86_64:
    // x86-64: Use System V AMD64 ABI (Linux, macOS, BSD)
    return std::make_unique<SystemVABIInfo>(ctx, dataLayout);

  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
  case llvm::Triple::aarch64_32:
    // ARM64: Use AAPCS (Procedure Call Standard for ARM 64-bit)
    // Used on: Linux ARM64, macOS Apple Silicon, iOS
    // isDarwin controls variadic HFA coercion (Darwin=GP-only va_list).
    return std::make_unique<AAPCSABIInfo>(ctx, dataLayout, triple.isOSDarwin());

  case llvm::Triple::x86:
    // 32-bit x86: Use default pass-through ABI
    // TODO: Could implement cdecl/stdcall/fastcall variants if needed
    return std::make_unique<DefaultCABIInfo>(ctx, dataLayout);

  default:
    // Unsupported architecture: use default pass-through ABI
    return std::make_unique<DefaultCABIInfo>(ctx, dataLayout);
  }
}

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

int64_t CabiUtils::getStructSize(mlir::LLVM::LLVMStructType type,
                                 const LLVMDataLayout &dataLayout) {
  // Use LLVMDataLayout which works with LLVM types.
  // This gives us the actual LLVM layout including padding from alignment.
  return dataLayout.getTypeStoreSize(type);
}

/// Recursively determine the canonical leaf float bit width for a struct.
///
/// Traverses all fields (recursing into nested LLVMStructType fields) and
/// returns the bit width if every leaf field is the same float type.
/// Returns 0 if any non-float field is found, the struct is empty, or
/// float types are mixed (e.g., f32 and f64 together).
static unsigned getLeafFloatBitWidth(mlir::LLVM::LLVMStructType type) {
  std::optional<unsigned> canonicalBitWidth;
  for (mlir::Type fieldType : type.getBody()) {
    unsigned fieldBitWidth = 0;
    if (auto floatType = dyn_cast<mlir::FloatType>(fieldType)) {
      fieldBitWidth = floatType.getWidth();
    } else if (auto vecType = dyn_cast<mlir::VectorType>(fieldType)) {
      if (auto floatElem = dyn_cast<mlir::FloatType>(vecType.getElementType()))
        fieldBitWidth = floatElem.getWidth();
      else
        return 0; // Non-float vector element
    } else if (auto nestedStruct =
                   dyn_cast<mlir::LLVM::LLVMStructType>(fieldType)) {
      fieldBitWidth = getLeafFloatBitWidth(nestedStruct);
      if (fieldBitWidth == 0)
        return 0; // Nested struct contains non-float or mixed floats
    } else {
      return 0; // Non-float field (integer, pointer, etc.)
    }
    if (!canonicalBitWidth)
      canonicalBitWidth = fieldBitWidth;
    else if (*canonicalBitWidth != fieldBitWidth)
      return 0; // Heterogeneous: mixed float types (e.g., f32 + f64)
  }
  return canonicalBitWidth.value_or(0);
}

bool CabiUtils::isAllFloatStruct(mlir::LLVM::LLVMStructType type) {
  auto fields = type.getBody();
  if (fields.empty())
    return false;

  // ARM64 AAPCS HFA (Homogeneous Floating-point Aggregate) limit: at most 4
  // top-level fields. Note: this checks top-level fields, not leaf float count.
  if (fields.size() > 4)
    return false;

  return getLeafFloatBitWidth(type) != 0;
}

mlir::IntegerType CabiUtils::getIntegerTypeForSize(int64_t size,
                                                   mlir::MLIRContext *ctx) {
  unsigned bitWidth;
  if (size == 1) {
    bitWidth = 8;
  } else if (size == 2) {
    bitWidth = 16;
  } else if (size <= 4) {
    bitWidth = 32;
  } else if (size <= 8) {
    bitWidth = 64;
  } else {
    llvm_unreachable("Invalid size for integer type");
  }
  return mlir::IntegerType::get(ctx, bitWidth);
}

CoercionInfo CabiUtils::classifySmallIntegerStruct(int64_t size,
                                                   mlir::MLIRContext *ctx) {
  CoercionInfo info;
  info.argClass = ABIArgClass::Integer;
  info.coercedType = getIntegerTypeForSize(size, ctx);
  return info;
}

CoercionInfo CabiUtils::classifyExtendedScalar(mlir::Type type) {
  CoercionInfo info;
  // The integers narrower than C `int`, which is 32 bits on every target that
  // uses this.
  if (auto intTy = dyn_cast<mlir::IntegerType>(type);
      intTy && intTy.getWidth() < 32)
    info.ext = IntExtKind::ZExt;
  return info;
}

bool CabiUtils::isCABIEnabled() {
  // C ABI coercion is enabled by default.
  // Use --skip-c-abi-coercion to disable (for debugging).
  return !CABIOpts->skipCoercion;
}
