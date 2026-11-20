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

#include "CABIAAPCS.h"
#include "LLVMLoweringUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//

AAPCSABIInfo::AAPCSABIInfo(mlir::MLIRContext *ctx,
                           const POPToLLVMTypeConverter &typeConverter,
                           bool isDarwinOS)
    : CABIInfo(ctx, typeConverter), isDarwin(isDarwinOS) {}

//===----------------------------------------------------------------------===//
// Homogeneous Aggregates
//===----------------------------------------------------------------------===//

std::optional<CoercionInfo> AAPCSABIInfo::classifyHomogeneousAggregate(
    mlir::Type type, mlir::LLVM::LLVMStructType structType) const {
  // The base type of a homogeneous aggregate: a float, or a short vector.
  auto isBaseType = [&](mlir::Type member) {
    if (isa<mlir::FloatType>(member))
      return true;
    int64_t size = dataLayout.getTypeAllocSize(member);
    return isa<mlir::VectorType>(member) && (size == 8 || size == 16);
  };
  // Vectors of one size are the same base type, whatever their elements.
  auto isSameBaseType = [&](mlir::Type a, mlir::Type b) {
    return a == b ||
           (isa<mlir::VectorType>(a) && isa<mlir::VectorType>(b) &&
            dataLayout.getTypeAllocSize(a) == dataLayout.getTypeAllocSize(b));
  };

  // A member without size holds no data, so it does not count. The members of
  // a union are at one offset, so count the offsets.
  mlir::Type baseType;
  bool isHomogeneous = true;
  llvm::SmallVector<int64_t, 4> offsets;
  CabiUtils::forEachMember(
      type, /*offset=*/0, typeConverter,
      [&](mlir::Type member, int64_t offset) {
        if (dataLayout.getTypeAllocSize(member) == 0)
          return;
        if (!isBaseType(member) ||
            (baseType && !isSameBaseType(baseType, member))) {
          isHomogeneous = false;
          return;
        }
        baseType = baseType ? baseType : member;
        if (!llvm::is_contained(offsets, offset))
          offsets.push_back(offset);
      });
  if (!isHomogeneous || !baseType || offsets.size() > 4)
    return std::nullopt;

  // There must be no padding.
  int64_t numMembers = offsets.size();
  if (numMembers * dataLayout.getTypeAllocSize(baseType) !=
      dataLayout.getTypeAllocSize(structType))
    return std::nullopt;

  // Coerce to an array of the base type, as clang does. The backend keeps
  // only an array argument in consecutive registers: if the free SIMD
  // registers are too few, all of the array goes on the stack (AAPCS64 C.3).
  // The members of a struct argument are separate, so the first members can
  // go in registers and the remaining members on the stack.
  CoercionInfo info;
  info.argClass = ABIArgClass::SSE;
  info.coercedType = mlir::LLVM::LLVMArrayType::get(baseType, numMembers);
  return info;
}

//===----------------------------------------------------------------------===//
// Argument Classification
//===----------------------------------------------------------------------===//

CoercionInfo AAPCSABIInfo::classifyArgumentType(mlir::Type type,
                                                mlir::Location loc,
                                                bool isVariadicArg) const {
  // Arrays should be passed indirectly, as per the C specification
  mlir::Type llvmType = convertType(type);
  if (isa<mlir::LLVM::LLVMArrayType>(llvmType))
    return CoercionInfo::indirectArgument(/*useByval=*/false);

  // Only types that convert to an LLVM struct need C ABI classification
  auto structType = dyn_cast<mlir::LLVM::LLVMStructType>(llvmType);
  if (!structType) {
    // Non-struct (scalar, pointer, etc.): pass through directly, no coercion.
    // Apple's arm64 ABI, unlike AAPCS64, requires the caller to extend
    // narrow integers.
    return isDarwin ? CabiUtils::classifyExtendedScalar(type, typeConverter)
                    : CoercionInfo{};
  }

  // A homogeneous aggregate goes in SIMD registers (V0-V3). For a variadic
  // argument, this depends on the OS ABI:
  //   Darwin (macOS/iOS): va_list has only a GP area, so the aggregate is
  //     classified as a struct of integers and goes in GP registers, where
  //     va_arg reads it.
  //   Linux AAPCS64: va_list has separate GP and VR save areas (AAPCS64 spec
  //     IHI0055 B.4), so the aggregate goes in SIMD registers, where va_arg
  //     reads it.
  //
  // Reference implementation: clang/lib/CodeGen/Targets/AArch64.cpp,
  //   EmitDarwinVAArg() vs EmitAAPCSVAArg().
  if (std::optional<CoercionInfo> info =
          classifyHomogeneousAggregate(type, structType);
      info && (!isVariadicArg || !isDarwin)) {
    return *info;
  }

  // Get struct size in bytes
  int64_t size = CabiUtils::getStructSize(structType, dataLayout);

  // RULE 1: Structs >16 bytes are passed by pointer
  if (size > 16) {
    return CoercionInfo::indirectArgument(/*useByval=*/false);
  }

  // ARM64 AAPCS: A struct that is not a homogeneous aggregate goes in GPRs
  // (integer registers), whatever the types of its fields.

  // RULE 2: 1-8 byte non-HFA structs → coerce to iN
  if (size <= 8) {
    return CabiUtils::classifySmallIntegerStruct(size, ctx);
  }

  // RULE 3: 9-16 byte non-HFA structs → two-register passing
  return classifyTwoRegisterStruct(size, loc);
}

//===----------------------------------------------------------------------===//
// Return Value Classification
//===----------------------------------------------------------------------===//

CoercionInfo AAPCSABIInfo::classifyReturnType(mlir::Type type,
                                              mlir::Location loc) const {
  // Only types that convert to an LLVM struct need C ABI classification
  auto structType = dyn_cast<mlir::LLVM::LLVMStructType>(convertType(type));
  if (!structType) {
    // Non-struct (scalar, pointer, etc.): return directly, no coercion.
    // On Darwin the callee extends a narrow integer result, as clang does.
    return isDarwin ? CabiUtils::classifyExtendedScalar(type, typeConverter)
                    : CoercionInfo{};
  }

  // A homogeneous aggregate is returned in SIMD registers.
  if (std::optional<CoercionInfo> info =
          classifyHomogeneousAggregate(type, structType))
    return *info;

  // Get struct size in bytes
  int64_t size = CabiUtils::getStructSize(structType, dataLayout);

  // RULE: Structs >16 bytes use indirect return (X8 pointer)
  if (size > 16) {
    return CoercionInfo::sretReturn();
  }

  // ARM64 AAPCS: Non-HFA structs always use GPRs for return values.

  // RULE: 1-8 byte non-HFA structs → return in X0 as iN
  if (size <= 8) {
    return CabiUtils::classifySmallIntegerStruct(size, ctx);
  }

  // RULE: 9-16 byte non-HFA structs → two-register return
  return classifyTwoRegisterStruct(size, loc);
}

//===----------------------------------------------------------------------===//
// Phase 2: Two-Register Classification (9-16 byte structs)
//===----------------------------------------------------------------------===//

CoercionInfo AAPCSABIInfo::classifyTwoRegisterStruct(int64_t size,
                                                     mlir::Location loc) const {

  // ARM64 AAPCS: Split struct into two 8-byte chunks
  // First chunk: bytes 0-7 (always 8 bytes)
  // Second chunk: bytes 8-15 (may be less than 8 bytes)

  CoercionInfo info;
  info.argClass = ABIArgClass::IntegerPair; // Both passed as integers on ARM64

  // First register: always i64 (8 bytes)
  info.coercedType = CabiUtils::getIntegerTypeForSize(8, ctx);

  // Second register: size depends on remaining bytes
  int64_t secondSize = size - 8; // Bytes 8 through (size-1)
  info.coercedSecondType = CabiUtils::getIntegerTypeForSize(secondSize, ctx);

  return info;
}

//===----------------------------------------------------------------------===//
// Whole-signature classification (rollback-to-stack)
//===----------------------------------------------------------------------===//

unsigned AAPCSABIInfo::argGPRUsage(const CoercionInfo &info,
                                   mlir::Type llvmType) const {
  switch (info.argClass) {
  case ABIArgClass::Integer: {
    int64_t size = dataLayout.getTypeStoreSize(info.coercedType);
    return std::max<unsigned>((size + 7) / 8, 1);
  }
  case ABIArgClass::IntegerPair:
    return 2;
  case ABIArgClass::Memory:
    // A byval argument is passed in the stack argument area; the raw
    // pointer of a >16-byte struct is an ordinary pointer argument in a GPR.
    return info.useByval ? 0 : 1;
  case ABIArgClass::SSE:
    return 0; // Homogeneous aggregate: V registers or the stack
  case ABIArgClass::NoClass:
    break; // Identity arg: derive usage from the LLVM type below.
  default:
    break; // SSEPair/Mixed are not produced on ARM64.
  }

  if (isa<mlir::LLVM::LLVMPointerType>(llvmType))
    return 1;
  if (auto intTy = dyn_cast<mlir::IntegerType>(llvmType)) {
    unsigned regs = (intTy.getWidth() + 63) / 64;
    return std::max(regs, 1u);
  }
  if (isa<mlir::FloatType>(llvmType))
    return 0; // V register
  if (isa<mlir::VectorType>(llvmType))
    return 0; // V registers
  if (isa<mlir::LLVM::LLVMStructType>(llvmType))
    return 0; // Identity struct (HFA): V registers or the stack
  assert(false && "unhandled identity argument type in register accounting");
  return 0;
}

SignatureClassification
AAPCSABIInfo::classifySignature(mlir::TypeRange argTypes, mlir::Type retType,
                                mlir::Location loc, size_t numFixedArgs) const {
  SignatureClassification result;
  if (retType)
    result.ret = classifyReturnType(retType, loc);

  // AAPCS64 stage C searches X0-X7 for composite arguments. An sret return
  // uses X8, which is outside the argument register set, so it does not
  // reduce the budget (unlike System V, where sret takes RDI).
  unsigned freeGPRs = 8;

  // The alignment that selects the form of a composite. Apple's arm64 ABI uses
  // the alignment of the type. AAPCS64 uses the natural alignment, which does
  // not include the alignment that `@align` gives the struct itself.
  auto compositeAlign = [&](mlir::Type type) {
    return isDarwin ? getTypeABIAlign(type) : getNaturalTypeABIAlign(type);
  };

  // Coerce a two-register composite to a single by-value integer, so that the
  // backend assigns all of it to registers or all of it to the stack. A
  // 16-byte-aligned composite is an i128, and other composites are a
  // [2 x i64]. This is the form that clang emits.
  auto singleIntegerArgument = [&](mlir::Type type) {
    CoercionInfo info;
    info.argClass = ABIArgClass::Integer;
    if (compositeAlign(type) >= 16) {
      info.coercedType = mlir::IntegerType::get(ctx, 128);
    } else {
      info.coercedType =
          mlir::LLVM::LLVMArrayType::get(mlir::IntegerType::get(ctx, 64), 2);
    }
    return info;
  };

  result.args.reserve(argTypes.size());
  for (auto [idx, type] : llvm::enumerate(argTypes)) {
    bool isVariadicArg = idx >= numFixedArgs;
    CoercionInfo info = classifyArgumentType(type, loc, isVariadicArg);
    mlir::Type llvmType = convertType(type);

    // A 16-byte-aligned composite is an i128 at each position. On Linux, it
    // then starts at an even register.
    if (info.isTwoRegister() && compositeAlign(type) >= 16)
      info = singleIntegerArgument(type);

    // Apple's arm64 ABI passes all variadic arguments on the stack, so they
    // use no argument registers. A composite has the same form at each
    // position.
    if (isDarwin && isVariadicArg) {
      if (info.isTwoRegister())
        info = singleIntegerArgument(type);
      result.args.push_back(info);
      continue;
    }

    // Linux AAPCS64 rounds NGRN up to an even register for i128. Darwin
    // permits an odd starting register, so it does not consume this slot.
    mlir::Type passedType = info.coercedType ? info.coercedType : llvmType;
    if (!isDarwin && passedType.isInteger(128) && freeGPRs % 2)
      --freeGPRs;

    // Unlike x86-64, variadic arguments on Linux AAPCS64 are allocated like
    // fixed arguments (the callee prologue spills X0-X7 into the GP save
    // area), so they take part in the register accounting.
    unsigned need = argGPRUsage(info, llvmType);
    if (need <= freeGPRs) {
      freeGPRs -= need;
    } else {
      // An argument that does not fit in the remaining registers is passed
      // entirely in memory; every later argument goes to the stack too, even
      // though a register remains. The backend does this for a single
      // by-value integer, but not for a two-register composite.
      if (info.isTwoRegister())
        info = singleIntegerArgument(type);
      freeGPRs = 0;
    }
    result.args.push_back(info);
  }
  return result;
}
