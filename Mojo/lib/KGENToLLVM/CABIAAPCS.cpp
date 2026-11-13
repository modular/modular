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
#include "mlir/IR/Diagnostics.h"

using namespace M;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//

AAPCSABIInfo::AAPCSABIInfo(mlir::MLIRContext *ctx,
                           const LLVMDataLayout &dataLayout, bool isDarwinOS)
    : CABIInfo(ctx, dataLayout), isDarwin(isDarwinOS) {}

//===----------------------------------------------------------------------===//
// Argument Classification
//===----------------------------------------------------------------------===//

CoercionInfo AAPCSABIInfo::classifyArgumentType(mlir::Type type,
                                                mlir::Location loc,
                                                bool isVariadicArg) const {
  // Arrays should be passed indirectly, as per the C specification
  if (isa<mlir::LLVM::LLVMArrayType>(type))
    return CoercionInfo::indirectArgument(/*useByval=*/false);

  // Only LLVM struct types need C ABI classification
  auto structType = dyn_cast<mlir::LLVM::LLVMStructType>(type);
  if (!structType) {
    // Non-struct (scalar, pointer, etc.): pass through directly, no coercion.
    // Apple's arm64 ABI, unlike AAPCS64, requires the caller to extend
    // narrow integers.
    return isDarwin ? CabiUtils::classifyExtendedScalar(type) : CoercionInfo{};
  }

  // HFA (Homogeneous Float Aggregate) classification:
  //
  // Fixed (non-variadic) args: coerced to an array, {f32, f32} → [2 x f32],
  // which LLVM passes in SIMD registers (V0–V3) or entirely on the stack.
  //
  // Variadic args depend on the OS ABI:
  //   Darwin (macOS/iOS): va_list is flat (GP area only). HFA variadic args
  //     must be coerced to integer so they land in GP registers where
  //     va_arg reads them.
  //   Linux AAPCS64: va_list has separate GP and VR save areas (AAPCS64 spec
  //     IHI0055 §B.4). HFA variadic args are passed like non-variadic ones, so
  //     they land in VR registers where va_arg for HFA structs reads them.
  //
  // Reference implementation: clang/lib/CodeGen/Targets/AArch64.cpp,
  //   EmitDarwinVAArg() vs EmitAAPCSVAArg().
  std::optional<HomogeneousAggregate> aggregate =
      getHomogeneousAggregate(structType);
  if (aggregate && (!isVariadicArg || !isDarwin))
    return classifyHomogeneousAggregate(*aggregate);

  // Get struct size in bytes
  int64_t size = CabiUtils::getStructSize(structType, dataLayout);

  // RULE 1: Structs >16 bytes are passed by pointer
  if (size > 16) {
    return CoercionInfo::indirectArgument(/*useByval=*/false);
  }

  // ARM64 AAPCS: Non-HFA structs are always passed in GPRs (integer
  // registers), regardless of whether fields are int, float, or mixed.
  // HFA structs (all-float, same type, ≤4 fields) use SIMD registers,
  // but those are handled above by the HFA early return.

  // RULE 2: 1-8 byte non-HFA structs → coerce to i64
  //
  // In a register any integer type gives the same result. The type matters
  // when x0-x7 are taken and the struct goes on the stack, where C gives it
  // a whole 8-byte slot:
  //   Darwin (macOS/iOS): a stack slot has the size of the argument's type,
  //     so the type must be i64.
  //   Linux AAPCS64: every stack slot is 8 bytes, so i64 and an integer of
  //     the struct's size (iN) are passed the same way.
  //
  // clang 23 uses i64 on both. clang's main branch uses iN on Linux
  // (llvm/llvm-project#215264).
  // TODO: Use iN on Linux once that clang is released.
  if (size <= 8) {
    return CabiUtils::classifySmallIntegerStruct(8, ctx);
  }

  // RULE 3: 9-16 byte non-HFA structs → two-register passing
  return classifyTwoRegisterStruct(structType);
}

//===----------------------------------------------------------------------===//
// Return Value Classification
//===----------------------------------------------------------------------===//

CoercionInfo AAPCSABIInfo::classifyReturnType(mlir::Type type,
                                              mlir::Location loc) const {
  // Only LLVM struct types need C ABI classification
  auto structType = dyn_cast<mlir::LLVM::LLVMStructType>(type);
  if (!structType) {
    // Non-struct (scalar, pointer, etc.): return directly, no coercion.
    // On Darwin the callee extends a narrow integer result, as clang does.
    return isDarwin ? CabiUtils::classifyExtendedScalar(type) : CoercionInfo{};
  }

  // An HFA or HVA keeps its type. LLVM returns the members of such a struct
  // in SIMD registers (V0–V3), which is what C expects.
  if (getHomogeneousAggregate(structType)) {
    return CoercionInfo{}; // Identity: defer to standard lowering
  }

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
  return classifyTwoRegisterStruct(structType);
}

//===----------------------------------------------------------------------===//
// Phase 2: Two-Register Classification (9-16 byte structs)
//===----------------------------------------------------------------------===//

CoercionInfo
AAPCSABIInfo::classifyTwoRegisterStruct(mlir::LLVM::LLVMStructType type) const {
  // The struct is passed as one value, not as two integers (i64, iN).
  // This prevents splitting it between a register and the stack.
  CoercionInfo info;
  info.argClass = ABIArgClass::Integer;
  // A 16-byte aligned struct is passed as i128, to keep its alignment.
  if (dataLayout.getTypeABIAlign(type) >= 16) {
    info.coercedType = mlir::IntegerType::get(ctx, 128);
  } else {
    info.coercedType =
        mlir::LLVM::LLVMArrayType::get(mlir::IntegerType::get(ctx, 64), 2);
  }
  return info;
}

//===----------------------------------------------------------------------===//
// HFA and HVA Detection
//===----------------------------------------------------------------------===//

CoercionInfo AAPCSABIInfo::classifyHomogeneousAggregate(
    const HomogeneousAggregate &aggregate) {
  CoercionInfo info;
  info.argClass = ABIArgClass::SSE;
  info.coercedType =
      mlir::LLVM::LLVMArrayType::get(aggregate.base, aggregate.numMembers);
  return info;
}

bool AAPCSABIInfo::isZeroLengthArray(mlir::Type type) {
  // Look through the struct that Mojo's `Array` puts around its array.
  auto structType = dyn_cast<mlir::LLVM::LLVMStructType>(type);
  if (structType && structType.getBody().size() == 1)
    type = structType.getBody().front();

  auto arrayType = dyn_cast<mlir::LLVM::LLVMArrayType>(type);
  if (!arrayType)
    return false;
  // The array itself has no elements, or its elements are such arrays.
  return arrayType.getNumElements() == 0 ||
         isZeroLengthArray(arrayType.getElementType());
}

// This follows clang's ABIInfo::isHomogeneousAggregate.
std::optional<AAPCSABIInfo::HomogeneousAggregate>
AAPCSABIInfo::getHomogeneousAggregate(mlir::LLVM::LLVMStructType type) const {
  // The largest HFA or HVA is four 128-bit vectors: 64 bytes.
  if (dataLayout.getTypeAllocSize(type) > 64)
    return std::nullopt;

  mlir::Type base;
  int64_t numMembers = 0;
  bool isHFA = true;
  bool isHVA = true;
  bool hasZeroLengthArray = false;
  auto countMember = [&](mlir::Type member, int64_t /*offset*/) {
    // With a fifth member the struct is already not an HFA or HVA.
    if (numMembers > 4)
      return;
    // A member without size (an empty struct) is ignored. The exception is
    // a zero-length array: with it the struct is not an HFA.
    int64_t size = dataLayout.getTypeAllocSize(member);
    if (size == 0) {
      hasZeroLengthArray |= isZeroLengthArray(member);
      return;
    }
    if (!base)
      base = member;
    ++numMembers;
    bool sameSize = size == dataLayout.getTypeAllocSize(base);
    // HFA: floats of one size. Even bfloat16 with float16 is okay.
    isHFA &= sameSize && isa<mlir::FloatType>(member);
    // HVA: vectors of one size, 64 or 128 bits, with any element type.
    isHVA &=
        sameSize && isa<mlir::VectorType>(member) && (size == 8 || size == 16);
  };
  CabiUtils::forEachMember(type, /*offset=*/0, dataLayout, countMember);
  if (hasZeroLengthArray)
    return std::nullopt;
  // AAPCS64 allows 1 to 4 members.
  if (numMembers == 0 || numMembers > 4)
    return std::nullopt;
  if (!isHFA && !isHVA)
    return std::nullopt;
  // The members must cover the whole struct, with no padding.
  if (dataLayout.getTypeAllocSize(base) * numMembers !=
      dataLayout.getTypeAllocSize(type))
    return std::nullopt;

  return HomogeneousAggregate{base, numMembers};
}
