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

#include "CABISystemV.h"
#include "LLVMLoweringUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/STLExtras.h"

using namespace M;
using namespace M::KGEN;

//===----------------------------------------------------------------------===//
// Constructor
//===----------------------------------------------------------------------===//

SystemVABIInfo::SystemVABIInfo(mlir::MLIRContext *ctx,
                               const LLVMDataLayout &dataLayout)
    : CABIInfo(ctx, dataLayout) {}

//===----------------------------------------------------------------------===//
// Argument Classification
//===----------------------------------------------------------------------===//

CoercionInfo SystemVABIInfo::classifyArgumentType(mlir::Type type,
                                                  mlir::Location loc,
                                                  bool isVariadicArg) const {
  // On x86-64 System V, variadic struct args follow the same classification
  // rules as non-variadic args. LLVM handles placing them on the stack.
  // (isVariadicArg is used by ARM64 AAPCS but not needed here.)
  (void)isVariadicArg;

  // Arrays should be passed indirectly, as per the C specification
  if (isa<mlir::LLVM::LLVMArrayType>(type))
    return CoercionInfo::indirectArgument(/*useByval=*/false);

  return classifyStructType(type, loc, /*useSRet=*/false);
}

//===----------------------------------------------------------------------===//
// Return Value Classification
//===----------------------------------------------------------------------===//

CoercionInfo SystemVABIInfo::classifyReturnType(mlir::Type type,
                                                mlir::Location loc) const {
  return classifyStructType(type, loc, /*useSRet=*/true);
}

//===----------------------------------------------------------------------===//
// Common Struct Classification
//===----------------------------------------------------------------------===//

CoercionInfo SystemVABIInfo::classifyStructType(mlir::Type type,
                                                mlir::Location loc,
                                                bool useSRet) const {
  // Only LLVM struct types need C ABI classification
  auto structType = dyn_cast<mlir::LLVM::LLVMStructType>(type);
  if (!structType) {
    // The psABI leaves the upper bits of a narrow integer unspecified, but
    // clang extends at the producer and relies on it at the consumer, so
    // match clang.
    return CabiUtils::classifyExtendedScalar(type);
  }

  int64_t size = CabiUtils::getStructSize(structType, dataLayout);

  // Structs >16 bytes use MEMORY class (pointer for args, sret for returns)
  if (size > 16) {
    return useSRet ? CoercionInfo::sretReturn()
                   : CoercionInfo::indirectArgument(/*useByval=*/true);
  }

  // clang passes nothing for a 0-byte struct, but CoercionInfo cannot express
  // "nothing", so it keeps its previous integer coercion.
  if (size == 0)
    return CabiUtils::classifySmallIntegerStruct(size, ctx);

  // Classify each member at its offset and merge the classes per eightbyte.
  EightbyteClass lo = EightbyteClass::NoClass; // Bytes 0-7
  EightbyteClass hi = EightbyteClass::NoClass; // Bytes 8-15
  CabiUtils::forEachMember(structType, /*offset=*/0, dataLayout,
                           [&](mlir::Type member, int64_t offset) {
                             auto [memberLo, memberHi] =
                                 classifyMember(member, offset);
                             lo = mergeEightbyteClass(lo, memberLo);
                             hi = mergeEightbyteClass(hi, memberHi);
                           });

  // Memory anywhere: the whole struct goes in memory.
  if (lo == EightbyteClass::Memory || hi == EightbyteClass::Memory)
    return useSRet ? CoercionInfo::sretReturn()
                   : CoercionInfo::indirectArgument(/*useByval=*/true);

  // A 128-bit vector: one SSE register whatever the vector's element type.
  if (lo == EightbyteClass::SSE && hi == EightbyteClass::SSEUp) {
    CoercionInfo info;
    info.argClass = ABIArgClass::SSE;
    info.coercedType = mlir::VectorType::get({2}, mlir::Float64Type::get(ctx));
    return info;
  }

  // One eightbyte: one register. For two floats this is f64 where clang uses
  // <2 x float>: the bytes are copied through memory, not converted, and both
  // types use the same SSE register.
  if (size <= 8) {
    CoercionInfo info;
    info.argClass =
        lo == EightbyteClass::SSE ? ABIArgClass::SSE : ABIArgClass::Integer;
    info.coercedType = getEightbyteType(lo, size);
    return info;
  }

  // Two eightbytes: two registers.
  CoercionInfo info;
  if (lo == EightbyteClass::SSE && hi == EightbyteClass::SSE)
    info.argClass = ABIArgClass::SSEPair;
  else if (lo == EightbyteClass::Integer && hi == EightbyteClass::Integer)
    info.argClass = ABIArgClass::IntegerPair;
  else
    info.argClass = ABIArgClass::Mixed;
  info.coercedType = getEightbyteType(lo, 8);
  info.coercedSecondType = getEightbyteType(hi, size - 8);
  return info;
}

//===----------------------------------------------------------------------===//
// Eightbyte Classification (structs up to 16 bytes)
//===----------------------------------------------------------------------===//

mlir::Type SystemVABIInfo::getEightbyteType(EightbyteClass eightbyteClass,
                                            int64_t size) const {
  if (eightbyteClass == EightbyteClass::SSE) {
    // SSE class: use float types
    if (size <= 4) {
      return mlir::Float32Type::get(ctx);
    }
    assert(size <= 8 && "SSE eightbyte must not exceed 8 bytes");
    return mlir::Float64Type::get(ctx);
  }

  // Integer class (or NoClass): use integer types
  return CabiUtils::getIntegerTypeForSize(size, ctx);
}

SystemVABIInfo::EightbyteClass
SystemVABIInfo::mergeEightbyteClass(EightbyteClass accum,
                                    EightbyteClass field) {
  if (accum == field || field == EightbyteClass::NoClass)
    return accum;
  if (accum == EightbyteClass::NoClass)
    return field;
  if (accum == EightbyteClass::Memory || field == EightbyteClass::Memory)
    return EightbyteClass::Memory;
  if (accum == EightbyteClass::Integer || field == EightbyteClass::Integer)
    return EightbyteClass::Integer;
  // Only SSE and SSEUp remain. A new class needs its own rule above.
  assert(
      llvm::is_contained({EightbyteClass::SSE, EightbyteClass::SSEUp}, accum) &&
      llvm::is_contained({EightbyteClass::SSE, EightbyteClass::SSEUp}, field) &&
      "unhandled eightbyte class");
  return EightbyteClass::SSE;
}

// This follows clang's X86_64ABIInfo::classify.
std::pair<SystemVABIInfo::EightbyteClass, SystemVABIInfo::EightbyteClass>
SystemVABIInfo::classifyMember(mlir::Type type, int64_t offset) const {
  int64_t size = dataLayout.getTypeAllocSize(type);
  assert(offset + size <= 16 && "member outside a 16-byte struct");
  // A member without size holds no data, so it is in no eightbyte.
  if (size == 0)
    return {EightbyteClass::NoClass, EightbyteClass::NoClass};

  // A 16-byte member fills both eightbytes.
  if (size == 16) {
    assert(!isa<mlir::FloatType>(type) &&
           "floats wider than 64 bits are not supported");
    // A 128-bit vector fills one SSE register.
    if (isa<mlir::VectorType>(type))
      return {EightbyteClass::SSE, EightbyteClass::SSEUp};
    // An i128 takes two integer registers.
    return {EightbyteClass::Integer, EightbyteClass::Integer};
  }

  // Member sizes are 2^n, so a smaller member that lies across both
  // eightbytes is misaligned: it is a vector aligned only to its element. As
  // in clang, it sends the struct to memory.
  if (offset < 8 && offset + size > 8)
    return {EightbyteClass::Memory, EightbyteClass::NoClass};

  // Any other member lies in one eightbyte.
  EightbyteClass memberClass;
  if (isa<mlir::VectorType>(type)) {
    if (size <= 4)
      memberClass = EightbyteClass::Integer;
    else if (size == 8)
      // Can't match gcc/clang behavior: C passes <1 x double> in memory.
      // In Mojo Float64 := SIMD[DType.float64, 1], so a one-element vector is
      // the scalar itself and cannot be told apart from <1 x double>.
      memberClass = EightbyteClass::SSE;
    else
      memberClass = EightbyteClass::Memory;
  } else if (isa<mlir::FloatType>(type)) {
    memberClass = EightbyteClass::SSE;
  } else {
    // Integers and pointers. This includes the explicit [N x i8] padding Mojo
    // inserts before an aligned field: after KGEN→LLVM conversion it cannot
    // be told apart from an i8 array holding data.
    memberClass = EightbyteClass::Integer;
  }

  if (offset < 8)
    return {memberClass, EightbyteClass::NoClass};
  return {EightbyteClass::NoClass, memberClass};
}

//===----------------------------------------------------------------------===//
// Whole-signature classification (rollback-to-stack)
//===----------------------------------------------------------------------===//

std::pair<unsigned, unsigned>
SystemVABIInfo::argRegisterUsage(const CoercionInfo &info,
                                 mlir::Type llvmType) const {
  switch (info.argClass) {
  case ABIArgClass::Integer:
    return {1, 0};
  case ABIArgClass::SSE:
    return {0, 1};
  case ABIArgClass::IntegerPair:
    return {2, 0};
  case ABIArgClass::SSEPair:
    return {0, 2};
  case ABIArgClass::Mixed:
    return {1, 1};
  case ABIArgClass::Memory:
    // Passed in memory; consumes no argument registers.
    return {0, 0};
  case ABIArgClass::NoClass:
    break; // Identity arg: derive usage from the LLVM type below.
  }

  if (isa<mlir::LLVM::LLVMPointerType>(llvmType))
    return {1, 0};
  if (auto intTy = dyn_cast<mlir::IntegerType>(llvmType)) {
    unsigned regs = (intTy.getWidth() + 63) / 64;
    return {std::max(regs, 1u), 0};
  }
  if (auto floatTy = dyn_cast<mlir::FloatType>(llvmType)) {
    // f32/f64 use one SSE register; wider x87/quad types are not register args.
    if (floatTy.getWidth() <= 64)
      return {0, 1};
    return {0, 0};
  }
  if (auto vecTy = dyn_cast<mlir::VectorType>(llvmType)) {
    unsigned bytes = dataLayout.getTypeStoreSize(vecTy);
    unsigned regs = (bytes + 15) / 16;
    return {0, std::max(regs, 1u)};
  }

  // pointer / integer / float / vector are handled above; an identity arg of
  // any other type means the register accounting is incomplete. Guessing a
  // count here would silently mis-budget and break rollback decisions.
  assert(false && "unhandled identity argument type in register accounting");
  return {0, 0};
}

SignatureClassification
SystemVABIInfo::classifySignature(mlir::TypeRange argTypes, mlir::Type retType,
                                  mlir::Location loc,
                                  size_t numFixedArgs) const {
  SignatureClassification result;
  if (retType)
    result.ret = classifyReturnType(retType, loc);

  // SysV provides 6 GP and 8 SSE argument registers. An sret return reserves
  // one GP register for the hidden result pointer.
  unsigned freeInt = 6;
  unsigned freeSSE = 8;
  if (result.ret.useSRet)
    freeInt -= 1;

  result.args.reserve(argTypes.size());
  for (auto [idx, type] : llvm::enumerate(argTypes)) {
    bool isVariadicArg = idx >= numFixedArgs;
    CoercionInfo info = classifyArgumentType(type, loc, isVariadicArg);

    // Only fixed args take part in register accounting; variadic args keep
    // their classification.
    if (!isVariadicArg) {
      auto [needInt, needSSE] = argRegisterUsage(info, type);
      if (needInt <= freeInt && needSSE <= freeSSE) {
        freeInt -= needInt;
        freeSSE -= needSSE;
      } else if (!info.isIdentity() && !info.useIndirect) {
        // An in-register aggregate that does not fit rolls back to memory
        // rather than being split across registers and stack.
        info = CoercionInfo::indirectArgument(/*useByval=*/true);
      }
      // A scalar that does not fit stays direct; the backend places it on the
      // stack.
    }
    result.args.push_back(info);
  }
  return result;
}
