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
  classifyMember(structType, /*offset=*/0, lo, hi);

  // Memory anywhere: the whole struct goes in memory.
  if (lo == EightbyteClass::Memory || hi == EightbyteClass::Memory)
    return useSRet ? CoercionInfo::sretReturn()
                   : CoercionInfo::indirectArgument(/*useByval=*/true);

  CoercionInfo info;

  // A 128-bit vector: one SSE register. clang also uses <2 x double>; the
  // register is the same whatever the vector's element type.
  if (lo == EightbyteClass::SSE && hi == EightbyteClass::SSEUp) {
    info.argClass = ABIArgClass::SSE;
    info.coercedType = mlir::VectorType::get({2}, mlir::Float64Type::get(ctx));
    return info;
  }

  // From here each eightbyte is one register, Integer or SSE. Bytes 0-7 always
  // hold data: the first field with data starts at offset 0, and an empty
  // struct returned above. Bytes 8-15 have no class only when they are tail
  // padding, which needs extra alignment. clang then uses one register; this
  // uses an integer register for them, as before.
  if (hi == EightbyteClass::NoClass)
    hi = EightbyteClass::Integer;

  // One eightbyte: one register. Two floats get f64 where clang uses
  // <2 x float>; the store/load in prepareCoercedArgument puts both in the
  // same SSE register.
  if (size <= 8) {
    if (lo == EightbyteClass::Integer)
      return CabiUtils::classifySmallIntegerStruct(size, ctx);
    info.argClass = ABIArgClass::SSE;
    info.coercedType = getEightbyteType(lo, size);
    return info;
  }

  // Two eightbytes: one register each.
  if (lo != hi)
    info.argClass = ABIArgClass::Mixed;
  else if (lo == EightbyteClass::Integer)
    info.argClass = ABIArgClass::IntegerPair;
  else
    info.argClass = ABIArgClass::SSEPair;
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

SystemVABIInfo::EightbyteClass
SystemVABIInfo::classifyLeaf(mlir::Type type) const {
  int64_t bits = dataLayout.getTypeSizeInBits(type);
  if (auto vectorType = dyn_cast<mlir::VectorType>(type)) {
    if (bits <= 32)
      return EightbyteClass::Integer;
    // gcc passes <1 x double> in memory, and clang follows it.
    if (bits == 64 && !vectorType.getElementType().isF64())
      return EightbyteClass::SSE;
    if (bits == 128)
      return EightbyteClass::SSE;
    return EightbyteClass::Memory;
  }
  // Mojo has no float wider than 64 bits; Memory keeps a wider one safe.
  if (isa<mlir::FloatType>(type))
    return bits <= 64 ? EightbyteClass::SSE : EightbyteClass::Memory;
  // This includes the explicit [N x i8] padding Mojo inserts before an
  // aligned field: after KGEN→LLVM conversion it cannot be told apart from an
  // i8 array holding data.
  return EightbyteClass::Integer;
}

void SystemVABIInfo::classifyMember(mlir::Type type, int64_t offset,
                                    EightbyteClass &lo,
                                    EightbyteClass &hi) const {
  if (auto arrayType = dyn_cast<mlir::LLVM::LLVMArrayType>(type)) {
    mlir::Type elementType = arrayType.getElementType();
    int64_t elementSize = dataLayout.getTypeAllocSize(elementType);
    if (elementSize == 0)
      return; // An array of empty structs.
    for (int64_t i = 0, e = arrayType.getNumElements(); i < e; ++i)
      classifyMember(elementType, offset + i * elementSize, lo, hi);
    return;
  }
  if (auto structType = dyn_cast<mlir::LLVM::LLVMStructType>(type)) {
    // Lay out the fields with the same steps getTypeSizeInBits uses for a
    // struct: align each field, then add its alloc size. Every field is then
    // aligned, so clang's unaligned-field rule never applies. A packed struct
    // would break that; Mojo does not emit one.
    // TODO(MOCO-3369): Add LLVMDataLayout::getStructFieldOffsets() to replace
    // this manual accumulation with a single pre-computed offset array,
    // mirroring llvm::DataLayout::getStructLayout()->getElementOffset().
    assert(!structType.isPacked() &&
           "packed structs are not classified; their fields may be unaligned");
    int64_t fieldOffset = 0;
    for (mlir::Type fieldType : structType.getBody()) {
      fieldOffset =
          llvm::alignTo(fieldOffset, dataLayout.getTypeABIAlign(fieldType));
      classifyMember(fieldType, offset + fieldOffset, lo, hi);
      fieldOffset += dataLayout.getTypeAllocSize(fieldType);
    }
    return;
  }

  EightbyteClass leaf = classifyLeaf(type);
  int64_t size = dataLayout.getTypeStoreSize(type);
  int64_t end = offset + size;
  assert(end <= 16 && "member outside a 16-byte struct");

  // The leaf lies in bytes 0-7, in bytes 8-15, or across both.
  if (end <= 8) {
    lo = mergeEightbyteClass(lo, leaf);
    return;
  }
  if (offset >= 8) {
    hi = mergeEightbyteClass(hi, leaf);
    return;
  }
  // Across both, only a 16-byte leaf at offset 0 is aligned: an i128, or a
  // 128-bit vector, which fills one SSE register. A smaller leaf there is
  // misaligned, which happens because LLVMDataLayout aligns a vector only to
  // its element; as in clang, an unaligned member sends the struct to memory.
  if (size != 16) {
    lo = mergeEightbyteClass(lo, EightbyteClass::Memory);
    return;
  }
  lo = mergeEightbyteClass(lo, leaf);
  hi = mergeEightbyteClass(
      hi, isa<mlir::VectorType>(type) ? EightbyteClass::SSEUp : leaf);
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
