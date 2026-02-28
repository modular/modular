//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "CABILowering.h"
#include "CABIAAPCS.h"
#include "CABISystemV.h"
#include "LLVMLoweringUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
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

bool CabiUtils::isAllFloatStruct(mlir::LLVM::LLVMStructType type) {
  auto fields = type.getBody();
  if (fields.empty()) {
    return false;
  }

  // ARM64 AAPCS HFA (Homogeneous Floating-point Aggregate) requirements:
  // 1. All fields must be the SAME float type (f16, f32, or f64)
  // 2. At most 4 fields
  // 3. Passed in SIMD registers V0-V3

  // Check field count: HFA has at most 4 fields
  if (fields.size() > 4) {
    return false;
  }

  // Track the canonical float type bit width (must be homogeneous)
  std::optional<unsigned> canonicalBitWidth;

  for (mlir::Type fieldType : fields) {
    unsigned bitWidth;

    // MLIR FloatType (f16, f32, f64, etc.)
    if (auto floatType = dyn_cast<mlir::FloatType>(fieldType)) {
      bitWidth = floatType.getWidth();
    }
    // Vector types with float elements (converted from POP::SIMDType)
    else if (auto vecType = dyn_cast<mlir::VectorType>(fieldType)) {
      auto elemType = vecType.getElementType();
      if (auto floatElem = dyn_cast<mlir::FloatType>(elemType)) {
        bitWidth = floatElem.getWidth();
      } else {
        return false; // Non-float vector
      }
    }
    // TODO: Recursively check nested LLVM struct fields. For now,
    // conservatively reject nested structs.
    else {
      return false; // Non-float field
    }

    // Check homogeneity: all fields must have the same bit width
    if (!canonicalBitWidth) {
      canonicalBitWidth = bitWidth; // First field sets the canonical type
    } else if (*canonicalBitWidth != bitWidth) {
      return false; // Heterogeneous: mixed float types (e.g., f32 + f64)
    }
  }

  // Must have at least one field, and all fields must be the same float type
  return canonicalBitWidth.has_value();
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

bool CabiUtils::isCABIEnabled() {
  // C ABI coercion is enabled by default.
  // Use --skip-c-abi-coercion to disable (for debugging).
  return !CABIOpts->skipCoercion;
}
