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
//
// MLIR-lowering policy for Apple Metal (air64) targets.
//
//===----------------------------------------------------------------------===//

#include "Target/TargetLowering.h"
#include "Target/max_license/Metal/MetalAddressSpace.h"
#include "Target/max_license/Metal/MetalTraits.h"

#include "KGENToLLVM/LLVMLoweringUtils.h"
#include "KGENToLLVM/Target/max_license/GPUCommon.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/POPDialect/POPAttrs.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/ToolCommon/CompilationOptions.h"
#include "Support/Compiler/MLIRDType.h"
#include "Support/Error.h"
#include "Support/ErrorOr.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TargetParser/Triple.h"

#include <string>
#include <utility>

using namespace mlir;

namespace M::KGEN {

// Attribute marking Metal kernel entry functions: set during this KGEN->LLVM
// lowering and read back by the Metal AIR backend (MetalAIRPass).
const char *kMetalKernelAttr = "metal.kernel";

namespace {

//===----------------------------------------------------------------------===//
// expandOperands
//===----------------------------------------------------------------------===//

/// Expand any KGEN struct operands into their (remapped) LLVM field values, so
/// AIR intrinsic calls receive the individual scalar/vector arguments.
static SmallVector<Value> expandOperands(DirectRewriter &rewriter, Location loc,
                                         ValueRange args, TypeRange origTypes,
                                         const POPToLLVMTypeConverter &tc) {
  SmallVector<Value> operands;
  operands.reserve(args.size());
  for (auto [value, origType] : llvm::zip(args, origTypes)) {
    // Squash pointless conversion casts that will get in the way of folds.
    value = squashPointlessCasts(value);

    // Check if the original type is a KGEN struct type.
    if (auto kgenStructTy = dyn_cast<StructType>(origType)) {
      // Get the element types - should be resolved at this point.
      if (auto elemTypes = kgenStructTy.getElementTypes()) {
        // Iterate over KGEN logical field indices and extract using remapped
        // LLVM indices to account for any padding fields.
        for (size_t i = 0, e = elemTypes->size(); i != e; ++i) {
          int64_t llvmIdx = tc.getRemappedFieldIndex(kgenStructTy, i);
          auto elt =
              rewriter.createOrFold<LLVM::ExtractValueOp>(loc, value, llvmIdx);
          operands.push_back(elt);
        }
      } else {
        // Parametric struct - just push the value as-is.
        operands.push_back(value);
      }
    } else {
      operands.push_back(value);
    }
  }
  return operands;
}

//===----------------------------------------------------------------------===//
// doesMetalRequireAIRIntrinsicForCast
//===----------------------------------------------------------------------===//

/// Return false if `pop.cast` can be lowered to LLVM instruction. Return true
/// otherwise, i.e. Metal emits special AIR intrinsic to do conversion from one
/// type to another.
/// NOTE: For some M-versions Metal backend seems to implement legalization of
/// the LLVM instruction, but unfortunately not for every target, which could
/// lead that `fptrunc`, for example, is supported on M3, but not on M2.
bool doesMetalRequireAIRIntrinsicForCast(POP::CastOp cast) {
  // https://linear.app/modularml/issue/MOCO-2875#comment-1b6b887a
  KGENDType inDType = *cast.getInput().getType().getResolvedDType();
  KGENDType outDType = *cast.getOutput().getType().getResolvedDType();
  const uint64_t size = *cast.getInput().getType().getResolvedSize();
  if (size > 1) {
    // Metal always uses AIR intrinsic to do vector conversion
    return true;
  }

  if (inDType.isFloat() != outDType.isFloat()) {
    // Metal always uses AIR intrinsic to do int <-> fp conversion
    return true;
  }

  // `fptrunc` and `fpext` can only be used for
  // f32 <-> bf16
  // f32 <-> f16
  if (inDType.isFloat() && outDType.isFloat()) {
    MLIRContext *ctx = cast.getContext();
    FloatType inFloat = getEquivalentFloatType(ctx, inDType);
    FloatType outFloat = getEquivalentFloatType(ctx, outDType);
    bool canUseLLVMInstruction =
        (isa<Float32Type>(inFloat) &&
         isa<Float16Type, BFloat16Type>(outFloat)) ||
        (isa<Float16Type, BFloat16Type>(inFloat) && isa<Float32Type>(outFloat));
    if (!canUseLLVMInstruction)
      return true;
  }

  // By default assume use of LLVM IR is supported
  return false;
}

//===----------------------------------------------------------------------===//
// MetalRoundToLLVM
//===----------------------------------------------------------------------===//

/// Lowers `pop.round` on Metal. Apple's Metal/AIR backend cannot codegen
/// `llvm.roundeven` (nor `llvm.nearbyint`): MTLCompilerService crashes JIT-ing
/// the GPU machine code with XPC_ERROR_CONNECTION_INTERRUPTED. It does support
/// `llvm.rint`, which rounds using the current FP rounding mode --
/// round-to-nearest-even by default, matching `pop.round`'s documented
/// banker's-rounding semantics, so the result is numerically identical.
/// Registered with a higher benefit than the generic `ConvertPOPRound` so it
/// wins on Metal.
struct MetalRoundToLLVM : public ConvertPOPToLLVMPattern<POP::RoundOp> {
  using ConvertPOPToLLVMPattern::ConvertPOPToLLVMPattern;

  LogicalResult matchAndRewrite(POP::RoundOp op, POP::RoundOpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    KGENDType dtype = *op.getType().getResolvedDType();
    if (!dtype.isFloat()) {
      rewriter.replaceOp(op, adaptor.getOperand());
      return success();
    }
    Type type = this->convertType(op.getType());
    rewriter.replaceOpWithNewOp<LLVM::CallIntrinsicOp>(
        op, type, rewriter.getStringAttr("llvm.rint"),
        SmallVector<Value>{adaptor.getOperand()});
    return success();
  }
};

/// Takes both spellings: `llvm.air.wg.barrier` and `air.wg.barrier`.
static bool isAIRBarrierName(StringRef name) {
  name.consume_front("llvm.");
  return name == "air.wg.barrier" || name == "air.simdgroup.barrier";
}

//===----------------------------------------------------------------------===//
// ConvertSymbolOpToAIR
//===----------------------------------------------------------------------===//

template <typename OpT>
struct ConvertSymbolOpToAIR : public ConvertSymbolOpToLLVM<OpT> {
  using BaseT = ConvertSymbolOpToLLVM<OpT>;
  using BaseT::BaseT;
  using BaseT::symtab;

  // Convenient class to hold immutable AIR intrinsic name and if it needs to be
  // mangled.
  class AIRIntrinsicName {
  public:
    explicit AIRIntrinsicName(const std::string &name,
                              bool requiresMangling = true)
        : name(name), needsMangling(requiresMangling) {}

    std::string getName() const { return name; }

    // Return true if AIR intrinsic was requested not to be mangled or has no
    // type overloading, i.e. no need to add types to its name.
    bool requiresMangling() const {
      // Function-local static to avoid a global constructor/destructor.
      static const llvm::StringSet<> unmangledSymbols = {
          "air.wg.barrier",
          "air.simdgroup.barrier",
      };
      return needsMangling && !unmangledSymbols.contains(name);
    }

  private:
    std::string name;
    bool needsMangling = true;
  };

  // Helper function to mangle type according to LLVM's mangling of args of
  // intrinsics.
  // See LLVM's getMangledTypeStr
  std::string mangleType(Type type) const {
    if (auto intTy = dyn_cast<IntegerType>(type)) {
      return "i" + llvm::utostr(intTy.getWidth());
    } else if (isa<BFloat16Type>(type)) {
      return "bf16";
    } else if (isa<Float8E5M2Type>(type)) {
      return "f8e5m2";
    } else if (isa<Float8E4M3FNType>(type)) {
      return "f8e4m3fn";
    } else if (isa<Float8E4M3Type>(type)) {
      return "f8e4m3";
    } else if (auto fltTy = dyn_cast<FloatType>(type)) {
      return "f" + llvm::utostr(fltTy.getWidth());
    } else if (auto vecTy = dyn_cast<VectorType>(type)) {
      auto shape = vecTy.getShape();
      assert(shape.size() == 1 &&
             "Multidimensional vectors are not supported.");
      assert(!vecTy.allDimsScalable() &&
             "Scalable vectors are not yet supported.");
      return "v" + llvm::utostr(shape[0]) + mangleType(vecTy.getElementType());
    } else if (auto simdTy = dyn_cast<SIMDType>(type)) {
      // Mangle from the dtype, not the LLVM lowering: keeps fp8 (which lowers
      // to i8 storage) in the name as its logical type.
      KGENDType dtype = *simdTy.getResolvedDType();
      return "v" + llvm::utostr(*simdTy.getResolvedSize()) +
             mangleType(dtype.getEquivalentBuiltinType(type.getContext()));
    } else {
      llvm_unreachable("Unknown type to mangle");
    }
  }

  LLVM::LLVMFuncOp createAIRFunction(DirectRewriter &rewriter, ModuleOp module,
                                     Location loc, AIRIntrinsicName &airName,
                                     ValueRange operands,
                                     TypeRange resultTypes) const {
    SmallVector<Type> argTypes;
    for (Value value : operands)
      argTypes.push_back(value.getType());

    // Convert AIR intrinsic to regular function call with mangled argument
    // types. Additionally remove "llvm." prefix to get
    // "air.function_name.<type0>.<type1>..."
    // TODO: Support result type if it's different to type of argument
    std::string funcName = airName.getName();
    if (airName.requiresMangling()) {
      SmallVector<Type> uniqueTypes(argTypes.begin(), argTypes.end());
      for (auto it = uniqueTypes.begin(), ei = llvm::unique(uniqueTypes);
           it != ei; ++it) {
        funcName += "." + mangleType(*it);
      }
    }

    auto func = symtab.template lookup<LLVM::LLVMFuncOp>(funcName);
    if (!func) {
      // Create LLVM function type from the intrinsic operation

      Type returnType = resultTypes.empty()
                            ? LLVM::LLVMVoidType::get(rewriter.getContext())
                            : resultTypes[0];
      auto llvmFuncType = LLVM::LLVMFunctionType::get(returnType, argTypes);

      // Get or create the AIR function declaration
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());
      func = LLVM::LLVMFuncOp::create(rewriter,
                                      UnknownLoc::get(rewriter.getContext()),
                                      funcName, llvmFuncType);
      symtab.insert(func);
    }
    // Outside the branch: `external_call` may have declared it unmarked.
    if (isAIRBarrierName(funcName))
      func.setConvergent(true);
    return func;
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCallToAIRIntrinsic
//===----------------------------------------------------------------------===//

struct ConvertPOPCallToAIRIntrinsic
    : public ConvertSymbolOpToAIR<POP::CallLLVMIntrinsicOp> {
  using BaseT = ConvertSymbolOpToAIR<POP::CallLLVMIntrinsicOp>;
  using BaseT::BaseT;
  using BaseT::createAIRFunction;

  LogicalResult matchAndRewrite(POP::CallLLVMIntrinsicOp op,
                                POP::CallLLVMIntrinsicOpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    SmallVector<Type> types;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), types)))
      return failure();

    StringRef intrinsicName = cast<StringAttr>(op.getIntrin());
    assert(intrinsicName.starts_with("llvm.air.") &&
           "non-AIR LLVM intrinsic must be processed by "
           "ConvertPOPCallLLVMIntrinsic");

    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return failure();

    SmallVector<Value> newOperands =
        expandOperands(rewriter, op.getLoc(), adaptor.getOperands(),
                       op.getOperands().getTypes(), *getTypeConverter());

    // Handle texture intrinsics with special mangling based on data type.
    // Texture intrinsics are mangled by the pixel data type only, not all args.
    StringRef airName = intrinsicName.drop_front(5); // Drop "llvm."
    if (airName.starts_with("air.read_texture_") ||
        airName.starts_with("air.write_texture_") ||
        airName.starts_with("air.sample_texture_")) {
      LLVM::LLVMFuncOp func = createTextureAIRFunction(
          rewriter, module, op.getLoc(), airName, newOperands, types);
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, func, newOperands);
      return success();
    }

    // simdgroup_matrix intrinsics use custom mangling
    if (airName.starts_with("air.simdgroup_matrix_")) {
      ErrorOr<LLVM::LLVMFuncOp> funcOr = createSimdgroupMatrixAIRFunction(
          rewriter, module, op.getLoc(), airName, newOperands, types,
          op.getOperands().getTypes());
      if (funcOr.isError()) {
        op.emitError() << funcOr.takeError();
        return failure();
      }
      rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, funcOr.takeValue(),
                                                newOperands);
      return success();
    }

    AIRIntrinsicName airFunctionName(airName.str());
    LLVM::LLVMFuncOp func = createAIRFunction(
        rewriter, module, op.getLoc(), airFunctionName, newOperands, types);

    // Replace the intrinsic call with a regular function call
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, func, newOperands);
    return success();
  }

private:
  // Create AIR function for texture intrinsics with special mangling.
  // Texture intrinsics are mangled by the pixel data type (vector float type),
  // not by all argument types.
  // For read_texture: mangle by the result type's vector element
  // For write_texture: mangle by the data argument type (3rd operand)
  LLVM::LLVMFuncOp createTextureAIRFunction(DirectRewriter &rewriter,
                                            ModuleOp module, Location loc,
                                            StringRef airName,
                                            ValueRange operands,
                                            TypeRange resultTypes) const {
    SmallVector<Type> argTypes;
    for (Value value : operands)
      argTypes.push_back(value.getType());

    // Determine the data type for mangling based on intrinsic type
    Type dataType;
    if (airName.starts_with("air.read_texture_") ||
        airName.starts_with("air.sample_texture_")) {
      // For read/sample: result is {<N x T>, i8}, mangle by the vector type
      if (!resultTypes.empty()) {
        Type resultType = resultTypes[0];
        if (auto structTy = dyn_cast<LLVM::LLVMStructType>(resultType)) {
          // Extract the vector type from the struct {<4 x float>, i8}
          if (!structTy.getBody().empty())
            dataType = structTy.getBody()[0];
        } else {
          dataType = resultType;
        }
      }
    } else if (airName.starts_with("air.write_texture_")) {
      // For write: data is the 3rd operand (index 2): ptr, coords, data, ...
      if (operands.size() >= 3)
        dataType = operands[2].getType();
    }

    // Build the mangled function name
    std::string funcName = airName.str();
    if (dataType) {
      funcName += "." + mangleType(dataType);
    }

    AIRIntrinsicName airFunctionName(funcName, /*requiresMangling=*/false);
    LLVM::LLVMFuncOp func = createAIRFunction(
        rewriter, module, loc, airFunctionName, operands, resultTypes);
    return func;
  }

  // Create AIR function for simdgroup_matrix intrinsics with custom mangling.
  ErrorOr<LLVM::LLVMFuncOp>
  createSimdgroupMatrixAIRFunction(DirectRewriter &rewriter, ModuleOp module,
                                   Location loc, StringRef airName,
                                   ValueRange operands, TypeRange resultTypes,
                                   TypeRange origOpTypes) const {
    enum class SimdgroupMatrixKind {
      k8x8,
      k16x16,
      kUnknown,
    };

    SmallVector<Type> argTypes;
    for (Value value : operands)
      argTypes.push_back(value.getType());
    const SimdgroupMatrixKind simdgroupKind = [&airName]() {
      if (airName.contains("16x16"))
        return SimdgroupMatrixKind::k16x16;
      if (airName.contains("8x8"))
        return SimdgroupMatrixKind::k8x8;
      return SimdgroupMatrixKind::kUnknown;
    }();

    if (simdgroupKind == SimdgroupMatrixKind::kUnknown)
      return Error("unknown 'simdgroup_matrix' intrinsic");

    if (simdgroupKind == SimdgroupMatrixKind::k16x16 && argTypes.size() != 5) {
      return Error(
          "expected expanded operands: A, transpose_a, B, transpose_b, C");
    } else if (simdgroupKind == SimdgroupMatrixKind::k8x8 &&
               argTypes.size() != 3) {
      return Error("expected expanded operands: A, B, C");
    }

    if (resultTypes.empty())
      return Error("expected result type");

    // Extract KGENDType from the original kgen struct's A and B fields to
    // determine the scalar-type chars.
    if (origOpTypes.size() != 1)
      return Error("expected single struct operand");

    auto kgenStructTy = cast<StructType>(origOpTypes[0]);
    auto elemTypes = *kgenStructTy.getElementTypes();
    const uint32_t aIdx = 0;
    const uint32_t bIdx = simdgroupKind == SimdgroupMatrixKind::k16x16 ? 2 : 1;
    const uint32_t cIdx = simdgroupKind == SimdgroupMatrixKind::k16x16 ? 4 : 2;

    auto aDType = *cast<SIMDType>(elemTypes[aIdx]).getResolvedDType();
    auto bDType = *cast<SIMDType>(elemTypes[bIdx]).getResolvedDType();

    //   'f' - floating point
    //   's' - signed integer
    //   'u' - unsigned integer
    auto mangleScalarTypeChar = [](KGENDType dtype) {
      assert(!dtype.isAddress() &&
             "address dtype not valid for simdgroup_matrix");
      if (dtype.isIntLike())
        return (dtype.isSInt() || dtype.isIndex()) ? 's' : 'u';
      return 'f';
    };

    // Mangled name:
    //   airName.{charA}.{charB}.{mangledD}.{mangledA}.{mangledB}.{mangledC}
    std::string funcName = airName.str();
    funcName += '.';
    funcName += mangleScalarTypeChar(aDType);
    funcName += '.';
    funcName += mangleScalarTypeChar(bDType);
    funcName += "." + mangleType(resultTypes[0]);  // D
    funcName += "." + mangleType(elemTypes[aIdx]); // A
    funcName += "." + mangleType(elemTypes[bIdx]); // B
    funcName += "." + mangleType(argTypes[cIdx]);  // C

    AIRIntrinsicName airFunctionName(funcName, /*requiresMangling=*/false);
    return createAIRFunction(rewriter, module, loc, airFunctionName, operands,
                             resultTypes);
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPMinToAIR
//===----------------------------------------------------------------------===//

struct ConvertPOPMinToAIR : public ConvertSymbolOpToAIR<POP::MinOp> {
  using BaseT = ConvertSymbolOpToAIR<POP::MinOp>;
  using BaseT::BaseT;
  using BaseT::createAIRFunction;

  LogicalResult matchAndRewrite(POP::MinOp op, POP::MinOpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    SmallVector<Type> types;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), types)))
      return failure();

    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return failure();

    SmallVector<Value> operands = adaptor.getOperands();
    KGENDType dtype = *op.getType().getResolvedDType();

    std::string funcName;

    if (dtype.isSInt() || dtype.isIndex()) {
      funcName = "air.min.s";
    } else if (dtype.isBool() || dtype.isUInt() || dtype.isUIndex()) {
      funcName = "air.min.u";
    } else {
      // That corresponds to fp min with precise fp model.
      // TODO: With 'fast' fp model support, use `air.fast_min` intrinsic
      // instead.
      funcName = "air.fmin";
    }

    AIRIntrinsicName airFunctionName(funcName);
    LLVM::LLVMFuncOp func = createAIRFunction(rewriter, module, op.getLoc(),
                                              airFunctionName, operands, types);

    // Replace the intrinsic call with a regular function call
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, func, operands);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPMaxToAIR
//===----------------------------------------------------------------------===//

struct ConvertPOPMaxToAIR : public ConvertSymbolOpToAIR<POP::MaxOp> {
  using BaseT = ConvertSymbolOpToAIR<POP::MaxOp>;
  using BaseT::BaseT;
  using BaseT::createAIRFunction;

  LogicalResult matchAndRewrite(POP::MaxOp op, POP::MaxOpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    SmallVector<Type> types;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), types)))
      return failure();

    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return failure();

    SmallVector<Value> operands = adaptor.getOperands();
    KGENDType dtype = *op.getType().getResolvedDType();

    std::string funcName;

    if (dtype.isSInt() || dtype.isIndex()) {
      funcName = "air.max.s";
    } else if (dtype.isBool() || dtype.isUInt() || dtype.isUIndex()) {
      funcName = "air.max.u";
    } else {
      // That corresponds to fp max with precise fp model.
      // TODO: With 'fast' fp model support, use `air.fast_max` intrinsic
      // instead.
      funcName = "air.fmax";
    }

    AIRIntrinsicName airFunctionName(funcName);
    LLVM::LLVMFuncOp func = createAIRFunction(rewriter, module, op.getLoc(),
                                              airFunctionName, operands, types);

    // Replace the intrinsic call with a regular function call
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, func, operands);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPCastToAIR
//===----------------------------------------------------------------------===//

struct ConvertPOPCastToAIR : public ConvertSymbolOpToAIR<POP::CastOp> {
  using BaseT = ConvertSymbolOpToAIR<POP::CastOp>;
  using BaseT::BaseT;
  using BaseT::createAIRFunction;
  using BaseT::mangleType;

  LogicalResult matchAndRewrite(POP::CastOp op, POP::CastOpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    SmallVector<Type> types;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), types)))
      return failure();

    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return failure();

    SmallVector<Value> operands = adaptor.getOperands();
    KGENDType inDType = *op.getInput().getType().getResolvedDType();
    KGENDType outDType = *op.getOutput().getType().getResolvedDType();

    std::string funcName = "air.convert";

    // Helper function to generate AIR-specific mangling of the type that has
    // special prefix
    //  's' - signed integer
    //  'u' - unsigned integer
    //  'f' - floating point
    auto mangleType = [&](Type type, KGENDType dtype) -> std::string {
      if (dtype.isIntLike()) {
        return (dtype.isSInt() || dtype.isIndex() ? "s." : "u.") +
               BaseT::mangleType(type);
      }
      return "f." + BaseT::mangleType(type);
    };

    Type inType = operands[0].getType();
    Type outType = types[0];
    funcName +=
        "." + mangleType(outType, outDType) + "." + mangleType(inType, inDType);

    AIRIntrinsicName airFunctionName(funcName, /*requiresMangling=*/false);
    LLVM::LLVMFuncOp func = createAIRFunction(rewriter, module, op.getLoc(),
                                              airFunctionName, operands, types);

    // Replace the intrinsic call with a regular function call
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, func, operands);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// checkAtomicWidth / checkAtomicOrdering / resolveAtomicAddressSpace
//===----------------------------------------------------------------------===//

//  TODO: bool has to be zexted
// AIR's atomic builtin (used by RMW/cmpxchg) only exists at 32 bits;
// verified against Apple's `metal` compiler.
LogicalResult checkAtomicWidth(mlir::Operation *op, KGENDType dtype,
                               TargetInfoAttr target) {
  if (dtype.getWidthInBits(target) != 32)
    return op->emitError(
        "Atomic operation is not supported for this type on Apple GPU");
  return success();
}

// AIR only exposes `relax`; the remaining orderings have no AIR equivalent.
LogicalResult checkAtomicOrdering(mlir::Operation *op,
                                  POP::AtomicOrdering ordering) {
  switch (ordering) {
  case POP::AtomicOrdering::NOT_ATOMIC:
  case POP::AtomicOrdering::UNORDERED:
  case POP::AtomicOrdering::MONOTONIC:
    return success();
  case POP::AtomicOrdering::RELEASE:
    return op->emitError(
        "Apple GPU does not support `release` atomic ordering");
  case POP::AtomicOrdering::ACQUIRE:
    return op->emitError(
        "Apple GPU does not support `acquire` atomic ordering");
  case POP::AtomicOrdering::ACQUIRE_RELEASE:
    return op->emitError(
        "Apple GPU does not support `acquire_release` atomic ordering");
  case POP::AtomicOrdering::SEQUENTIALLY_CONSISTENT:
    return op->emitError(
        "Apple GPU does not support `seq_cst` atomic ordering");
  }
  llvm_unreachable("unknown atomic ordering");
}

// AIR atomics only support threadgroup and device memory; address space 4
// (`AddressSpace.CONSTANT`) is deliberately rejected here (see MSTDL-1974 for
// the open question of what it should mean on Apple GPU). Unannotated kernel
// argument pointers (address space 0) default to device memory.
FailureOr<int64_t> resolveAtomicAddressSpace(mlir::Operation *op,
                                             PointerType ptrType) {
  int64_t addressSpace = ptrType.getAddrSpaceOrZero();
  if (addressSpace == 0) {
    addressSpace = MetalAddressSpace::Device;
  }

  if (addressSpace != MetalAddressSpace::ThreadGroup &&
      addressSpace != MetalAddressSpace::Device) {
    op->emitError("Atomic operation is only supported for local (")
        << static_cast<int>(MetalAddressSpace::ThreadGroup) << ") and global ("
        << static_cast<int>(MetalAddressSpace::Device)
        << ") address spaces on Apple GPU, but got address space '"
        << addressSpace << "'";
    return failure();
  }
  return addressSpace;
}

//===----------------------------------------------------------------------===//
// ConvertPOPAtomicRMWToAIR
//===----------------------------------------------------------------------===//

class ConvertPOPAtomicRMWToAIR : public ConvertSymbolOpToAIR<POP::AtomicRMWOp> {
public:
  using BaseT = ConvertSymbolOpToAIR<POP::AtomicRMWOp>;
  using BaseT::BaseT;
  using BaseT::createAIRFunction;
  using BaseT::mangleType;

  LogicalResult matchAndRewrite(POP::AtomicRMWOp op,
                                POP::AtomicRMWOpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    SmallVector<Type> types;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), types)))
      return failure();
    TargetInfoAttr target = getTypeConverter()->getTarget();

    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return failure();

    SmallVector<Value> operands = {
        adaptor.getPtr(),
        adaptor.getVal(),
    };
    KGENDType dtype = *cast<SIMDType>(op.getType()).getResolvedDType();

    if (failed(checkAtomicWidth(op, dtype, target)))
      return failure();
    const std::string opStr = getOpStr(op.getBinOp());
    if (opStr.empty()) {
      return op.emitError("atomic operation ")
             << op.getBinOp() << " is not supported on Apple GPU";
    }

    if (failed(checkAtomicOrdering(op, op.getOrdering())))
      return failure();

    FailureOr<int64_t> addressSpaceOr =
        resolveAtomicAddressSpace(op, op.getPtr().getType());
    if (failed(addressSpaceOr))
      return failure();
    int64_t addressSpace = *addressSpaceOr;

    std::string funcName = "air.atomic.";
    if (addressSpace == MetalAddressSpace::Device)
      funcName += "global";
    else if (addressSpace == MetalAddressSpace::ThreadGroup)
      funcName += "local";

    // Helper function to generate AIR-specific mangling of the type that has
    // special prefix
    //  's' - signed integer
    //  'u' - unsigned integer
    //  'f' - floating point
    auto mangleType = [&](Type type, KGENDType dtype) -> std::string {
      if (dtype.isIntLike()) {
        return (dtype.isSInt() || dtype.isIndex() ? "s." : "u.") +
               BaseT::mangleType(type);
      }
      // Unlike air.convert, atomics don't have `f.` prefix in mangled name.
      return BaseT::mangleType(type);
    };

    funcName += '.' + opStr + '.' + mangleType(types[0], dtype);

    Value zeroInt = LLVM::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), 0));
    Value twoInt = LLVM::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), 2));
    Value trueVal = LLVM::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 1));
    operands.push_back(zeroInt);
    operands.push_back(twoInt);
    operands.push_back(trueVal);

    AIRIntrinsicName airFunctionName(funcName, /*requiresMangling=*/false);
    LLVM::LLVMFuncOp func = createAIRFunction(rewriter, module, op.getLoc(),
                                              airFunctionName, operands, types);

    // Replace the intrinsic call with a regular function call
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, func, operands);
    return success();
  }

private:
  std::string getOpStr(POP::AtomicBinOp binOp) const {
    switch (binOp) {
    case POP::AtomicBinOp::XCHG:
      return "xchg";
    case POP::AtomicBinOp::ADD:
      return "add";
    case POP::AtomicBinOp::SUB:
      return "sub";
    case POP::AtomicBinOp::AND:
      return "and";
    case POP::AtomicBinOp::NAND:
      return ""; // Not supported directly
    case POP::AtomicBinOp::OR:
      return "or";
    case POP::AtomicBinOp::XOR:
      return "xor";
    case POP::AtomicBinOp::MAX:
      return "max";
    case POP::AtomicBinOp::MIN:
      return "min";
    }
    llvm_unreachable("unknown atomic binop");
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPAtomicCmpXchgToAIR
//===----------------------------------------------------------------------===//

// AIR only exposes `cmpxchg.weak`, so this pattern requires the op's `weak`
// bit and rejects the non-weak form. It returns the old value rather than a
// success flag, so success is derived by comparing that to `cmp`.
class ConvertPOPAtomicCmpXchgToAIR
    : public ConvertSymbolOpToAIR<POP::AtomicCmpXchgOp> {
public:
  using BaseT = ConvertSymbolOpToAIR<POP::AtomicCmpXchgOp>;
  using BaseT::BaseT;
  using BaseT::createAIRFunction;
  using BaseT::mangleType;

  LogicalResult matchAndRewrite(POP::AtomicCmpXchgOp op,
                                POP::AtomicCmpXchgOpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    SmallVector<Type> types;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), types)))
      return failure();
    TargetInfoAttr target = getTypeConverter()->getTarget();

    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return failure();

    KGENDType dtype = *cast<SIMDType>(op.getCmp().getType()).getResolvedDType();

    if (failed(checkAtomicWidth(op, dtype, target)))
      return failure();

    if (!op.getWeak()) {
      return op.emitError(
          "Apple GPU only supports `weak` compare-exchange; AIR exposes no "
          "strong compare-exchange primitive");
    }

    for (POP::AtomicOrdering ordering :
         {op.getSuccessOrdering(), op.getFailureOrdering()}) {
      if (failed(checkAtomicOrdering(op, ordering)))
        return failure();
    }

    FailureOr<int64_t> addressSpaceOr =
        resolveAtomicAddressSpace(op, op.getPtr().getType());
    if (failed(addressSpaceOr))
      return failure();
    int64_t addressSpace = *addressSpaceOr;

    // AIR encodes memory scope as an explicit operand (1 for threadgroup, 2
    // for device) rather than deriving it from the pointer's address space.
    // AIR doesn't document this encoding; the values match Metal's own output.
    bool isGlobal = addressSpace == MetalAddressSpace::Device;
    std::string funcName = isGlobal ? "air.atomic.global" : "air.atomic.local";
    int32_t scope = isGlobal ? 2 : 1;

    Type valueType = adaptor.getCmp().getType();
    funcName += ".cmpxchg.weak." + mangleType(valueType);

    // AIR's cmpxchg takes the expected value by pointer, unlike LLVM's
    // `cmpxchg` instruction, so it must be spilled to a scratch alloca.
    Value one = LLVM::ConstantOp::create(rewriter, op.getLoc(),
                                         rewriter.getI64IntegerAttr(1));
    Value expectedPtr = LLVM::AllocaOp::create(
        rewriter, op.getLoc(),
        LLVM::LLVMPointerType::get(rewriter.getContext()), valueType, one);
    LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getCmp(), expectedPtr);

    Value zeroInt = LLVM::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), 0));
    Value scopeInt = LLVM::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIntegerAttr(rewriter.getI32Type(), scope));
    Value trueVal = LLVM::ConstantOp::create(
        rewriter, op.getLoc(),
        rewriter.getIntegerAttr(rewriter.getI1Type(), 1));

    SmallVector<Value> operands = {
        adaptor.getPtr(), expectedPtr, adaptor.getVal(), zeroInt,
        zeroInt,          scopeInt,    trueVal};

    AIRIntrinsicName airFunctionName(funcName, /*requiresMangling=*/false);
    LLVM::LLVMFuncOp func = createAIRFunction(
        rewriter, module, op.getLoc(), airFunctionName, operands, {valueType});
    Value oldValue =
        LLVM::CallOp::create(rewriter, op.getLoc(), func, operands).getResult();

    Value swapped =
        LLVM::ICmpOp::create(rewriter, op.getLoc(), LLVM::ICmpPredicate::eq,
                             adaptor.getCmp(), oldValue);

    Value result = LLVM::UndefOp::create(rewriter, op.getLoc(), types[0]);
    result = LLVM::InsertValueOp::create(rewriter, op.getLoc(), result,
                                         oldValue, ArrayRef<int64_t>{0});
    result = LLVM::InsertValueOp::create(rewriter, op.getLoc(), result, swapped,
                                         ArrayRef<int64_t>{1});

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertPOPFenceToAIR
//===----------------------------------------------------------------------===//

// air.atomic.fence(i32 mem_flags, i32 memory_order, i32 thread_scope), per
// Apple's `metal` compiler output; orders use C++ std::memory_order values.

enum class AIRFenceMemFlags : int32_t {
  Device = 1,
  Threadgroup = 2,
  All = 3, // Device | Threadgroup
};

enum class AIRMemoryOrder : int32_t {
  Acquire = 2,
  Release = 3,
  AcqRel = 4,
  SeqCst = 5,
};

enum class AIRThreadScope : int32_t {
  Thread = 0,
  Threadgroup = 1,
  Device = 2,
  Simdgroup = 4,
};

// Directional orderings need the `metal4_0` deployment-floor target feature.
FailureOr<AIRMemoryOrder> mapFenceOrdering(mlir::Operation *op,
                                           POP::AtomicOrdering ordering,
                                           TargetInfoAttr target) {
  switch (ordering) {
  case POP::AtomicOrdering::SEQUENTIALLY_CONSISTENT:
    return AIRMemoryOrder::SeqCst;
  case POP::AtomicOrdering::ACQUIRE:
  case POP::AtomicOrdering::RELEASE:
  case POP::AtomicOrdering::ACQUIRE_RELEASE: {
    if (target.hasFeature("metal4_0")) {
      if (ordering == POP::AtomicOrdering::ACQUIRE)
        return AIRMemoryOrder::Acquire;
      if (ordering == POP::AtomicOrdering::RELEASE)
        return AIRMemoryOrder::Release;
      return AIRMemoryOrder::AcqRel;
    }
    op->emitError("Apple GPU supports `")
        << POP::stringifyAtomicOrdering(ordering)
        << "` fence ordering only with a Metal 4 deployment target; use "
           "`seq_cst` or raise the deployment target";
    return failure();
  }
  default:
    op->emitError("invalid fence ordering on Apple GPU");
    return failure();
  }
}

// Maps Mojo `Scope` names to Metal scopes. `system` is rejected: Metal
// cannot order against the host. An unscoped fence gets device scope.
FailureOr<AIRThreadScope> mapFenceSyncscope(mlir::Operation *op,
                                            std::optional<StringRef> scope) {
  if (!scope || scope->empty())
    return AIRThreadScope::Device;
  auto mapped = llvm::StringSwitch<std::optional<AIRThreadScope>>(*scope)
                    .Case("thread", AIRThreadScope::Thread)
                    .Case("warp", AIRThreadScope::Simdgroup)
                    .Case("block", AIRThreadScope::Threadgroup)
                    .Case("gpu", AIRThreadScope::Device)
                    .Default(std::nullopt);
  if (mapped)
    return *mapped;
  if (*scope == "cluster") {
    op->emitError("Apple GPU has no cluster scope; use syncscope \"gpu\" for "
                  "device-wide ordering (a strict superset of cluster)");
    return failure();
  }
  if (*scope == "system") {
    op->emitError("Apple GPU has no system-scope fence: Metal cannot order "
                  "memory against the host");
    return failure();
  }
  op->emitError("unknown fence syncscope \"")
      << *scope
      << "\" on Apple GPU; expected \"thread\", \"warp\", \"block\", or "
         "\"gpu\"";
  return failure();
}

class ConvertPOPFenceToAIR : public ConvertSymbolOpToAIR<POP::FenceOp> {
public:
  using BaseT = ConvertSymbolOpToAIR<POP::FenceOp>;
  using BaseT::BaseT;
  using BaseT::createAIRFunction;

  LogicalResult matchAndRewrite(POP::FenceOp op, POP::FenceOpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    auto module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return failure();
    TargetInfoAttr target = getTypeConverter()->getTarget();

    FailureOr<AIRMemoryOrder> order =
        mapFenceOrdering(op, adaptor.getOrdering(), target);
    if (failed(order))
      return failure();

    std::optional<StringRef> scope;
    if (StringAttr scopeAttr = adaptor.getSyncscopeAttr())
      scope = scopeAttr.getValue();
    FailureOr<AIRThreadScope> threadScope = mapFenceSyncscope(op, scope);
    if (failed(threadScope))
      return failure();

    auto i32Const = [&](int32_t value) {
      return LLVM::ConstantOp::create(
          rewriter, op.getLoc(),
          rewriter.getIntegerAttr(rewriter.getI32Type(), value));
    };
    SmallVector<Value> operands = {
        i32Const(static_cast<int32_t>(AIRFenceMemFlags::All)),
        i32Const(static_cast<int32_t>(*order)),
        i32Const(static_cast<int32_t>(*threadScope))};

    AIRIntrinsicName airFunctionName("air.atomic.fence",
                                     /*requiresMangling=*/false);
    LLVM::LLVMFuncOp func =
        createAIRFunction(rewriter, module, op.getLoc(), airFunctionName,
                          operands, /*resultTypes=*/{});
    rewriter.replaceOpWithNewOp<LLVM::CallOp>(op, func, operands);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MetalLowering
//===----------------------------------------------------------------------===//

class MetalLowering final : public TargetLowering {
public:
  const TargetTraits *traits() const override { return &MetalTraits::get(); }

  mlir::Type lowerKernelArgToMemory(mlir::Type type) const override {
    return lowerAggregateArgToMemory(type);
  }

  // MetalDeviceContext cannot distinguish a `ptr` argument from a `{{ptr}, {}}`
  // argument, so a `kgen.pointer` passed in a register is wrapped to force it
  // to be passed by memory.
  mlir::Type
  getKernelArgIndirectionType(mlir::Type type,
                              ArgConvention convention) const override {
    if (auto ptr = mlir::dyn_cast<PointerType>(type);
        ptr && convention == ArgConvention::ImmReg)
      return PointerType::get(ptr);
    return {};
  }

  bool needsVerification() const override { return true; }

  mlir::LogicalResult verifyOp(mlir::Operation *op) const override {
    return verifyNoGPUHeapAllocation(op);
  }

  // AIR barriers are external calls, so the intrinsic probe never flags them.
  bool isConvergentOp(mlir::Operation *op) const override {
    auto intrinsic = mlir::dyn_cast<POP::CallLLVMIntrinsicOp>(op);
    return intrinsic &&
           isAIRBarrierName(
               mlir::cast<mlir::StringAttr>(intrinsic.getIntrin()).getValue());
  }

  void populateLowerPOPToLLVMPatterns(mlir::RewritePatternSet &patterns,
                                      mlir::LLVMTypeConverter &converter,
                                      TargetInfoAttr target) const override {
    patterns.add<MetalRoundToLLVM>(converter, mlir::PatternBenefit(2));
  }

  void populateLowerGlobalPOPToLLVMPatterns(
      mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &converter,
      mlir::SymbolTable &symtab, TargetInfoAttr target) const override {
    patterns.insert<ConvertPOPCallToAIRIntrinsic, ConvertPOPMinToAIR,
                    ConvertPOPMaxToAIR, ConvertPOPCastToAIR,
                    ConvertPOPAtomicRMWToAIR, ConvertPOPAtomicCmpXchgToAIR,
                    ConvertPOPFenceToAIR>(converter, symtab);
  }

  bool isLoweredInGlobalPOPPass(mlir::Operation *op) const override {
    if (isa<POP::MaxOp, POP::MinOp, POP::AtomicRMWOp, POP::AtomicCmpXchgOp,
            POP::FenceOp>(op))
      return true;
    if (auto cast = dyn_cast<POP::CastOp>(op))
      return doesMetalRequireAIRIntrinsicForCast(cast);
    // AIR intrinsics are lowered to their function declarations in the global
    // pass, which owns the symtab.
    if (auto call = dyn_cast<POP::CallLLVMIntrinsicOp>(op)) {
      return cast<StringAttr>(call.getIntrin())
          .getValue()
          .starts_with("llvm.air.");
    }
    return false;
  }

  void markExportedKernel(mlir::Operation *func) const override {
    auto funcOp = mlir::cast<mlir::LLVM::LLVMFuncOp>(func);
    mlir::Builder b(funcOp.getContext());
    llvm::SmallVector<mlir::Attribute> passthrough =
        llvm::to_vector(funcOp.getPassthroughAttr());
    passthrough.push_back(b.getArrayAttr(
        {b.getStringAttr(kMetalKernelAttr), b.getStringAttr("true")}));
    funcOp.setPassthroughAttr(b.getArrayAttr(passthrough));
  }

  mlir::LogicalResult
  mapFuncMetadata(mlir::Operation *func, mlir::NamedAttribute attr,
                  mlir::NamedAttrList &funcAttrs,
                  llvm::SmallVectorImpl<mlir::Attribute> &passthrough,
                  bool &handled) const override {
    handled = false;
    if (!isa<POP::POPDialect>(attr.getNameDialect()))
      return success();
    if (attr.getName() != "pop.air.max_work_group_size")
      return success();
    // Metal does not support a min value for workgroup size, so a min other
    // than 1 is ignored. Record `pop.air.max_work_group_size` as a passthrough
    // string attribute (dropping the `pop.` prefix) for LLVM IR.
    ErrorOr<std::pair<int64_t, int64_t>> valuesOr =
        getWorkGroupSizeRange(attr.getValue());
    if (valuesOr.isError())
      return emitError(func->getLoc(), "unsupported `")
             << attr.getName() << "`: " << valuesOr.takeError();
    std::pair<int64_t, int64_t> values = valuesOr.get();
    Builder b(func->getContext());
    passthrough.push_back(
        b.getArrayAttr({b.getStringAttr(attr.getName().strref().drop_front(
                            StringRef("pop.").size())),
                        b.getStringAttr(std::to_string(values.second))}));
    handled = true;
    return success();
  }

  bool typeNeedsExtraLegalization(KGENDType dtype,
                                  TargetInfoAttr target) const override {
    // M1 (Apple7) and M2 (Apple8) lack native bf16 arithmetic; M3+ supports it.
    return dtype == KGENDType::bf16 && isAppleM1OrM2(target);
  }

  bool castHandledByLaterPass(KGENDType fromDType, KGENDType toDType,
                              TargetInfoAttr target) const override {
    // On Metal, int<->float casts go through AIR intrinsics and don't need
    // legalization here; fp8 types still do.
    if (fromDType.isFloat() == toDType.isFloat())
      return false;
    KGENDType floatDType = fromDType.isFloat() ? fromDType : toDType;
    return floatDType.getWidthInBits() > 8;
  }

  // This is max only target
  bool isBaseTarget() const override { return false; }

private:
  /// Returns whether `target` is an Apple M1 or M2 GPU.
  static bool isAppleM1OrM2(TargetInfoAttr target) {
    StringRef arch = target.getArch();
    bool isM1 = arch == "1" || arch.starts_with("1-") || arch == "apple-m1" ||
                arch.starts_with("apple-m1-");
    bool isM2 = arch == "2" || arch.starts_with("2-") || arch == "apple-m2" ||
                arch.starts_with("apple-m2-");
    return isM1 || isM2;
  }
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wglobal-constructors"
RegisterTargetLowering<MetalLowering> registerMetalLowering;
#pragma GCC diagnostic pop

} // namespace
} // namespace M::KGEN
