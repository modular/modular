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

#include "CABICallHelpers.h"
#include "LLVMLoweringUtils.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"

using namespace M;
using namespace KGEN;
namespace LLVM = mlir::LLVM;

#define DEBUG_TYPE "lower-runtime-closures"

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERRUNTIMECLOSURES
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerRuntimeClosuresPass
    : M::KGEN::impl::LowerRuntimeClosuresBase<LowerRuntimeClosuresPass> {
  using LowerRuntimeClosuresBase::LowerRuntimeClosuresBase;

  void runOnOperation() override;
};
} // namespace

struct CreateRuntimeClosureOpConversion
    : public ConvertPOPToLLVMPattern<CreateFuncPtrOp> {
  using ConvertPOPToLLVMPattern<CreateFuncPtrOp>::ConvertPOPToLLVMPattern;

  LogicalResult matchAndRewrite(CreateFuncPtrOp op,
                                CreateFuncPtrOpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::AddressOfOp>(
        op, convertType(op.getType()),
        cast<FlatSymbolRefAttr>(
            cast<SymbolConstantAttr>(op.getCallee()).getSymbol()));
    return success();
  }
};

struct CallIndirectOpConversion
    : public ConvertPOPToLLVMPattern<CallIndirectOp> {
  using ConvertPOPToLLVMPattern<CallIndirectOp>::ConvertPOPToLLVMPattern;
  LogicalResult matchAndRewrite(CallIndirectOp op,
                                CallIndirectOpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    // If there are no result types, set it to void. Otherwise, set the result
    // type to the packed result types.
    Type resultType;
    if (!op.getNumResults())
      resultType = getVoidType();
    else
      resultType = getTypeConverter()->packFunctionResults(op.getResultTypes());
    if (!resultType)
      return emitError(op.getLoc(), "failed to convert call result type");

    Value callee = op.getCallee();
    LLVM::CallOp llvmCall;
    auto isCABIFnPtr = [](Type type) {
      if (auto sigType = dyn_cast<FuncTypeGeneratorType>(type))
        return sigType.getBody().isCABI();
      return false;
    };

    if (isCABIFnPtr(callee.getType())) {
      // C ABI function-pointer call: apply platform C ABI coercion.
      CABICallHelper cabi(getTypeConverter(), getContext(), op.getOperation());
      Location loc = op.getLoc();

      // Original LLVM return type (null for void functions).
      Type origRetTy = op.getNumResults() > 0 ? resultType : Type{};

      auto prep =
          cabi.prepareCall(op.getArguments().getTypes(), adaptor.getArguments(),
                           origRetTy, loc, rewriter);
      // Insert the indirect call target.
      prep.callArgs.insert(prep.callArgs.begin(), adaptor.getCallee());

      llvmCall = createLLVMCall(rewriter, loc, prep.signature, prep.callArgs);
      if (Value result =
              cabi.finishCall(llvmCall, prep, op.getTailKind(), loc, rewriter))
        rewriter.replaceOp(op, result);
      else
        rewriter.eraseOp(op);
      return success();
    }

    // Mojo ABI indirect call.
    // Note: adaptor.getOperands() is a list of callee followed by inputs.
    SmallVector<Type> wrapperFnArgTypes;
    llvm::append_range(wrapperFnArgTypes, adaptor.getArguments().getTypes());
    auto wrapperFnType = LLVM::LLVMFunctionType::get(getContext(), resultType,
                                                     wrapperFnArgTypes, 0);
    llvmCall = createLLVMCall(rewriter, op.getLoc(), wrapperFnType,
                              adaptor.getOperands());
    applyTailKind(llvmCall, op.getTailKind());

    if (op.getNumResults() <= 1) {
      rewriter.replaceOp(op, llvmCall.getResults());
      return success();
    }

    // Unpack the struct if necessary.
    SmallVector<Value> results;
    results.reserve(op.getNumResults());
    for (unsigned i = 0, e = op.getNumResults(); i < e; ++i)
      results.push_back(LLVM::ExtractValueOp::create(rewriter, op.getLoc(),
                                                     llvmCall.getResult(), i));

    // Replace the call operation.
    rewriter.replaceOp(op, results);
    return success();
  }
};

void LowerRuntimeClosuresPass::runOnOperation() {
  ModuleOp theModule = getOperation();

  mlir::RewritePatternSet patterns(&getContext());
  TargetInfoAttr targetInfo = lookupTargetInfo(theModule);
  if (!targetInfo) {
    mlir::emitError(theModule.getLoc(),
                    "could not find an enclosing target specification");
    return signalPassFailure();
  }
  POPToLLVMTypeConverter typeConverter(targetInfo);

  patterns.insert<CreateRuntimeClosureOpConversion>(typeConverter);
  patterns.insert<CallIndirectOpConversion>(typeConverter);

  auto mustConvert = [](Operation *op) {
    return isa<CreateFuncPtrOp, CallIndirectOp>(op);
  };
  if (failed(M::applyDirectConversion(theModule, &typeConverter,
                                      std::move(patterns), /*skipOp=*/nullptr,
                                      mustConvert)))
    return signalPassFailure();
}
