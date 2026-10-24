//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTensorInterface.h"
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h"
#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace M::MGP {

// Generate support code for VerifyModelArgumentsPass.
#define GEN_PASS_DEF_VERIFYMODELARGUMENTSPASS
#include "GraphCompiler/MGPDialect/Transforms/MGPTransforms.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// VerifyModelArgumentsPass
//===----------------------------------------------------------------------===//

static std::string shapeToString(ArrayRef<int64_t> shape) {
  const auto dimToString = [](int64_t dim) {
    static constexpr std::string_view kDynamicDimensionLabel = "any";
    if (dim == ShapedType::kDynamic)
      return std::string(kDynamicDimensionLabel);

    return std::to_string(dim);
  };
  std::string str;
  llvm::raw_string_ostream stream(str);
  for (uint64_t i = 0; i < shape.size() - 1; ++i)
    stream << dimToString(shape[i]) << ",";
  stream << dimToString(shape.back());

  return str;
}

class VerifyModelArgumentsPass
    : public impl::VerifyModelArgumentsPassBase<VerifyModelArgumentsPass> {
  using VerifyModelArgumentsPassBase::VerifyModelArgumentsPassBase;

  void runOnOperation() override {
    auto &ctx = getContext();
    OpBuilder builder(&ctx);
    ModelOp model = getOperation();
    BufferBuilderContext builderContext =
        TensorBufferDialectInterface::getAllBufferBuilders(model);
    Block *executeBlock = model.getExecuteBlock();
    ExecuteState executeState(&ctx, model);
    Location loc = model->getLoc();

    builder.setInsertionPointToStart(executeBlock);
    for (BlockArgument arg : executeState.getExtraArgs()) {
      auto tensorArg = dyn_cast<DeviceTensorType>(arg.getType());
      if (!tensorArg)
        continue;

      if (tensorArg.getNumDynamicDims() == (size_t)tensorArg.getRank())
        continue;

      MGPBufferBuilder &bufferBuilder =
          builderContext.getBufferBuilder(tensorArg.getDevice());
      TensorSpecType expectedTensorSpec = tensorArg.getTensorSpec();
      auto runtimeTensorSpec =
          bufferBuilder.tensorExtractTensorSpec(builder, loc, arg);

      Value areEqual;
      for (size_t i = 0; i < expectedTensorSpec.getRank(); ++i) {
        if (expectedTensorSpec.isDynamicDim(i))
          continue;
        auto expectedDim = mlir::index::ConstantOp::create(
            builder, loc, expectedTensorSpec.getDims()[i].value());
        auto runtimeDim =
            TensorSpecGetDimOp::create(builder, loc, runtimeTensorSpec, i);
        auto dimEqual = mlir::index::CmpOp::create(
            builder, loc, mlir::index::IndexCmpPredicate::EQ, runtimeDim,
            expectedDim);
        if (!areEqual)
          areEqual = dimEqual;
        else
          areEqual = KGEN::POP::AndOp::create(builder, loc, builder.getI1Type(),
                                              areEqual, dimEqual);
      }
      if (!areEqual)
        // All dimensions are dynamic, so nothing to check
        continue;

      std::string message;
      llvm::raw_string_ostream stream(message);
      stream << llvm::format(
          "Tensor type mismatch for argument %ld: Expected shape: [%s]",
          arg.getArgNumber() - model.getExecuteImplicitArguments().size(),
          shapeToString(expectedTensorSpec.getShape()).c_str());
      AssertOp::create(builder, loc, areEqual,
                       StringAttr::get(&ctx, stream.str()));
    }
  }
};

} // namespace
} // namespace M::MGP
