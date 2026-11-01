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

#include "GraphCompiler/Conversion/ParameterLowering.h"
#include "Support/AssertStream.h"

#include "mlir/IR/BuiltinAttributes.h"

using namespace M;
using namespace mlir;

ErrorOr<Value> M::ParameterLoweringState::lowerExpression(OpBuilder &b,
                                                          Location loc,
                                                          Region *parentRegion,
                                                          TypedAttr attr,
                                                          LowerExprFunc func) {
  if (Value lowered = lookupLowered(parentRegion, attr))
    return lowered;

  Value newValue;
  if (auto declRef = dyn_cast<KGEN::ParamDeclRefAttr>(attr)) {
    // A decl ref we haven't seen before gets a placeholder.
    newValue = PlaceholderOp::create(b, loc, declRef).getResult();
  } else if (auto opAttr = dyn_cast<KGEN::ParamOperatorAttr>(attr)) {
    ArrayRef<TypedAttr> operands = opAttr.getOperands();
    TypedAttr newOpAttr = attr;
    if (opAttr.getOpcode() == KGEN::POC::Apply) {
      // Apply is special, since its first argument is an mlir op.
      newOpAttr = operands.front();
      operands = operands.drop_front();
    }
    auto newOperandsOr = lowerOperands(b, loc, parentRegion, operands, func);
    if (newOperandsOr.isError())
      return newOperandsOr.takeError();
    auto newOperands = std::move(*newOperandsOr);
    auto newValueOr = func(*this, b, loc, parentRegion, newOpAttr, newOperands);
    if (newValueOr.isError())
      return newValueOr.takeError();
    newValue = std::move(*newValueOr);
  } else if (auto identical = dyn_cast<KGEN::ParamIdenticalAttr>(attr)) {
    auto newOperandsOr =
        lowerOperands(b, loc, parentRegion, identical.getOperands(), func);
    if (newOperandsOr.isError())
      return newOperandsOr.takeError();
    auto newValueOr =
        func(*this, b, loc, parentRegion, attr, std::move(*newOperandsOr));
    if (newValueOr.isError())
      return newValueOr.takeError();
    newValue = std::move(*newValueOr);
  } else {
    auto newValueOr = func(*this, b, loc, parentRegion, attr, ValueRange{});
    if (newValueOr.isError())
      return newValueOr.takeError();
    newValue = std::move(*newValueOr);
  }

  emplaceLowered(parentRegion, attr, newValue);
  return newValue;
}

Value M::ParameterLoweringState::lowerParamDecl(Region *parentRegion,
                                                KGEN::ParamDeclAttr declAttr,
                                                Value value) {
  auto declRef = KGEN::ParamDeclRefAttr::get(declAttr);
  if (Value lowered = lookupLowered(parentRegion, declRef)) {
    auto placeholder = lowered.getDefiningOp<PlaceholderOp>();
    // Parameter verification rejects duplicate declarations.
    ASSERT_STREAM(placeholder, << "duplicate declaration of parameter");
    eraseLowered(parentRegion, declRef);
    emplaceLowered(parentRegion, declRef, value);
    return placeholder;
  }

  emplaceLowered(parentRegion, declRef, value);
  return {};
}

ErrorOr<SmallVector<Value>> M::ParameterLoweringState::lowerOperands(
    OpBuilder &b, Location loc, Region *parentRegion,
    ArrayRef<TypedAttr> operands, LowerExprFunc func) {
  SmallVector<Value> newOperands;
  for (TypedAttr operand : operands) {
    auto newOperandOr = lowerExpression(b, loc, parentRegion, operand, func);
    if (newOperandOr.isError())
      return newOperandOr.takeError();
    newOperands.emplace_back(std::move(*newOperandOr));
  }
  return newOperands;
}
