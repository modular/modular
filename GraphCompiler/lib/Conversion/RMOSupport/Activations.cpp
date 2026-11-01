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

#include "GraphCompiler/Conversion/RMOSupport/Activations.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/RMODialect/RMOOps.h"

using M::MO::TensorValue;

namespace M::RMO {

TensorValue convertPRelu(OpBuilder &builder, Location loc, TensorValue x,
                         TensorValue alpha) {
  auto zero = MO::ConstantOp::create(builder, loc, 0.f);

  TensorValue xValue;
  bool cast = false;
  if (x.getType().getDtype() != zero.getType().getDtype()) {
    cast = true;
    xValue = RMO::MOCastOp::create(builder, loc, x, zero.getType().getDtype());
  } else {
    xValue = x;
  }

  auto broadcastZero = RMO::BroadcastLikeOp::create(builder, loc, zero, xValue);
  auto broadcastAlpha =
      RMO::BroadcastLikeOp::create(builder, loc, alpha, xValue);
  auto min = RMO::MOMinOp::create(builder, loc, xValue.getType(), xValue,
                                  broadcastZero);
  auto minXAlpha = RMO::MOMulOp::create(builder, loc, broadcastAlpha.getType(),
                                        broadcastAlpha, min);
  auto max = RMO::MOMaxOp::create(builder, loc, xValue.getType(), xValue,
                                  broadcastZero);

  auto result =
      RMO::MOAddOp::create(builder, loc, max.getType(), max, minXAlpha);
  if (cast) {
    return RMO::MOCastOp::create(builder, loc, result, x.getType().getDtype());
  } else {
    return result;
  }
}

TensorValue convertRelu(OpBuilder &builder, Location loc, TensorValue x) {
  auto zero = MO::ConstantOp::create(builder, loc, 0.f);
  auto xType = x.getType().getDtype();
  bool cast = false;

  TensorValue xValue;
  if (x.getType().getDtype() != zero.getType().getDtype()) {
    // Cast X to fp32, then cast back
    xValue = RMO::MOCastOp::create(builder, loc, x, zero.getType().getDtype());
    cast = true;
  } else {
    xValue = x;
  }

  auto broadcastZero = RMO::BroadcastLikeOp::create(builder, loc, zero, xValue);
  auto max = RMO::MOMaxOp::create(builder, loc, xValue.getType(), xValue,
                                  broadcastZero);
  if (cast) {
    return RMO::MOCastOp::create(builder, loc, max, xType);
  } else {
    return max;
  }
}

TensorValue convertRelu6(OpBuilder &builder, Location loc, TensorValue x) {
  auto six = MO::ConstantOp::create(builder, loc, 6.f);
  auto zero = MO::ConstantOp::create(builder, loc, 0.f);
  TensorValue sixValue, zeroValue;
  if (x.getType().getDtype() != six.getType().getDtype()) {
    sixValue = RMO::MOCastOp::create(builder, loc, six, x.getType().getDtype());
    zeroValue =
        RMO::MOCastOp::create(builder, loc, zero, x.getType().getDtype());
  } else {
    sixValue = six;
    zeroValue = zero;
  }
  auto broadcastSix = RMO::BroadcastLikeOp::create(builder, loc, sixValue, x);
  auto broadcastZero = RMO::BroadcastLikeOp::create(builder, loc, zeroValue, x);
  auto min = MinOp::create(builder, loc, x, broadcastSix);
  return MaxOp::create(builder, loc, broadcastZero, min);
}

} // namespace M::RMO
