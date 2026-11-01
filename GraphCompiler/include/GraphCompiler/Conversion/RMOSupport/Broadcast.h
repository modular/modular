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
//
// Broadcast helpers for frontends converting to RMO.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MODialect/MOTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_BROADCAST_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_BROADCAST_H

namespace M::RMO {

/// Reshape the `from` tensor using an intermediate shape, then attempt to
/// broadcast to the shape of the `to` tensor.
/// Does not currently support reshaping to dynamic or parametric shapes.
MO::TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                      MO::TensorValue from,
                                      MO::TensorType reshapeType,
                                      MO::TensorValue to);

/// Reshape the `from` tensor using an intermediate shape, then attempt to
/// broadcast to the shape of the `to` tensor.
/// Does not currently support reshaping to dynamic or parametric shapes.
MO::TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                      MO::TensorValue from,
                                      ArrayRef<int64_t> reshapeShape,
                                      MO::TensorValue to);

/// Reshape the `from` tensor using an intermediate shape (which can be
/// dynamic), then attempt to broadcast to the shape of the `to` tensor.
MO::TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                      MO::TensorValue from,
                                      MO::TensorType reshapeType,
                                      MO::TensorValue reshapeShape,
                                      MO::TensorValue to);

/// Reshape the `from` tensor using an intermediate shape (which can be
/// dynamic), then attempt to broadcast to the shape of the `to` tensor.
MO::TensorValue reshapeAndBroadcastTo(OpBuilder &builder, Location loc,
                                      MO::TensorValue from,
                                      ArrayRef<int64_t> reshapeDims,
                                      MO::TensorValue reshapeShape,
                                      MO::TensorValue to);

/// Broadcast tensor `x` to the shape of `s`. Works with static shapes only.
MO::TensorValue broadcastToStaticShape(OpBuilder &builder, Location loc,
                                       MO::TensorValue x, MO::ShapeAttr s);

/// Broadcast tensor `x` and `y` to each other, such that they have the same
/// shape after broadcasting. The method broadcasts statically, if the input
/// shapes are static, or dynamically otherwise.
/// All inputs must have a static rank and compatible shapes.
std::pair<MO::TensorValue, MO::TensorValue>
reifyImplicitBroadcasting(OpBuilder &builder, Location loc, MO::TensorValue x,
                          MO::TensorValue y);

/// Broadcast tensor `x`, `y` and `z` to each other, such that they have the
/// same shape after broadcasting. The method broadcasts statically, if the
/// input shapes are static, or dynamically otherwise.
/// All inputs must have a static rank and compatible shapes.
std::tuple<MO::TensorValue, MO::TensorValue, MO::TensorValue>
reifyImplicitBroadcasting(OpBuilder &builder, Location loc, MO::TensorValue x,
                          MO::TensorValue y, MO::TensorValue z);

/// Broadcast tensor `x` to `y`. Requires that `x` can be broadcast
/// unidirectionally to `y`.
MO::TensorValue broadcastTo(OpBuilder &builder, Location loc, MO::TensorValue x,
                            MO::TensorValue y);

/// Broadcasts a float `scalar` to `toVal`'s shape and element type.
MO::TensorValue broadcastFloatTo(OpBuilder &builder, Location loc,
                                 const APFloat &scalar, MO::TensorValue toVal);

/// Given a vector of tensors, return an analogous vector where each tensor is
/// broadcast to a common shape. Does not simplify shapes symbolically.
SmallVector<MO::TensorValue>
broadcastToCompatibleShape(SmallVector<MO::TensorValue> &values,
                           OpBuilder &rewriter);

} // namespace M::RMO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOSUPPORT_BROADCAST_H
