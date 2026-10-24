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
// This file declares the operation classes for the RMO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_H
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_H

#include "GraphCompiler/MODialect/MOAttrs.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/RMODialect/RMOInterfaces.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "mlir/IR/OpImplementation.h"

using M::KGEN::ParamDeclAttr;
using M::MO::parseTransferResult;
using M::MO::printTransferResult;
using M::MOSH::ShapeAttr;

namespace M::RMO {

/// Combines two types describing the same shape into the most refined type.
MO::TensorType refineShape(MO::TensorType ty1, MO::TensorType ty2);
MO::TensorType refineShapeWithDType(MO::TensorType ty1, MO::TensorType ty2,
                                    DType dtype);

/// inferMoTypesImpl for scatter-reduction ops whose axis is a scalar tensor
/// operand (`scatter.add/max/min/mul`).
void inferMOTypesScatterImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue updates, TensorValue indices, TensorValue axis,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls);

/// inferMoTypesImpl for `scatter`, whose axis is an attribute, not an operand.
void inferMOTypesScatterImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue updates, TensorValue indices,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls);

/// inferMoTypesImpl for all elementwise-unary-like ops
void inferMOTypesElemwiseUnaryImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls);

/// inferMoTypesImpl for all unary-like ops where output has an explicit type.
void inferMOTypesUnaryExplicitOutputTypeImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls);

/// inferMoTypesImpl for TopK-like ops
void inferMOTypesTopKImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input, TensorValue k,
    TensorValue axis, TensorValue sorted, std::optional<ValueRange> rmoResults,
    SmallVectorImpl<Type> &moOperandTypes, SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls);

/// inferMoTypesImpl for all scatternd-like ops
void inferMOTypesScatterNdImpl(
    MLIRContext *context, MO::GraphOp graphOp, TensorValue input,
    TensorValue updates, TensorValue indices,
    std::optional<ValueRange> rmoResults, SmallVectorImpl<Type> &moOperandTypes,
    SmallVectorImpl<Type> &moResultTypes,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moInputParamDecls,
    SmallVectorImpl<SmallVector<KGEN::ParamDeclAttr>> &moOutputParamDecls);

/// Returns the promotion dtype of `x` and `y`:
///   (max(category(x), category(y)), max(bitwidth(x), bitwidth(y)))
/// with categories ordered `bool < unsigned int < signed int < float`. Errors
/// if either input may hold a value the result cannot represent, or if the
/// dtypes share category and bitwidth but differ in representation.
ErrorOr<DType> promoteDtype(DType x, DType y);

/// Returns the device attached to the type, if any.
std::optional<M::DeviceRefAttr> getDeviceIfAvailable(Type t);
std::optional<M::DeviceRefAttr> getDeviceIfAvailable(Value v);

/// As `getDeviceIfAvailable`, but the device must exist.
M::DeviceRefAttr getDevice(Type t);
M::DeviceRefAttr getDevice(Value v);

} // namespace M::RMO
//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GraphCompiler/RMODialect/RMO.h.inc"

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOOPS_H
