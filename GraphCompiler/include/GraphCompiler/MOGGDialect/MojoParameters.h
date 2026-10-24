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

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJO_PARAMETERS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJO_PARAMETERS_H

#include "llvm/ADT/StringRef.h"

namespace M::MOGG {

/// Matmul and BatchMatmul op
static constexpr llvm::StringLiteral PARAMETER_MATMUL_TRANSPOSED_WEIGHTS =
    "transpose_b";
static constexpr llvm::StringLiteral PARAMETER_MATMUL_PACKED_WEIGHTS =
    "packed_b";

/// Conv op
static constexpr llvm::StringLiteral PARAMETER_CONV_PACKED_FILTER =
    "filter_packed";
static constexpr llvm::StringLiteral PARAMETER_CONV_INPUT_LAYOUT =
    "input_layout";
static constexpr llvm::StringLiteral PARAMETER_CONV_FILTER_LAYOUT =
    "filter_layout";
static constexpr llvm::StringLiteral PARAMETER_CONV_STATIC_STRIDES =
    "static_strides";
static constexpr llvm::StringLiteral PARAMETER_CONV_STATIC_DILATIONS =
    "static_dilations";
static constexpr llvm::StringLiteral PARAMETER_CONV_STATIC_PADDING =
    "static_padding";
static constexpr llvm::StringLiteral PARAMETER_CONV_INPUT_RANK = "input_rank";

/// ConvTranspose op
static constexpr llvm::StringLiteral PARAMETER_CONVTRANSPOSE_INPUT_RANK =
    "input_rank";

/// Gather ND op
static constexpr llvm::StringLiteral PARAMETER_GATHER_ND_BATCH_DIMENSIONS =
    "batch_dims";

/// Resize op
static constexpr llvm::StringLiteral
    PARAMETER_RESIZE_COORDINATE_TRANSFORM_MODE = "coordinate_transform_mode";
static constexpr llvm::StringLiteral PARAMETER_RESIZE_ROUNDING_MODE =
    "round_mode";
static constexpr llvm::StringLiteral PARAMETER_RESIZE_ANTIALIASING =
    "antialias";

/// Cum Sum op
static constexpr llvm::StringLiteral PARAMETER_CUM_SUM_EXCLUSIVE = "exclusive";

/// AvgPool op
static constexpr llvm::StringLiteral PARAMETER_AVG_POOL_COUNT_BOUNDARY =
    "count_boundary";

/// ROI Align op
static constexpr llvm::StringLiteral PARAMETER_ROI_ALIGNMENT = "aligned";
static constexpr llvm::StringLiteral PARAMETER_ROI_ALIGNMENT_MODE = "mode";

// TransposeOp
static constexpr llvm::StringLiteral PARAMETER_TRANSPOSE_STATIC_PERM =
    "static_permutations";

// SliceOp
static constexpr llvm::StringLiteral PARAMETER_SLICE_STATIC_STARTS =
    "static_starts";
static constexpr llvm::StringLiteral PARAMETER_SLICE_STATIC_STEPS =
    "static_steps";

} // namespace M::MOGG

#endif
