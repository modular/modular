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

#pragma once

#include "GraphCompiler/MODialect/MOOps.h"
#include "mlir/IR/PatternMatch.h"

namespace M::MO {

class DeviceInfoMapping;

/// Maps `(inputType, kgenParams)` to the packed-weight `TensorType` from a
/// matmul packing shape function, so each `MatmulConstBPattern` in a pass run
/// lowers it at most once per key.
class PackParamCache {
public:
  void put(MO::TensorType inputTy, mlir::DictionaryAttr kgenParams,
           MO::TensorType outputTy);

  /// Returns the cached packed weight type, or {} on miss.
  MO::TensorType get(MO::TensorType inputTy, mlir::DictionaryAttr kgenParams);

private:
  llvm::DenseMap<MO::TensorType,
                 llvm::DenseMap<mlir::DictionaryAttr, MO::TensorType>>
      cache;
};

/// Options for MultiHeadAttention pattern matching.
struct MHAOptions {
  bool enable = false;
  bool allowUnsafeBCast = false;
  bool allowInternalReshape = false;
  bool debug = false;
};

/// Fuses `mo.allreduce.sum` + add + rms_norm + optional quantize_fp8 into a
/// single op; no-op if the chain does not match.
void applyFuseAllReduceAddRMSNormQuantFP8(MO::AllReduceSumOp allReduceOp,
                                          M::DeviceRefAttr targetDeviceRef);

/// Cross-boundary fuser for `mo.parallel { mo.bundled.allreduce.sum }`
/// followed by per-device add + rms_norm + quantize_fp8 chains outside
/// the parallel.
void applyFuseParallelBundledAllReduceAddRMSNormQuantFP8(
    MO::BundledAllReduceSumOp bundledAllReduceOp);

/// Adds the main fusion rewrite patterns (LayerNorm, RMSNorm fusions,
/// Matmul folds, RoPE, Conv-const, attention, etc.).
void populateFusionPatterns(mlir::RewritePatternSet &patterns,
                            PackParamCache &cache,
                            M::DeviceRefAttr targetDeviceRef,
                            const MHAOptions &mhaOpts);

/// Adds `MatmulAddPattern`, which the driver pass enables only on SM100-class
/// GPUs.
void populateFusionMatmulAddPatterns(const DeviceInfoMapping &deviceInfoMapping,
                                     mlir::RewritePatternSet &patterns);

/// Adds `MegaFFNNvfp4Pattern`, which fuses the two-op NVFP4 MoE FFN composite
/// chain into `mo.composite.mega_ffn_nvfp4`. It gates itself to SM100-class
/// GPUs; the driver pass also requires the `mega-ffn-enable` option.
void populateFusionMegaFFNPatterns(const DeviceInfoMapping &deviceInfoMapping,
                                   mlir::RewritePatternSet &patterns);

} // namespace M::MO
