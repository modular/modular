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

#include "GraphCompiler/Conversion/MOTarget/Conversion.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/RMODialect/RMODialect.h"

using namespace M;
using namespace MO;
using namespace RMO;

RMOConversionTarget::RMOConversionTarget(MLIRContext &ctx)
    : ConversionTarget(ctx) {
  addLegalDialect<RMO::RMODialect>();

  // RMO reuses the following operations for now:
  addLegalOp<
      // Reuses the same graph region
      MO::GraphOp, MO::OutputOp,
      // Constants
      MO::ConstantOp, MO::ConstantExternalOp,
      // Control flow
      MO::IfOp, MO::YieldOp, MO::WhileOp, MO::WhileConditionOp,
      // Custom ops allowed for now
      MO::CustomOp,
      // TODO (GEX-723)
      MO::ReduceGroupNormOp, MO::LayerNormOp, MO::ReduceMinAndMaxOp,
      MO::Composite::ReduceRMSNormFusedResidualAddOp,
      MO::Composite::RMSNormResidualAddOp, MO::ReduceRMSNormOp,
      // Buffer operations
      MO::ChainCreateOp,
      // Copied from MOConversionTarget
      ModuleOp, mlir::UnrealizedConversionCastOp>();
}

MOConversionTarget::MOConversionTarget(MLIRContext &ctx)
    : ConversionTarget(ctx) {
  addLegalDialect<MO::MODialect>();
  addLegalOp<ModuleOp, mlir::UnrealizedConversionCastOp>();
}
