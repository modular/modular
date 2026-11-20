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

#ifndef GENERICML_GRAPHCOMPILER_CONVERSION_MOTARGET_CONVERSION_H
#define GENERICML_GRAPHCOMPILER_CONVERSION_MOTARGET_CONVERSION_H

#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Transforms/DialectConversion.h"

namespace M::MO {

/// ConversionTarget for converting to the MO dialect.
class MOConversionTarget : public ConversionTarget {
public:
  explicit MOConversionTarget(MLIRContext &ctx);
};

} // namespace M::MO

namespace M::RMO {
/// ConversionTarget for converting to the MO dialect.
class RMOConversionTarget : public ConversionTarget {
public:
  explicit RMOConversionTarget(MLIRContext &ctx);
};
} // namespace M::RMO

#endif // GENERICML_GRAPHCOMPILER_CONVERSION_MOTARGET_CONVERSION_H
