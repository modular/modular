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

#ifndef GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLDPASS_H
#define GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLDPASS_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"

#include <optional>
#include <string>

namespace M::MO {

/// Creates a constant folding pass, which registers foreign buffer interfaces
/// in a namespace prefixed by `compiledFrameworkLabel`.
///
/// Only `cacheDir` and `kernelLibraryPaths` affect the correctness of the
/// constant-fold sub-compilation, so no other options are passed.
std::unique_ptr<mlir::Pass>
createConstantFoldPass(std::string_view compiledFrameworkLabel,
                       std::optional<std::string> cacheDir,
                       llvm::ArrayRef<std::string> kernelLibraryPaths);

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_CONSTANTFOLDPASS_H
