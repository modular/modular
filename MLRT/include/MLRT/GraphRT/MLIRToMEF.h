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
// This file declares the interface to the MLIRToMEF library.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_MLIRTOMEF_H
#define MLRT_GRAPHRT_MLIRTOMEF_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace M {
class ConcatenationTree;

/// This function converts the specified MLIR module containing a executor
/// compatible program to the BinaryExecutableFormat (MEF) format, which is the
/// binary format that the executor takes.  When `enableMEFMetadata` is true
/// this adds metadata sections that allow MEF2MLIR to produce full fidelity
/// input.
///
/// On error, this emits the error message through the MLIR error handler, and
/// returns an empty MEFBuffer.
ConcatenationTree translateMLIRToMEF(ModuleOp module,
                                     bool enableMEFMetadata = false);

// Register this translation for GraphRTTranslate.
void registerMLIRToMEFTranslation(DialectRegistry &otherDialects);

} // namespace M

#endif // MLRT_GRAPHRT_MLIRTOMEF_H
