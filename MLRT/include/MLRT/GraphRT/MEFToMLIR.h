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
// This file declares the interface to the MEFToMLIR library.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_MEFTOMLIR_H
#define MLRT_GRAPHRT_MEFTOMLIR_H

#include "Support/LLVMCompilerForwardDecls.h"

namespace M {

/// Read the specified a MEF file from the specified binary buffer, and
/// returning the created module on success.  On failure, report an error at the
/// specified location and return null.
ModuleOp translateMEFToMLIR(ArrayRef<uint8_t> buffer, Location loc);

// Register this translation for GraphRTTranslate.
void registerMEFToMLIRTranslation();

} // namespace M

#endif // MLRT_GRAPHRT_MEFTOMLIR_H
