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
// This file registers all the dialects in the GraphRT library.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_INITALLDIALECTS_H
#define MLRT_GRAPHRT_INITALLDIALECTS_H

#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M {
// Add all the MLIR dialects to the provided registry.
inline void registerAllGraphRTDialects(DialectRegistry &registry) {
  registry.insert<MEF::MEFDialect, GRT::GRTDialect>();
}

} // namespace M

#endif // MLRT_GRAPHRT_INITALLDIALECTS_H
