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

#ifndef GLIDIR_FONT_H
#define GLIDIR_FONT_H

#include "Support/SymbolExport.h"
#include "llvm/ADT/StringRef.h"

namespace M::GlidIR {

MODULAR_CXX_EXPORT llvm::StringRef getFontData();
MODULAR_CXX_EXPORT llvm::StringRef getEmojiFontData();

} // namespace M::GlidIR

#endif // GLIDIR_FONT_H
