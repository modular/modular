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

#include "font.h"
#include "Support/Compiler/MLIRDenseAttr.h"

namespace M::GlidIR {

alignas(M::kAsmResourceBlobAlignment) constexpr char fontDataArray[] = {
#embed MLIR_VIZ_FONT
};

alignas(M::kAsmResourceBlobAlignment) constexpr char emojiFontDataArray[] = {
#embed MLIR_VIZ_EMOJI_FONT
};

llvm::StringRef getFontData() { return {fontDataArray, sizeof(fontDataArray)}; }
llvm::StringRef getEmojiFontData() {
  return {emojiFontDataArray, sizeof(emojiFontDataArray)};
}

} // namespace M::GlidIR
