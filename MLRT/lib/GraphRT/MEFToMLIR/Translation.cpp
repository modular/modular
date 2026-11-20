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
// This registers MEFToMLIR as an MLIR translation for use with mlir-translate
// based tools.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-translate/Translation.h"
#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
#include "MLRT/GraphRT/MEFToMLIR.h"
#include "Support/MDialect/MDialect.h"
#include "Support/ML/SizeUtils.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/SourceMgr.h"
using namespace M;

void M::registerMEFToMLIRTranslation() {
  static mlir::TranslateToMLIRRegistration toMLIR(
      "import-mef", "Import a MEF file and translate to MLIR",
      [](llvm::SourceMgr &sourceMgr, MLIRContext *ctx) {
        ctx->loadDialect<mlir::quant::QuantDialect>();
        ctx->loadDialect<MDialect>();
        const llvm::MemoryBuffer *input =
            sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
        Location loc =
            FileLineColLoc::get(ctx, input->getBufferIdentifier(), 0, 0);
        auto contents = ArrayRef<uint8_t>(
            reinterpret_cast<const uint8_t *>(input->getBufferStart()),
            input->getBufferSize());
        return translateMEFToMLIR(contents, loc).getOperation();
      },
      [](DialectRegistry &) {}, maxAttributeAlignment());
}
