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
// This registers MLIRToMEF as an MLIR translation for use with
// mef-translate.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-translate/Translation.h"
#include "MLRT/GraphRT/MEFDialect/MEFDialect.h"
#include "MLRT/GraphRT/MLIRToMEF.h"
#include "Support/ADT/ConcatenationTree.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/IR/BuiltinOps.h"
using namespace M;

void M::registerMLIRToMEFTranslation(DialectRegistry &otherDialects) {
  static llvm::cl::opt<bool> enableMEFMetadata(
      "enable-mef-metadata",
      llvm::cl::desc("Enable metadata sections for better MEF2MLIR"),
      llvm::cl::init(false));

  static mlir::TranslateFromMLIRRegistration toMEF(
      "export-mef", "Translate MLIR to MEF",
      [](ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        auto data = translateMLIRToMEF(module, enableMEFMetadata);
        if (data.getSize() == 0)
          return failure();
        // write all the returned data out to a stream.
        data.traverse([&](ArrayRef<uint8_t> data) {
          os.write(reinterpret_cast<const char *>(data.data()), data.size());
        });
        return success();
      },
      [&](DialectRegistry &registry) {
        registry.insert<MEF::MEFDialect, MDialect>();
        // If the caller provided any other dialects, add them.
        otherDialects.appendTo(registry);
      });
}
