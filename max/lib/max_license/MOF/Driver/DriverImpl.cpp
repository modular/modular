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

#include "max/max_license/MOF/Driver/DriverImpl.h"

#include "max/max_license/MOF/InitAllDialects.h"
#include "mlir/Parser/Parser.h"

using namespace M;
using namespace M::MOF;

OwningOpRef<ModuleOp> MOF::parseMODialect(M::BufferRef buffer,
                                          MLIRContext *ctx) {
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  ctx->appendDialectRegistry(registry);
  ctx->allowUnregisteredDialects(true);
  auto filePathOr = buffer->getFilePath();
  if (filePathOr)
    return mlir::parseSourceString<ModuleOp>(buffer->getBuffer(), ctx,
                                             filePathOr->string());
  else
    return mlir::parseSourceString<ModuleOp>(buffer->getBuffer(), ctx);
}
