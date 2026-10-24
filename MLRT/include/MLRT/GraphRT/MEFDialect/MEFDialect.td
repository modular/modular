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
// This contains the MEFDialect definition to be included in other files.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_MEFDIALECT_MEFDIALECT_TD
#define MLRT_GRAPHRT_MEFDIALECT_MEFDIALECT_TD

include "mlir/IR/DialectBase.td"

def MEFDialect : Dialect {
  let name = "mef";

  let summary = "The 'mef' dialect for the ModularExecutorFormat";
  let description = [{
    This dialect defines the `mef` dialect, which captures operations core to
    the GraphRT "MEF" executor.
  }];

  let cppNamespace = "::M::MEF";

  let useDefaultTypePrinterParser = 1;
}

#endif // MLRT_GRAPHRT_MEFDIALECT_MEFDIALECT_TD
