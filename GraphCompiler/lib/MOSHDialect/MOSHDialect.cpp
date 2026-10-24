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

#include "GraphCompiler/MOSHDialect/MOSHDialect.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "GraphCompiler/MOSHDialect/MOSHOps.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Support/MDialect/MDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"

using namespace M;
using namespace MOSH;

//===----------------------------------------------------------------------===//
// MOSHDialect
//===----------------------------------------------------------------------===//

void MOSHDialect::initialize() {
  registerAttributes();
  registerTypes();
  registerOperations();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOSHDialect/MOSHDialect.cpp.inc"
