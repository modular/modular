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

#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPAttrs.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/MODialect/MOUtils.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace MAP;

//===----------------------------------------------------------------------===//
// MAPDialect
//===----------------------------------------------------------------------===//

void MAPDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "GraphCompiler/MAPDialect/Dialect/MAP.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "GraphCompiler/MAPDialect/Dialect/MAPTypes.cpp.inc"
      >();
  registerAttributes();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MAPDialect/Dialect/MAPDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "GraphCompiler/MAPDialect/Dialect/MAPTypes.cpp.inc"
