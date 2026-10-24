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
// This file declares types for the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPTYPES_H
#define GENERICML_MGPDIALECT_MGPTYPES_H

#include "MLRT/GraphRT/MEFDialect/MEFTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LLVMForwardDecls.h"
#include "Support/MDialect/MAttrs.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Types.h"

#include "GraphCompiler/MGPDialect/Dialect/MGPDialect.h"

namespace M {
class DType;
} // namespace M

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MGPDialect/Dialect/MGPTypeInterfaces.h.inc"

#define GET_TYPEDEF_CLASSES
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h.inc"

namespace M::MGP {

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

/// Parses a compact tensor type description.
ParseResult parseTensorSpec(AsmParser &p, TensorSpecType &tensorSpec);

/// Prints a compact tensor type description.
void printTensorSpec(AsmPrinter &p, TensorSpecType tensorSpec);

/// Parses the optional `, <device>`, `, mut`, and `, stable` tail of a device
/// tensor type.
ParseResult parseDeviceTensorTail(AsmParser &p, M::DeviceRefAttr &device,
                                  bool &mut, bool &stable);

/// Prints the optional `, <device>`, `, mut`, and `, stable` tail of a device
/// tensor type.
void printDeviceTensorTail(AsmPrinter &p, M::DeviceRefAttr device, bool mut,
                           bool stable);

} // namespace M::MGP

#endif // GENERICML_MGPDIALECT_MGPTYPES_H
