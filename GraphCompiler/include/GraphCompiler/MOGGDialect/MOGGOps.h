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
// This file declares the operation classes for the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_H

#include "GraphCompiler/Interfaces/EmitsMojoInterface.h"
#include "GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGEnums.h"
#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h"
#include "GraphCompiler/MOGGDialect/MOGGTypes.h"
#include "Mojo/KGENDialect/KGENParameters.h"
#include "Mojo/LITDialect/LITTypes.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Needed because ODS-generated build methods for mogg.kernel.decl use the
// unqualified name.
using M::KGEN::LIT::FnTypeGeneratorType;

namespace M::MOGG {
template <typename ConcreteType>
class CallLike : public mlir::OpTrait::TraitBase<ConcreteType, CallLike> {};
} // namespace M::MOGG

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "GraphCompiler/MOGGDialect/MOGG.h.inc"

namespace M::MOGG {

/// Returns `__<ops>`, a stub name built from the ops `fusionInfo` fuses. The
/// caller inserts the stub into a symbol table for a unique name.
StringAttr getFuncName(FusionInfoAttr fusionInfo);

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOGGOPS_H
