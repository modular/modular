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
// Attributes and their helpers for the MGP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_MGPDIALECT_MGPATTRS_H
#define GENERICML_MGPDIALECT_MGPATTRS_H

#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/MGPDialect/Dialect/MGPAttrs.h.inc"

#endif // GENERICML_MGPDIALECT_MGPATTRS_H
