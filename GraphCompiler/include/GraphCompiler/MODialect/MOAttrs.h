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
// This file declares attributes for the MO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_H
#define GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_H

// `MO::LayoutAttr` uses `mlir::StringAttr`
#include "GraphCompiler/MODialect/MOEnums.h"
#include "GraphCompiler/MODialect/MOTypes.h"
#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Support/ML/DType.h"
#include "mlir/IR/Attributes.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Declarations
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "GraphCompiler/MODialect/MOAttrs.h.inc"
#include "GraphCompiler/MODialect/MOEnumAttrs.h.inc"

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MOATTRS_H
