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
// This file declares transforms for the RMO dialect.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOTRANSFORMS_H
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOTRANSFORMS_H

#include "GraphCompiler/MODialect/MODialect.h"
#include "GraphCompiler/RMODialect/RMODialect.h"
#include "mlir/Pass/Pass.h"

using namespace M::MO;

namespace mlir {
class Pass;
} // namespace mlir

namespace M::RMO {

#define GEN_PASS_DECL
#include "GraphCompiler/RMODialect/Transforms/RMOTransforms.h.inc"

#define GEN_PASS_REGISTRATION
#include "GraphCompiler/RMODialect/Transforms/RMOTransforms.h.inc"

} // namespace M::RMO

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOTRANSFORMS_H
