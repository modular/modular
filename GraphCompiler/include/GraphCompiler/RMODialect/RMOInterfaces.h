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
// This file declares the interface classes for the RMO dialect.
//
//===----------------------------------------------------------------------===//
#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMOINTERFACES_H
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMOINTERFACES_H

#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/MOTypes.h"

using M::KGEN::ParamDefValue;
using M::MO::TensorValue;

namespace M::RMO {
// Forward declare things used in RMOInterfaces.h.inc
class RMOOp;
LogicalResult verifyRMOOp(RMOOp op);
LogicalResult verifySameDTypeOperandsAndResults(RMOOp op);
} // namespace M::RMO

#include "GraphCompiler/RMODialect/RMOInterfaces.h.inc"

#endif // GENERICML_GRAPHCOMPILER_RMODIALECT_RMOINTERFACES_H
