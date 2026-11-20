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
// This file declares the operation classes for the GRT dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_GRTDIALECT_GRTOPS_H
#define MLRT_GRAPHRT_GRTDIALECT_GRTOPS_H

#include "MLRT/GraphRT/GRTDialect/GRTChainInterface.h"
#include "MLRT/GraphRT/GRTDialect/GRTControlFlowInterface.h"
#include "MLRT/GraphRT/GRTDialect/GRTDialect.h"
#include "MLRT/GraphRT/MEFDialect/MEFNonStrictInterface.h"
#include "MLRT/GraphRT/MEFDialect/MEFOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "MLRT/GraphRT/GRTDialect/GRT.h.inc"

#endif // MLRT_GRAPHRT_GRTDIALECT_GRTOPS_H
