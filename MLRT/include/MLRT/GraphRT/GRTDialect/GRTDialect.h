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
// This file defines an GraphRT MLIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLRT_GRAPHRT_GRTDIALECT_GRTDIALECT_H
#define MLRT_GRAPHRT_GRTDIALECT_GRTDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

// Pull in the dialect definition.
#include "MLRT/GraphRT/GRTDialect/GRTDialect.h.inc"

#endif // MLRT_GRAPHRT_GRTDIALECT_GRTDIALECT_H
