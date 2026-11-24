//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Mojo/DepGraphDialect/DepGraphTypes.h"
#include "Mojo/DepGraphDialect/DepGraphDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace DepGraph;

//===----------------------------------------------------------------------===//
// DepGraphDialect - type registration
//===----------------------------------------------------------------------===//

void DepGraphDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Mojo/DepGraphDialect/DepGraphTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "Mojo/DepGraphDialect/DepGraphTypes.cpp.inc"
