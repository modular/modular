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

#include "Mojo/DepGraphDialect/DepGraphDialect.h"
#include "Mojo/DepGraphDialect/DepGraphOps.h"
#include "Mojo/DepGraphDialect/DepGraphTypes.h"

using namespace M;
using namespace KGEN;
using namespace DepGraph;

//===----------------------------------------------------------------------===//
// DepGraphDialect
//===----------------------------------------------------------------------===//

void DepGraphDialect::initialize() {
  registerTypes();
  registerOperations();
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Mojo/DepGraphDialect/DepGraphDialect.cpp.inc"
