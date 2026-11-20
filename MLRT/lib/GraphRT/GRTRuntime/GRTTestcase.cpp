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

#include "MLRT/GraphRT/GRTTestcase.h"
#include "MLRT/GraphRT/MEFExecutor/PrimitiveRegistry.h"

using namespace M;

GRTTestcase::GRTTestcase(const PrimitiveDescriptor &op)
    : name(*op.getAttr<StringRef>(0)), body(op.getInvokable(0)) {
  assert(op.getNumInvokables() == 1 &&
         "cannot construct a MEFInvokablePrimitive from a TopLevelOp that has "
         "more than one primitive");
}

StringLiteral GRTTestcase::getEntityName() { return "grt.testcase"; }
