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
// This defines integer operations in GRT dialect.
//
//===----------------------------------------------------------------------===//

include "MLRT/GraphRT/GRTDialect/GRTOpsBase.td"
include "MLRT/GraphRT/MEFDialect/MEFTypes.td"
//===----------------------------------------------------------------------===//
// ConstantI1Op
//===----------------------------------------------------------------------===//

def ConstantI1Op : GRTOp<"constant.i1"> {
  let summary = "constant 1-bit integer";
  let description = [{
    The "graphrt.constant.i1" operation produces a 1-bit result in a bool
    AsyncValue.
  }];

  let arguments = (ins I1Attr:$value);
  let results = (outs I1:$result);
  let assemblyFormat = "$value attr-dict";
}

//===----------------------------------------------------------------------===//
// ConstantSI32Op
//===----------------------------------------------------------------------===//

def ConstantSI32Op : GRTOp<"constant.si32"> {
  let summary = "constant 32-bit signed integer";
  let description = [{
    The "graphrt.constant.si32" operation produces a 32-bit result in an int32_t
    AsyncValue.
  }];

  let arguments = (ins SI32Attr:$value);
  let results = (outs SI32:$result);
  let assemblyFormat = "$value attr-dict";
}

//===----------------------------------------------------------------------===//
// AddSI32Op
//===----------------------------------------------------------------------===//

def AddSI32Op : GRTOp<"add.si32"> {
  let summary = "Add two int32_t values";

  let arguments = (ins SI32:$lhs, SI32:$rhs);
  let results = (outs SI32:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

//===----------------------------------------------------------------------===//
// LessEqualSI32Op
//===----------------------------------------------------------------------===//

def LessEqualSI32Op : GRTOp<"lessequal.si32"> {
  let summary = "Compare two int32_t values for `<=` and return a bool";

  let arguments = (ins SI32:$lhs, SI32:$rhs);
  let results = (outs I1:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

//===----------------------------------------------------------------------===//
// PrintSI32Op
//===----------------------------------------------------------------------===//

def PrintSI32Op : GRTOp<"print.si32"> {
  let summary = "Print an int32_t value";

  let arguments = (ins SI32:$value, ChainType:$chain);
  let results = (outs ChainType:$result);
  let assemblyFormat = "$value `,` $chain attr-dict";
}
