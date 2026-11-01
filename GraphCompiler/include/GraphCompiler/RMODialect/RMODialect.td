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
// This declares the RMO dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_RMODIALECT_RMODIALECT_TD
#define GENERICML_GRAPHCOMPILER_RMODIALECT_RMODIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// RMODialect
//===----------------------------------------------------------------------===//

def RMODialect : Dialect {
  let name = "rmo";

  let summary = "Relaxed Modular Operations dialect.";
  let description = [{
    RMO (Relaxed Modular Operations) represents a graph of MO operations with
    relaxed type constraints, and lowers to MO, where operand shapes are
    strictly enforced.

    MO is strongly typed and each op does one thing; for example `mo.add` does
    not broadcast, and its operand and result types must provably match. This
    makes MO easy to optimize but hard to target: frontends need shape
    inference to satisfy the verifiers (not always possible with data-dependent
    shapes), and framework ops like broadcasting `add` or rank-dependent
    `matmul` lower to chains of strictly verified MO ops.

    RMO handles MO's strict typing by inserting rebinds and runtime checks. As
    a non-optimizing layer it can also have more dynamic semantics, such as a
    broadcasting add.

    Every MO op has an RMO analogue with the same semantics but a relaxed
    verifier: `rmo.mo.add` does not broadcast, like `mo.add`, but accepts
    nearly any shape. Dynamic ops such as a broadcasting add are built as
    compositions of these analogues.
  }];

  let cppNamespace = "::M::RMO";
  let dependentDialects = [
    "MO::MODialect"
  ];

  let extraClassDeclaration = [{
  private:
    void registerAttributes();
    void registerOperations();
  }];
}

#endif // GENERICML_GRAPHCOMPILER_MODIALECT_MODIALECT_TD
