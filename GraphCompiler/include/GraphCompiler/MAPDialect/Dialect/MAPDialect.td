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
// This declares the MAP dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPDIALECT_TD
#define GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// MAPDialect
//===----------------------------------------------------------------------===//

def MAPDialect : Dialect {
  let name = "map";

  let summary = "Modular Access Primitives dialect.";
  let description = [{
    MAP (Modular Access Primitives) expresses executable loop nests / kernels
    that load and store tensors and perform computation. It has two op
    families: `map.iter.opaque` wraps a hand-written Mojo kernel with a fusible
    store epilogue (`map.iter.epilogue` / `map.call.opaque`), while `map.iter.elementwise`
    expresses the loop body explicitly as scalar `map.load` / `map.store` /
    `map.call.elementwise` ops per index (`map.index<N>`).
  }];

  let cppNamespace = "::M::MAP";
  // MOGG owns attributes MAP ops carry, such as `dispatch` and `fusionInfo`.
  let dependentDialects = [
    "::M::MO::MODialect",
    "::M::MOGG::MOGGDialect",
  ];

  let hasCanonicalizer = 1;
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // GENERICML_GRAPHCOMPILER_MAPDIALECT_MAPDIALECT_TD
