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

#ifndef GENERICML_MGPDIALECT_MGPDIALECT_TD
#define GENERICML_MGPDIALECT_MGPDIALECT_TD

include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// MGPDialect
//===----------------------------------------------------------------------===//

def MGPDialect : Dialect {
  let name = "mgp";

  let summary = "Buffers, contexts and other framework/kernel independent "
                "glue.";
  let description = [{
    The MGP dialect is the 'glue' binding framework-specific ops (eg `mo.add`)
    and KGENned kernels into a single executable model.

    MGP provides no op implementations and translates directly to MEF (see
    MLIRToMEF.cpp), but supports source-to-source optimizations: hoisting
    allocations and avoiding unnecessary conversions, shape calculations and
    tensor wrappers. Framework-specific dialects may depend on MGP, not vice
    versa. Ops may be 'value semantic' (take and return values) or 'buffer
    semantic' (take buffers, some pre-allocated to hold outputs).

    MGP provides:
    1. `mgp.model`, holding the model's 'init' and 'execute' invokables and a
       framework name telling the runtime which framework-specific API
       (generally an `M::Model` subclass) to wrap around them.
    2. `!mgp.context`, backed by `M::MGP::Context`, holding any number of
       `M::MGP::StateSlot`s. Returned by 'init' and reused by each 'execute',
       it gives ops a place to cache op-specific values. A well-known slot can
       hold a framework-specific context, giving all ops access to global
       state.
    3. `!mgp.buffer`, backed by `M::MLRT::BufferRef`, a (possibly owned) array
       of uninterpreted bytes (no rank, dims, dtype or endianness). This
       supports non-tensor types and packed representations, and is the
       least-common denominator between ops using different tensor wrappers.
       Buffers may be cached in state slots for reuse as temporaries and
       outputs.
    4. `!mgp.tensor`, a statically-ranked tensor with shape, dtype and buffer,
       for framework-independent use. It is created from an `!mgp.tensor_spec`
       (backed by `M::TensorSpec`, buildable from dynamic index dims) and a
       buffer, giving baseline dynamic shape support.

    MGP does not include primitives/kernels (such as `MGP.add`); a supporting
    dialect and runtime must supply them.
  }];

  let cppNamespace = "::M::MGP";
  let dependentDialects = [
    "mlir::index::IndexDialect",
    "::M::MDialect",
    "::M::MEF::MEFDialect",
  ];

  let extraClassDeclaration = [{
   private:
    void registerTypes();
    void registerAttributes();
    void registerOperations();
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasConstantMaterializer = 1;
}

#endif // GENERICML_MGPDIALECT_MGPDIALECT_TD
