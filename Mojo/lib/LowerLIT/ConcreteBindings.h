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
//
// This file contains utilities for binding concrete (non-parametric) values
// to generator parameters. This is used by LowerLIT to remove singleton
// parameters from generator signatures.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_LOWERLIT_CONCRETEBINDINGS_H
#define KGEN_LOWERLIT_CONCRETEBINDINGS_H

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENInterfaces.h"
#include "Mojo/KGENDialect/ParameterReplacer.h"

namespace M::KGEN {

//===----------------------------------------------------------------------===//
// ConcreteBindingReplacer
//===----------------------------------------------------------------------===//

/// A single-pass replacer for specializing generator types when bindings are
/// concrete values. A concrete value in this case is a TypedAttr that is not an
//  ParamIndexRef nor a ParaDeclRef. This is a simpler alternative to
/// ParameterEvaluator for cases like singleton removal in LowerLIT.
class ConcreteBindingReplacer
    : public IndexParameterReplacer<ConcreteBindingReplacer> {
public:
  ConcreteBindingReplacer(ParamsBindingArrayAttr paramBindings)
      : bindings(paramBindings) {}
  SmallVector<Type> getRemappedUnboundParamTypes(ArrayRef<Type> originalTypes);
  Type getReboundType(Type type) { return replace(type); }
  TypedAttr getReboundAttribute(TypedAttr attr) { return replace(attr); }

  /// Specialize generator metadata using this replacer's bindings.
  PogListAttr
  specializeMetadata(PogListAttr genMetadata, ArrayRef<Type> paramTypes,
                     function_ref<InFlightDiagnostic()> emitErrorFn);

private:
  Attribute tryReplace(Attribute attr, size_t depth);
  Type tryReplace(Type, size_t) { return {}; }
  friend class IndexParameterReplacer<ConcreteBindingReplacer>;

  bool isBound(size_t idx) const {
    return bindings.getBoundMask().asArrayRef()[idx];
  }

  /// Return the position of input `idx` among the inputs with the same bound
  /// state: the number of bound/unbound values depending on whether `idx`
  /// itself is bound/unbound.
  size_t getRank(size_t idx) const {
    return llvm::count(bindings.getBoundMask().asArrayRef().take_front(idx),
                       isBound(idx));
  }

  ParamsBindingArrayAttr bindings;
};

//===----------------------------------------------------------------------===//
// Free functions for specializing generators with concrete bindings
//===----------------------------------------------------------------------===//

/// Get this GeneratorType with concrete bindings, no index refs.
GeneratorType getSpecializedWithConcreteBindings(
    GeneratorType gen, ParamsBindingArrayAttr paramBindings,
    function_ref<InFlightDiagnostic()> emitErrorFn = {});

/// Get this GeneratorAttr with concrete bindings, no index refs.
GeneratorAttr getSpecializedWithConcreteBindings(
    GeneratorAttr gen, ParamsBindingArrayAttr paramBindings,
    function_ref<InFlightDiagnostic()> emitErrorFn = {});

} // namespace M::KGEN

#endif // KGEN_LOWERLIT_CONCRETEBINDINGS_H
