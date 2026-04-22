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

#ifndef MAX_C_WEIGHTS_H
#define MAX_C_WEIGHTS_H

#include "max/c/symbol_export.h"
#include "max/c/types.h"

/// Creates a weights registry from parallel arrays of weight names and data
/// pointers.
///
/// The weights registry maps weight names to their backing data. It is used
/// with `M_initModel()` to provide weight data for models that use external
/// weights (via `constant_external` in the graph).
///
/// The data pointers are **borrowed, not copied**. You must keep the backing
/// memory alive for the lifetime of the weights registry.
///
/// @param names An array of null-terminated weight name strings.
/// @param data An array of pointers to weight data buffers. Each entry
/// corresponds to the weight name at the same index in `names`.
/// @param numWeights The number of entries in the `names` and `data` arrays.
/// @param status The status object for reporting errors.
///
/// @returns A pointer to the weights registry. You are responsible for the
/// memory associated with the pointer returned. The memory can be deallocated
/// by calling `M_freeWeightsRegistry()`. Returns `NULL` if creation fails,
/// with an error message in `status`.
MODULAR_API_EXPORT M_WeightsRegistry *M_newWeightsRegistry(const char **names,
                                                           const void **data,
                                                           size_t numWeights,
                                                           M_Status *status);

/// Deallocates the memory for the weights registry. No-op if
/// `weightsRegistry` is `NULL`.
///
/// @param weightsRegistry The weights registry to deallocate.
MODULAR_API_EXPORT void
M_freeWeightsRegistry(M_WeightsRegistry *weightsRegistry);

#endif // MAX_C_WEIGHTS_H
