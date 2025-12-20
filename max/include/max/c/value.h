//===----------------------------------------------------------------------===//
// Copyright (c) 2025, Modular Inc. All rights reserved.
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

#ifndef MAX_C_VALUE_H
#define MAX_C_VALUE_H

#include "max/c/symbol_export.h"
#include "max/c/types.h"

/// Deallocates the memory for the container.  No-op if `value` is `NULL`.
///
/// @param value The value to deallocate.
MODULAR_API_EXPORT void M_freeValue(M_AsyncValue *value);

#endif // MAX_C_VALUE_H
