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

#if (defined(_WIN32) || defined(__CYGWIN__))
#ifdef MODULAR_BUILDING_FRAMEWORK
#define MODULAR_API_VISIBILITY_EXPORT __declspec(dllexport)
#else
#define MODULAR_API_VISIBILITY_EXPORT __declspec(dllimport)
#endif
#else
#define MODULAR_API_VISIBILITY_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
#define MODULAR_API_EXPORT extern "C" MODULAR_API_VISIBILITY_EXPORT
#else
#define MODULAR_API_EXPORT MODULAR_API_VISIBILITY_EXPORT
#endif
