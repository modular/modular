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
// This file centralizes all metal-cpp includes with proper warning
// suppressions.
//
//===----------------------------------------------------------------------===//

#ifndef TARGET_LIB_METAL_METALCPP_H
#define TARGET_LIB_METAL_METALCPP_H

#include "Support/PlatformUtils.h"

#if MODULAR_MACOS

// Suppress warnings from metal-cpp headers
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wdtor-name"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wundef"

#include "Foundation/Foundation.hpp"
#include "Metal/Metal.hpp"

#pragma clang diagnostic pop

#endif // MODULAR_MACOS

#endif // TARGET_LIB_METAL_METALCPP_H
