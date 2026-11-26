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
#include <stdint.h>

uint8_t c_enum_two(uint8_t value) { return value ^ 1; }
uint16_t c_enum_wide(uint16_t value) { return value == 256 ? 0 : 256; }

int64_t c_enum_callbacks(uint8_t (*two)(uint8_t), uint16_t (*wide)(uint16_t)) {
  return two(1) == 0 && wide(256) == 0;
}
