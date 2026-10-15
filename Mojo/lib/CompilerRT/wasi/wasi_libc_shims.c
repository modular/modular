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
// Symbols the stdlib calls that wasi-libc does not provide.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <time.h>

#include <wasi/api.h>

#ifdef __cplusplus
extern "C" {
#endif

/* wasi-libc's clockid_t is & _CLOCK_REALTIME, not 0/1. The stdlib passes an
 * int. */
int64_t clock_gettime_nsec_np(int clock_id) {
  struct timespec ts;
  clockid_t clk;
  switch (clock_id) {
  case 0:
    clk = CLOCK_REALTIME;
    break;
  case 1:
    clk = CLOCK_MONOTONIC;
    break;
  default:
    return -1;
  }
  if (clock_gettime(clk, &ts) != 0)
    return -1;
  return (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
}

/* WASI has no process CPU clock. Monotonic time since the first call. */
int64_t process_cputime_nsec_np(void) {
  static int have_start;
  static __wasi_timestamp_t start;
  __wasi_timestamp_t now = 0;
  if (__wasi_clock_time_get(__WASI_CLOCKID_MONOTONIC, 0, &now) != 0)
    return -1;
  if (!have_start) {
    start = now;
    have_start = 1;
  }
  return (int64_t)(now - start);
}

#ifdef __cplusplus
}
#endif
