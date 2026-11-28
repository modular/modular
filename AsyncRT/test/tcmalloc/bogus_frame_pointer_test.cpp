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
// A sampled TCMalloc allocation records its stack with absl::GetStackTrace,
// which walks frame pointers. Mojo code does not keep frame pointers, so at an
// allocation the frame pointer register can hold any value, including the end
// of the stack, and the walk then reads past it. Allocation sampling is off,
// so allocating must not walk the stack at all.
// https://github.com/modular/modular/issues/7526
//
//===----------------------------------------------------------------------===//

#include "AsyncRT/Runtime/Globals/Globals.h"

#include "gtest/gtest.h"

#include <cstddef>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace M::AsyncRT;

// Calls fn(arg) with %rbp set to fp, as code that uses %rbp as a general
// register may leave it.
extern "C" void *callWithFramePointer(void *fp, void *(*fn)(void *), void *arg);
asm(R"(
  .text
  .p2align 4
  .globl callWithFramePointer
  .hidden callWithFramePointer
  .type callWithFramePointer, @function
callWithFramePointer:
  push %rbp
  mov %rdi, %rbp
  mov %rdx, %rdi
  call *%rsi
  pop %rbp
  ret
  .size callWithFramePointer, .-callWithFramePointer
)");

namespace {

struct ThreadState {
  char *stackEnd;
  // The unwinder only follows a frame pointer up to 100000 bytes away, so the
  // stack end must be closer than that for the test to exercise anything.
  bool stackEndInReach = false;
  bool allocationsSucceeded = false;
};

void *allocate(void *arg) {
  auto *state = static_cast<ThreadState *>(arg);
  auto *frame = static_cast<char *>(__builtin_frame_address(0));
  state->stackEndInReach = state->stackEnd - frame < 50000;
  // With sampling on, TCMalloc would sample about one allocation per 2 MiB,
  // so 1000 MiB of allocations would sample hundreds of them.
  for (int i = 0; i < 1000; ++i) {
    void *ptr = TCMallocGlobals::tc_new(64, 1 << 20);
    if (!ptr)
      return nullptr;
    TCMallocGlobals::tc_delete(ptr);
  }
  state->allocationsSucceeded = true;
  return nullptr;
}

void *threadMain(void *arg) {
  auto *state = static_cast<ThreadState *>(arg);
  return callWithFramePointer(state->stackEnd, allocate, state);
}

// Allocates on a thread whose stack is directly followed by an inaccessible
// page, with the frame pointer register holding the end of the stack.
TEST(BogusFramePointerTest, AllocationsDoNotWalkTheStack) {
  const size_t pageSize = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  const size_t stackSize = 256 * 1024;
  void *mapping = mmap(nullptr, stackSize + pageSize, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(mapping, MAP_FAILED);
  char *stackEnd = static_cast<char *>(mapping) + stackSize;
  ASSERT_EQ(mprotect(stackEnd, pageSize, PROT_NONE), 0);

  pthread_attr_t attr;
  ASSERT_EQ(pthread_attr_init(&attr), 0);
  ASSERT_EQ(pthread_attr_setstack(&attr, mapping, stackSize), 0);
  ThreadState state{stackEnd};
  pthread_t thread;
  ASSERT_EQ(pthread_create(&thread, &attr, threadMain, &state), 0);
  ASSERT_EQ(pthread_join(thread, nullptr), 0);
  pthread_attr_destroy(&attr);
  munmap(mapping, stackSize + pageSize);

  EXPECT_TRUE(state.stackEndInReach);
  EXPECT_TRUE(state.allocationsSucceeded);
}

} // namespace
