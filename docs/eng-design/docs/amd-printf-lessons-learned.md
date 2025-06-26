---
title: AMD Print Lessons Learned
author: Lukas Hermann
date: January 21, 2025
---

- **Author:** Lukas Hermann
- **Date:** January 21, 2025

Here at Modular, our kernel engineers needed the ability to debug using print
statements inside of kernels on AMD GPUs for situations where a debugger
offered more signal than noise.  Printing is a fundamental operation in
programming, so it’s not unreasonable to expect drivers and operating systems
to provide it as a primitive.  And in fact, NVidia offers just that with their
`vprintf` syscall.

However as we’ll see, this is not the case with AMD.  A main motivation of this
work was to avoid including AMD's `device-libs`, as that would require adding a
whole other copy of LLVM, and basically including an entire OpenCL compiler in
our toolchain.

Before I dive in, I’d also like to thank Eric Hein and Tracy Sharpe for pairing
with me frequently on this project.  They were able to help figure out some key
issues during this process.

## What Do Hostcalls Look Like On AMD GPUs?

There are some operations that only make sense on a CPU.  Various system calls
require access to the operating system in ways that would be expensive and
impractical on GPUs.  A **hostcall** is an asynchronous message from the GPU
(the **device**) to the CPU (the **host**) instructing the CPU to execute some
command, and potentially pass the results back to the GPU.

Writing to stdout is one of those operations that only the CPU can do, and
therefore requires a hostcall. In order to understand why this is tricky, we
first have to understand how a print call works in a GPU kernel.  Thankfully,
AMD open sources their drivers, runtimes, and compute libraries, so we can take
a peek under the hood directly!  At a high level, the GPU driver [spawns a
listener
thread](https://github.com/ROCm/clr/blob/9d8d35ae4041ef0f37430b1265e0ad60698d5b51/rocclr/device/devhostcall.cpp#L222)
during the life of the program, and intercepts hostcalls.  This thread actually
only exists if a program makes a hostcall, but that’s just an optimization.
From the device side, we are able to [emit a
hostcall](https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs/ockl/src/hostcall.cl#L44)
, which this thread will receive and run.

Since `printf` is such a fundamental operation, the runtime even has a [special
handler](https://github.com/ROCm/clr/blob/9d8d35ae4041ef0f37430b1265e0ad60698d5b51/rocclr/device/devhcprintf.cpp#L230)
for print operations.

It’s also important to understand that users expect a `print` call to happen
during execution of the kernel, rather than at the end of execution.  This
restriction means that we can’t simply pass in a buffer and then print it out
after the kernel is done executing.  The ability to asynchronously execute the
print is especially important in debugging, where a kernel may be crashing or
infinitely looping, so there wouldn’t even be a result returned from which to
get an output buffer.  On AMD there is a nice(-ish) [trio of print specific
wrappers](https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs/ockl/src/services.cl#L367)
around this whole mechanism in the form of `__ockl_printf_begin` ,
`__ockl_printf_append_args` , and `__ockl_printf_append_string_n`.

## Why is this difficult?

This doesn’t sound all that tricky. For NVidia GPUs all we had to do was emit
the `vprintf` instruction, so in this case we can just emit the wrapper
hostcalls from the device, and the listener thread will pick it up, right? For
CUDA, we emit PTX, and then NVIDIA libraries handle the compilation to device
code (SASS). At this step the calls to print are linked against their runtime
NVidia device libraries.

For HIP (AMD), we actually need to produce the final device binary in the first
step, which requires having actual implementations for all the device runtime
functions like print.  Keen observers will notice that the wrapper functions in
the AMD code are written in OpenCL.

That means, if we wanted to use the AMD `device-libs`, we would have to include
an OpenCL compiler and somehow link to the OpenCL code.  This would arguably
defeat the purpose of Mojo, which is to be a generic compute solution that can
target any processor type.  OpenCL has this same target domain, but with worse
ergonomics as it is handcuffed to C’s interface.  Worse, we would also need to
vendor AMD’s fork of LLVM, since the device libs and relevant changes are in a
fork rather than upstream.

The solution we settled on was to port the relevant OpenCL code to Mojo, which
meant making sure the ABI lined up.  Since AMD open sourced all of the relevant
code, the hard part was gaining context.  There is relatively little
documentation that I could find, which meant that I spent time following the
thread of how hostcall happens and messages are sent from both the host and
device sides.

We also didn’t have the typical debugging setup one would use.  We couldn’t set
breakpoints in the driver API to see if our messages were coming through
without compiling the drivers ourselves (which would have been a big lift).  We
also couldn’t print debug (though the irony of wanting to print debug our print
implementation wasn’t lost on us).

The main ways we debugged our work was by:

- Passing in an `NDBuffer` to the kernel, which we used as a print buffer.
- Return pointer values directly from the function.
- Eric eventually realized that we could actually use the `AMD_LOG_LEVEL`
  environment variable to get information about the current state of the HIP
  runtime during the Mojo process.

## Bugs 🐛

There were two main issues that caused most of the grief experienced during development.

1. The Bad Address Most of the message passing for hostcalls centers around a
   [buffer](https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs/ockl/src/hostcall_impl.cl#L45)
   pointer.  The address to this buffer is passed in via the implicit
   arguments, which can be accessed with the `llvm.amdgcn.implicitarg.ptr`
   intrinsic.  There was some bad bitcasting going on here, which poisoned the
   entire code path. Unfortunately, when trying to use my scratch buffer
   method, it seems the uninitialized values just printed a default value for
   everything, which made it hard to know exactly what was wrong.  What made
   this issue particularly tricky to debug is that the device holds a pointer
   to memory on the host, which basically never happens, except in this case
   the drivers map the addresses to be same for this one pointer.  Because this
   mapping goes against regular intuition, seeing different pointer values was
   not an immediate cause for concern.  This issue was made even more difficult
   by the lack of documentation surrounding `implicitarg_ptr`.  LLVM’s docs
   have several tables trying to explain the mapping of these implicit
   arguments, but they are tough to find and did not line up exactly with the
   values we were getting.  Kudos to Eric who eventually figured this out as a
   root cause.  Eric basically had to dump the binary to even figure out the
   `implicitarg` table from the symbols.

2. Using `mut UInt64` instead of `UnsafePointer[UInt64]` - The
   [push](https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs/ockl/src/hostcall_impl.cl#L141)
   and
   [pop](https://github.com/ROCm/llvm-project/blob/656552edc693e2bb4abc9258399c39d190fce2b3/amd/device-libs/ockl/src/hostcall_impl.cl#L100)
   functions take a `ulong *top` pointer, which points to the top of the
   buffers ready and free stacks respectively.  In hopes of not needing raw
   pointers, I had hoped that `buffer[].ready_stack` being fed into an argument
   `mut UInt64` would mean that the function gets a pointer in a way morally
   equivalent to `&buffer->ready_stack`.  What Tracy realized is that this is
   actually making a copy of a local.  This bug was particularly tricky because
   because it didn’t manifest for a kernel with `block_size=64`, `grid_size=1`,
   but it did for `block_size=2`, `grid_size=2`.  The fact that it *sometimes*
   worked lead to the atomics focused red herring mentioned below.

## Red Herrings

1. Atomics - There were several times where it *seemed* like bugs in `print`
   were coming from our use of stricter atomics (`seq_cst`) than what the AMD
   runtime was expecting.  The first time was at the very start of the project.
   Since the OpenCL code used different atomics, I had tried to match those,
   which would have meant expanding our atomics support in Mojo.  We then
   decided that less strict atomic ordering is more of an optimization than a
   correctness issue, so we pushed it off.  However, after we got `print`
   working on arbitrary `block_size` s but not `grid_size`s, we assumed the
   issue was from a mismatch in atomic ordering.  As I wrote above, this turned
   out to be a different issue.

2. Needing to roll our own hostcall listener (or at least thinking we had to) -
   Originally, Eric and I had assumed that the `HostcallListener` struct wasn’t
   being activated on our runtime and therefore we needed to make our own and
   call the C++ driver code to spawn it. After a little bit of fiddling around,
   Eric realized that the runtime was spawning the listener either way.  A
   takeaway here might be to dig more into the `clr` code so that we have a
   better understanding of what we’re given "for free."

3. Not exactly a red herring, but due to our use of the `Writable` trait,
   getting `print` to work is different from getting `_printf` to work.  We
   realized midway through the project that we could simply rely on our type
   system to do the formatting we needed.

## Conclusion

Now that `print` is working, our kernel engineers have a valuable new asset in
their debugging toolbox.  While it took more work, we implemented this *the
right way* by porting OpenCL code as opposed to bloating Mojo with yet another
copy of LLVM and a whole other language runtime.  Some of the issues we faced
when debugging were subtle, so it helped to have a few eyes on them.  A major
asset during this process was AMD’s open source ethos, which meant that we were
able to understand why the ABI was designed in a certain way, and understand
the pipeline from end-to-end.  Moving forward, we’ll have a stronger
understanding of the AMD host runtime, and even better tools for debugging as
we implement features.
