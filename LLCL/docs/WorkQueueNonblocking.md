# The nonblocking design of `LLCL::WorkQueue`

One of the key problems that a thread pool must solve is how they behave when an
item of work they execute blocks its thread (e.g. on I/O).  When this happens,
the thread is implicitly taken out of the thread pool, and therefore the machine
ends up being over- or under-utilized.

For example, consider a 4-core machine: you can have 4 threads keeping the
machine busy, but if one of them blocks on disk or network I/O for 100ms, then
you've just given up 1/4 of your CPU cycles for 100ms that could be used to
execute other work in the work queue.

This is a challenging problem to deal with, particularly with large scale
software systems - most existing code in the world was built on top of
existing blocking APIs (e.g. even simple things like `printf` can block!). There
are two major approaches to solving this problem: adaptive thread pools... and
what LLCL does. :)

## Adaptive thread pools

One classical way to try to solve for this is with adaptive thread pools, this
is how (e.g.) Apple's Grand Central Dispatch (GCD) API works.

Unfortunately, there are many problems with adaptive thread pools:

1) They end up firing up many more threads than the CPU has cores, relying on
   the kernel to switch between them, or with equivalent user-space
   functionality, or hybrids (M:N, fibers, etc).  Regardless of the
   implementation approach, it is inefficient to lose processor caches and other
   state on each switch, and leads to poor latency stability.
2) You end up with weird edge cases where they run out of resources, e.g.
   crashing the system because you can't allocate enough thread stacks, or
   deadlock your app due to [other limitations](https://stackoverflow.com/questions/15150308/workaround-on-the-threads-limit-in-grand-central-dispatch).
3) The complexity of these systems escalate quickly because there is no
   structure to the problem.  Tasks get markable with Quality of Service markers
   to help the scheduler, new kinds of queues get introduced for special cases,
   etc.  The [source code for 
   GCD](https://github.com/apple/swift-corelibs-libdispatch) is open source and
   relatively portable for anyone to inspect.
4) Uncooperative legacy code often talks to other concurrency approaches and has
   other non-compositional behavior beyond just blocking.
5) They don't provide an incentive for developers to move to non-blocking APIs.

Unfortunately, after many many years of trying to solve this problem, and a fast
ramp of complexity, it has become clear that there isn't a reasonable way to
solve this problem, even at Apple scale.

Partially as a consequence of these learnings, Apple has rolled out an entirely
new language-based concurrency approach in Swift built on async/await and
actors that eliminates blocking... but that isn't helpful to us in C++ land.

## LLCL's Approach for Blocking Tasks

XXX Our approach for `LLCL/Runtime` and `WorkQueue` is ... 
XX For LLCL/Runtime, we take a different approach, which leverages our library
XX based design await/quiesce to donate the host thread.


TODO: keep writing.



