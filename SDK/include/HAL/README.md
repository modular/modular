# Driver SDK

## ABI-safe interface with hardware vendors

We envision a day where hardware vendors could independently prototype custom
Driver-implementations in order to hook into, and evaluate, the Modular stack.
These will be developed against (SDK versions corresponding to) one of our
(stable) releases. This sub-package prescribes our interface, as C API, towards
ABI-safety. Note that compiler-flags also play a role, beyond the C
specification. We will holistically evolve this with time.

Its consumers are plugins for various hardware.
Officially-supported consumers - CUDA, HIP and Metal - live in
`//Target/...`. Experimental consumers (like ongoing pre-market partnerships)
live in repos outside the mono-repo.

For the NeoDriver design built on this interface, including the Mojo client API,
plugin hooks, and capability-based extensions, see
[`max/mojo/max/_driver/max_license/README.md`](../../../max/mojo/max/_driver/max_license/README.md).

## Development Philosophy

This sub-package is a prescription for external HW vendors, not just for
internal developers. Therefore, it intentionally maintains only the bare
minimum dependencies on the mono-repo, in order to facilitate independent
development by external partners/vendors. As an important, desired side-
effect, it makes internal developers simulate (dog-food) the external-
developer UX.

In future, our notion of a SDK may become expand and we may expose
to expose a larger set of standardized dependencies to the outside world
through this (`//SDK`) aperture.
