# Plugins for hardware support

This folder hosts officially-supported plugins for the Driver's Hardware
Abstraction Layer (HAL). Each plugin supports a different hardware vendor.
For example, CUDA, HIP and Metal plugins will live here.

Plugins for confidential partnerships MUST NOT be placed here, or anywhere else
in the mono-repo! They MUST be developed in carefully controlled internal repos
created and maintained by the Platform Engineering team.

However, the architecture to consume all plugins, whether officially-supported
(resident in this sub-tree) or confidential (resident in other repos) is the
same.

## Development Philopsophy

Each plugin implements the ABI-safe C API functions declared in
`//SDK/include/HAL/*.h`. It must implement the bare minimum required to achieve
the functionality mandated by these API; higher-level functions and features
(reference-counting and caching, for example) must be done above the HAL so as
to apply uniformly to all supported devices. That is, plugins must only
translate the functionality required by each `M_driver_` HAL API function to
sequences of function calls to HW-vendor driver APIs (e.g., CUDA driver). The
rest of the (pan-HW) intelligence must be developed inside **our** HAL
implementation, never in a plugin.

We envision a day where hardware vendors could independently prototype custom
Driver-implementations in order to hook into, and evaluate, the Modular stack.
These will be developed against (SDK versions corresponding to) one of our
(stable) releases. While they must conform to the interface we specify, they
will be free to choose their dependencies. These plugins simulate that mindset.
**Hence, these plugins are an exception in our code-base: they must simulate
independent development, build and installation environments**. To that end,
the code in this folder strives to consume nothing other than dependencies
within `//SDK` - a narrow, controlled aperture into our mono-repo.

It can be tempting to argue for greater dependency on the mono-repo, say,
on `//Support:Base`. But do plugins really need to reference-count or
use mutexes and thread-pools? Even if they do, is this better done above the
HAL uniformly across all hardware types? We depend on `//SDK` to
regulate (not constrain!) dependencies. We expect the SDK to grow over time;
plugin-implementations can definitely source more dependencies from the
mono-repo with time, via the SDK, after getting past the above key questions
on a case-by-base basis. That is, while it can be argued that these plugins
are (also) for internal consumption and therefore qualify to be treated like
the rest of internally-developed code, it should also be noted that interfacing
with externally-developed hardware is **unique to this part of the code-base**
and that there is value in dog-fooding the external-developer experience. Also
note that the Driver-HAL - platform-agnostic complementary logic within the
Driver - is truly and completely internal to the mono-repo and is "business
as usual" up the stack starting with the Driver-HAL.

## Physical Organization

Why do we have a separate, top-level `Target` directory? Why is plugin-code not
placed inside `MLRT/...` sub-packages?

This is inspired by
<https://github.com/llvm/llvm-project/tree/main/llvm/lib/Target>.

The architecture here also applies to the Mojo Compiler: it could be
conceptually split into SDK, Plugin and HAL components. The `MLRT/...`
organization is not appropriate. This `Target` directory then consolidates
all target-specific code into one, discoverable location.
