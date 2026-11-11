# max._driver

Notes for agents working on the Mojo HAL, the HAL `DeviceContext`
(`max_license/device_context_hal`) and its plugin hooks
(`max_license/hal/_plugin`). The root `AGENTS.md` covers everything general.

## Resolve per-target things from the type's own target

A process holds contexts for more than one target, the host CPU alongside each
accelerator, so anything keyed on `get_device_spec[0]()` or
`default_accelerator()` is wrong for every context but one. Anything that
depends on the device, such as the plugin's hooks or its GPU info, comes from
the `CompilationTarget` the type is parameterized on, resolved right where it's
used:

```mojo
struct DeviceContext[target: CompilationTarget = default_accelerator()]:
    comptime _HALHooks = MaxPluginForTarget[Self.target].HalHooks
```

`MaxPluginForTarget` (`max/_plugin`) is the one place the selection rule lives,
and its `HalHooks` member holds the HAL half of a plugin. Always pass it the
type's own target. Don't add a helper that supplies the
target for you (eg, one keyed on `default_accelerator()`). It hides which
target a lookup is keyed on, and reads like a global even when it isn't one.

Code without a target parameter of its own gets one from the context it's
handed. `DeviceFunction`'s launch path is generic over a
`_HALFunctionEnqueuer`, so it reads the enqueuer's `_target_value`. That's a
raw `_TargetType` rather than a `CompilationTarget`, because each target is its
own `CompilationTarget` type and a trait member can't be typed as any of them.
A type that should carry a target takes it as a parameter, as `DeviceStreamBase`
does, with a defaulted alias (`DeviceStream`) for the default accelerator.
