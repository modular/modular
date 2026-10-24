# Device Context usage in the graph compiler

This is a short document to summarize how DeviceContexts are chosen by the graph
compiler, and how they are initialized at runtime. There was previously some
confusion about how this worked. At one point handling of the CPU device context
was special cased so a null pointer was passed, but this is no longer true.

Selection of the appropriate device contexts is made during MO to MOGG lowering.
If a MOGG Primitive takes a DeviceContext argument, it is added to the calls
during this lowering.

The added arguments are of the type `mogg.device_context_placeholder`, which
become `mgp.device_context` attributes. When lowered to MEF, all device
references used by the model are serialized as a JSON string.

When the model is loaded by the model executor, DeviceContexts are created for
each device used by the model (see `Model::setup` in
`GraphCompiler/lib/MEFModel/MEFModel.cpp`). These are then passed to the kernels
as expected when executing the primitives.
