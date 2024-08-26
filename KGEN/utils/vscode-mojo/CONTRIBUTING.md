# Developer Guidelines

## `null` vs `undefined`

We don't use `null` in the codebase, except for cases in which an external API
expects it.
We do this to simplify handling of optionals and for the conveniences that the
language provides for undefined values.

If you need a way to specify an absence of a value that can't just be described
with `undefined`, then use an enum.

As an convenience, use the `Optional` type to have a unified way to express optionals.
