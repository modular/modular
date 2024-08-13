# `mojo test` troubleshooting

## Argument parsing

If you see this particular assertion:

```txt
Assertion failed: ((unsigned) (id - 1) < getNumOptions() && "Invalid ID."), function getOption, file OptTable.cpp, line 147.
```

This is caused by side effects in the LLVM argument parser. We've seen this
happen with argument groups and calls to `InputArgList::filtered` in particular.
Extracting an argument value seems to deallocate internal state and cause this
assertion to be hit. The resolution varies, but a simple one is to just re-parse
the arguments before calling `filtered`.
