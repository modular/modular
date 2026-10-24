# cuda-gdb tests

These tests are based on llvm-lit and have a custom executor that launches
cuda-gdb the correct way, besides adding some facilities to write these test
more easily. They also have additional substitution variables for easier
pattern matching.
In order to enable this kind of tests, you need to have a line with
`# RUN: %mojo-debug-cuda`.

## GDB-COMMAND and %cuda-gdb-commands

These tests allow you to specify a list of cuda-gdb commands via the
`# GDB-COMMAND` directive, which are converted automatically into the
`%mojo-debug-cuda` substitution variable.

## Substitution variables

### %breakpoint<name>-<attribute>

Add a comment of the form `# breakpoint<name>` at the end of the lines where you
want to place breakpoints, then you'll have the following substitutions at your
disposal:

- `%breakpoint<name>:location` -> `<source_file_name>:<breakpoint line number>`
- `%breakpoint<name>:line` -> `<breakpoint line number>`

Just make sure to use different names for each breakpoint location.
