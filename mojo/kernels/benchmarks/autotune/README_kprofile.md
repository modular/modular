
# `kprofile`: Profile `kbench` output pickle

```plaintext

Options:
  -o, --output TEXT   Path to output file.
  -t, --top FLOAT     Form a new spec from frequent values of each param from
                      top percent.
  -s, --snippet TEXT  Path to snippet to replace the parameters with values.
  -v, --verbose       Verbose printing.
  --help              Show this message and exit.
```

# Example

- Simply print the top candidate:

```bash
kprofile output.pkl
```

- Find the most frequent values for each parameter in the top 5% of the candidates

```bash
kprofile output.pkl --top 0.05
```

- Replace the parameters in a code snippet with values from the top spec

```bash
kprofile output.pkl -s path_to_snippet.mojo
```

- To replace the values in snippet, simply encode each parameter as
    `[@parameter_name]`. For example, for parameter `NUM_BLOCKS` in the
    following snippet:

```mojo
alias num_blocks = [@NUM_BLOCKS]
```
