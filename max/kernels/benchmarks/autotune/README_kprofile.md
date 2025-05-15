
# `kprofile`: Profile `kbench` output pickle

```plaintext

Options:
  -o, --output TEXT   Path to output file.
  -t, --top FLOAT     Form a new spec from frequent values of each param from
                      top percent.
  -s, --snippet TEXT  Path to snippet to replace the parameters with values.
  -r, --ratio         Print the running time ratio of each entry to the top
                      entry.
  --head INTEGER      The number of elements at head to print (sorted by
                      running time).
  --tail INTEGER      The number of elements at tail to print (sorted by
                      running time).
  -v, --verbose       Print all the (unsorted) entries from pkl.
  --help              Show this message and exit.
```

`kprofile` is essential for reviewing and extracting insight from `kbench`
results stored in `pkl` files.

# Example

- Simply print the top result:

```bash
kprofile output.pkl
```

- Find the most frequent values for each parameter in the top 5% of the results

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

- Printing a simplified table with running time ratio of each entry to the top entry

```bash
kprofile sample.pkl -r
```

- Printing the head 10 best and tail 10 worst entries

```bash
kprofile sample.pkl --head 10 --tail 10
```
