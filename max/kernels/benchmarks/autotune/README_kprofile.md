
# `kprofile`: Profile `kbench` output pickle

`kprofile` is a tool to review and extract insight from `kbench` results
stored in `pkl` files.

`kprofile` can work with multiple pkl files, displaying the output of one
after the other.
This effectively groups the outputs per shape, allowing to select the
different tuning parameters.

## Example

- Simply print the top result:

```bash
kprofile output.pkl
```

- Find the most frequent values for each parameter in the top 5% of the results

```bash
kprofile output.pkl --top 0.05
```

- Printing a simplified table with running time ratio of each entry to the top entry

```bash
kprofile sample.pkl -r
```

- Printing the head 10 best and tail 10 worst entries

```bash
kprofile sample.pkl --head 10 --tail 10
```

- Grouping together multiple pkl files from different runs and showing the
best 2 results for each of them

```bash
kprofile file*.pk --head 2
```

## Filling in parameters in mojo files

You can use `kprofile` to replace the parameters from a mojo program
by the best tuning parameters, effectively injecting the best parameters
onto a file.

```bash
kprofile output.pkl -s path_to_snippet.mojo
```

The mechanism is a simple string and replace, so developers must be careful
with accidental pattern matching.

To replace the values in snippet, simply encode each parameter as
    `[@parameter_name]`. For example, for parameter `NUM_BLOCKS` in the
    following snippet:

```mojo
alias num_blocks = [@NUM_BLOCKS]
```
