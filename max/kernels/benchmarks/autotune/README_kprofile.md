
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

- Printing a simplified table with running time ratio of each entry to the top
  entry

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

## Generate checked-in tuning tables

`kprofile` analyzes benchmark results and writes selected configs to YAML. For
a checked-in dispatch table, this YAML file is the *manifest*: the ordered list
of configs that should appear in Mojo.

`tuning_codegen` is the supported path from a manifest to checked-in Mojo. It
renders each config through a snippet by replacing `[@NAME]` placeholders, then
replaces the target file's contents between unique begin and end markers.
Without `--check`, it updates the target file. With `--check`, it reports a diff
and fails if the checked-in region is stale.

Each manifest has a primary snippet. An entry can set `_template` to select a
snippet variant when one table contains configs with different Mojo constructor
shapes. If every entry has the same shape, use only the primary snippet.

The tuning-table workflow is:

1. Run `kprofile` on `kbench` output to create or update the YAML manifest.
2. Review the selected configs.
3. Run `tuning_codegen` to update the marked Mojo region.
4. Format the result and run the table's freshness test.
