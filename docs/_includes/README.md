# InstallModular component

The `InstallModular` component is a reusable MDX component that generates
installation instructions for the Modular package across multiple package
managers (pixi, uv, pip, and conda).

## Usage

Import and use the component in your MDX files:

```python
import InstallModular from '@site/docs/_includes/install-modular.mdx';

<InstallModular folder="my-project" extraLibraries={["torch", "transformers"]} />
```

## Properties

- `folder` (optional): The name of the project folder to create during installation.
  - Type: `string`
  - Default: `"example-project"`

- `extraLibraries` (optional):  Additional libraries to install alongside `modular`.
  - Type: `string[]`
  - Default: `[]` | `modular`

## Examples

Basic usage (modular only):

```markdown
<InstallModular />
```

 Custom project folder:

```markdown
<InstallModular folder="llama-tutorial" />
```

With additional libraries:

```markdown
<InstallModular folder="ml-project" extraLibraries={["torch", "numpy", "pandas"]} />
```

## Generated output

The component generates tabbed installation instructions for:

- **pixi**: Uses conda channels with `pixi add`
- **uv**: Uses pip with custom index URLs via `uv pip install`
- **pip**: Standard pip installation with extra index URLs
- **conda**: Uses conda-forge and Modular's conda channels
