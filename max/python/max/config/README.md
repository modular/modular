# Cyclopts Configuration Utilities

This module provides reusable utilities for configuring Cyclopts CLI
applications with environment variable and YAML config file support.

## Configuration Precedence

Configuration values are resolved in the following order (highest to lowest priority):

1. **CLI arguments** - Values provided directly on the command line
2. **Config files (YAML)** - Values from YAML configuration files specified via `--config-file`
3. **Environment variables** - Values from `MODULAR_*` environment variables
4. **Defaults** - Default values defined in configuration classes

## Quick Start

### Basic Example

```python
from cyclopts import Parameter
from max.config.cyclopts_config import create_cyclopts_app, setup_config_file_meta_command
from pydantic import BaseModel, Field

# Define your subconfig classes.
# Make sure to include this @Parameter decorator with name="*" and a group name
# if you'd like the fields of a subconfig to be grouped together.
@Parameter(name="*", group="Model Options")
class ModelConfig(BaseModel):
    model: str | None = Field(default=None)
    seed: int = Field(default=42)

@Parameter(name="*", group="Shape Options")
class ShapeConfig(BaseModel):
    batch_size: int = Field(default=1)
    input_len: int = Field(default=256)


def main() -> None:
    """Main entry point for the CLI application."""
    # Create the Cyclopts app
    app = create_cyclopts_app(
        name="my_cli",
        help_text="My CLI application"
    )

    # Define your command
    @app.default
    def run(
        model_config: ModelConfig | None = None,
        shape_config: ShapeConfig | None = None,
    ) -> None:
        """Run the application."""
        if model_config is None:
            model_config = ModelConfig()
        if shape_config is None:
            shape_config = ShapeConfig()
        
        print(f"Model: {model_config.model}")
        print(f"Batch size: {shape_config.batch_size}")

    # Set up config file support
    setup_config_file_meta_command(
        app,
        root_keys="my_config",
        must_exist=False,
        search_parents=True,
    )


if __name__ == "__main__":
    main()
```

### Configuration File Example

Create a YAML config file (`config.yaml`):

```yaml
my_config:
  model: "llama-2-7b"
  seed: 123
  batch_size: 8
  input_len: 512
```

Run your CLI:

```bash
# Using config file
python my_cli.py --config-file config.yaml

# CLI args override config file
python my_cli.py --config-file config.yaml --batch-size 16

# Environment variables override defaults but are overridden by config file
export MODULAR_BATCH_SIZE=4
python my_cli.py --config-file config.yaml  # batch_size will be 8 from config, not 4
```

## Usage Patterns

### Pattern 1: Environment Variables Only

```python
app = create_cyclopts_app(name="my_cli", help_text="My CLI")

@app.default
def run(model_config: ModelConfig | None = None) -> None:
    # Use model_config...

# Users can set: export MODULAR_MODEL=llama-2-7b
# Then run: python my_cli.py
```

### Pattern 2: Config Files with Root Keys

```python
app = create_cyclopts_app(name="my_cli", help_text="My CLI")
setup_config_file_meta_command(app, root_keys="app_config")

@app.default
def run(model_config: ModelConfig | None = None) -> None:
    # Use model_config...

# config.yaml:
# app_config:
#   model: llama-2-7b
# 
# Run: python my_cli.py --config-file config.yaml
```

### Pattern 3: Multiple Config Files

```python
app = create_cyclopts_app(name="my_cli", help_text="My CLI")
setup_config_file_meta_command(app, root_keys="config")

# python my_cli.py --config-file hardware.yaml --config-file benchmark.yaml
```

### Pattern 4: Testing Without sys.argv

```python
app = create_cyclopts_app(name="test_app", help_text="Test app")
setup_config_file_meta_command(
    app,
    root_keys="test_config",
    call_meta=False,  # Don't call app.meta() automatically
)

# In tests, manually set up config sources or call app.meta() with args
```

## Precedence Examples

### Example 1: CLI Overrides Everything

```bash
# config.yaml has: batch_size: 8
# Environment has: MODULAR_BATCH_SIZE=4
# Default is: batch_size: 1

python my_cli.py --config-file config.yaml --batch-size 16
# Result: batch_size = 16 (from CLI)
```

### Example 2: Config File Overrides Env Vars

```bash
# config.yaml has: batch_size: 8
# Environment has: MODULAR_BATCH_SIZE=4
# Default is: batch_size: 1

python my_cli.py --config-file config.yaml
# Result: batch_size = 8 (from config file, not env var)
```

### Example 3: Env Vars Override Defaults

```bash
# Environment has: MODULAR_BATCH_SIZE=4
# Default is: batch_size: 1

python my_cli.py
# Result: batch_size = 4 (from env var, not default)
```

### Example 4: Defaults Used When Nothing Else Provided

```bash
# No config file, no env vars
# Default is: batch_size: 1

python my_cli.py
# Result: batch_size = 1 (from default)
```

## Environment Variable Naming

Environment variables must be prefixed with `MODULAR_` and use uppercase
with underscores. The variable name is derived from the parameter name:

- Parameter: `batch_size` → Env var: `MODULAR_BATCH_SIZE`
- Parameter: `input_len` → Env var: `MODULAR_INPUT_LEN`
- Parameter: `model` → Env var: `MODULAR_MODEL`

## YAML Config File Format

### With Root Keys

If you use `root_keys="my_config"`:

```yaml
my_config:
  model: "llama-2-7b"
  batch_size: 8
  input_len: 512
```

### Without Root Keys

If `root_keys=None`:

```yaml
model: "llama-2-7b"
batch_size: 8
input_len: 512
```

## Best Practices

1. **Always provide defaults** in your configuration classes to ensure
   the app works without any configuration
2. **Use root keys** in YAML files to namespace your configuration and avoid conflicts
