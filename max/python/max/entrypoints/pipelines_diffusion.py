"""Diffusion-only CLI wrapper.

This exists so Bazel can keep `//max/python/max/entrypoints:pipelines` lean,
while allowing `//max/python/max/entrypoints:pipelines_diffusion` to pull in
extra runtime deps.
"""

from __future__ import annotations

import sys


def main() -> None:
    # Import the main pipelines CLI and dispatch into the `diffusion` group.
    #
    # NOTE: `max.entrypoints.pipelines.main` is a click command object. Calling it
    # with `args=[...]` is equivalent to invoking the CLI with those argv tokens.
    import max.entrypoints.pipelines as pipelines_cli

    pipelines_cli.main(
        prog_name="pipelines",
        args=["diffusion", *sys.argv[1:]],
    )


if __name__ == "__main__":
    main()


