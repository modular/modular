# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import functools
import operator
import shutil
import tempfile
from itertools import product
from collections.abc import Iterable
from dataclasses import InitVar, dataclass, field
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import os
import sys
import numpy as np
import pandas as pd
import click
from time import sleep
from rich.progress import MofNCompleteColumn, Progress
from model.utils.logging import CONSOLE

from modular.utils import logging
from modular.utils.subprocess import (
    list2cmdline,
    run_shell_command,
)
import subprocess
from modular.utils.yaml import YAML


MOJO_BINARY = shutil.which("mojo")
if not MOJO_BINARY:
    raise Exception(f"Could not find the `mojo` binary.")

CURRENT_FILE = Path(__file__).resolve()
LINE = 80 * "-"


@dataclass(repr=True)
class Param:
    name: str
    value: object

    def define(self) -> str:
        return ["-D", f"{self.name}={self.value}"]


def flatten(value) -> List[object]:
    """Flattens a python value containing tensors into a list.

    The function supports nested lists, dictionaries, and tuples.
    """
    if not isinstance(value, Iterable):
        return [value]
    if isinstance(value, list):
        res = functools.reduce(operator.concat, [value])
        return res if isinstance(res, Iterable) else [res]
    if isinstance(value, dict):
        return flatten(value.values())
    if isinstance(value, tuple):
        return flatten(list(value))
    return [value]


@dataclass(repr=True)
class ParamSpace:
    name: str
    value: object
    value_list: List[object] = field(default_factory=list)
    length: int = 0

    def __post_init__(self):
        """Flatten the values in self.value and store them in a List
        Also, get the length of value list and store it in `length`.
        """
        self.value_list = flatten(self.value)
        self.length = len(self.value_list)


@dataclass(frozen=True, repr=True)
class SpecInstance:
    name: str
    file: Path
    params: List[Param] = field(default_factory=list)

    def compile(
        self,
        *,
        output_file: Optional[Path] = None,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> Path:
        if not output_file:
            output_file = Path(tempfile.gettempdir()) / Path(
                next(tempfile._get_candidate_names())
            )

        cmd = [
            MOJO_BINARY,
            *list(
                np.array([param.define() for param in self.params]).flatten()
            ),
            self.file,
        ]
        cmd.extend(["-o", "%s" % (str(output_file))])
        if verbose:
            print(f"[output_file: {output_file}")
        try:
            if dry_run:
                print(list2cmdline(cmd))
            else:
                # TODO: needs better error handling and error messages.
                if verbose:
                    run_shell_command(cmd)
                else:
                    p = run_shell_command(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
        except Exception as exc:
            raise Exception(
                f"Unable to run the command {list2cmdline(cmd)}"
            ) from exc

        return output_file

    def to_yaml(self) -> str:
        yaml_str = "params:"
        for param in self.params:
            yaml_str += f"\n  {param.name}: {param.value}"
        return yaml_str


@dataclass(repr=True)
class Spec:
    name: str
    file: Path
    params: List[ParamSpace] = field(default_factory=list)
    mesh_idx: int = 0
    name_list: List[str] = field(default_factory=list)
    mesh: List[object] = field(default_factory=list)
    mesh_size: int = 0

    @staticmethod
    def load_yaml(file: Path) -> "Spec":
        """
        Loads the spec from a YAML file

        Args:
            file (Path): the yaml file Path

        Returns:
            Spec: the spec
        """
        if not file.exists():
            raise FileNotFoundError(
                f'Unable to find the spec file at "{file}".'
            )
        try:
            return Spec.loads(file.read_text())
        except Exception as exc:
            raise Exception(f"Could not load spec from {file}") from exc

    @staticmethod
    def loads(yaml_str: str) -> "Spec":
        """
        Deserializes a Spec object from the given yaml string.

        Args:
            yaml_str (str): the yaml string representation of the model manifest

        Returns:
            Spec: a Spec loaded from the given yaml string
        """
        obj = YAML(typ="safe").load(yaml_str)
        return Spec(
            name=obj["name"],
            file=obj["file"],
            params=[
                ParamSpace(name=k, value=v) for k, v in obj["params"].items()
            ],
        )

    def __post_init__(self):
        self.init_mesh()
        self.num_params = len(self.params)

    def __len__(self):
        assert self.mesh
        return self.mesh_size

    def init_mesh(self):
        """
        Setup a mesh (cartesian product) of all values for all params. For example,
        if we have 2 set of params M=[64,256] and N=[A,B,C], the mesh will include
        to the following values:

        M=[64,256] x N=[A,B,C]
        ======================
        idx  : values
        0    : [64,A]
        1    : [64,B]
        2    : [64,C]
        3    : [256,A]
        4    : [256,B]
        5    : [256,C]

        Return the total size of mesh.
        """
        param_list = [p.value_list for p in self.params]
        self.name_list = [p.name for p in self.params]
        self.mesh = list(product(*param_list))
        self.mesh_idx = 0
        self.mesh_size = len(self.mesh)
        return self.mesh_size

    def __iter__(self):
        return self

    def __next__(self) -> "SpecInstance":
        assert (
            self.mesh != None
        ), "Should call self.init_mesh after loading or in postinit."

        # Stop condition
        if self.mesh_idx == self.mesh_size:
            self.mesh_idx = 0
            raise StopIteration

        # Retrieve and update self.mesh_idx
        idx = self.mesh_idx
        self.mesh_idx = self.mesh_idx + 1

        return SpecInstance(
            name=self.name,
            file=self.file,
            params=[
                Param(name=self.name_list[i], value=self.mesh[idx][i])
                for i in range(self.num_params)
            ],
        )


# file: ../autotune_multistage_gemm.mojo
SPEC_CONTENT = """
name: multistage_gemm
file: ./sample.mojo
params:
  DTYPE: DType.float16
  M: [1024,512]
  N: [1024,512]
  STAGES: [4,8,12]

"""


def tune(yaml_path, output_path=None, verbose=False):
    spec = Spec.load_yaml(Path(yaml_path))
    # else:
    #     spec = Spec.loads(SPEC_CONTENT)

    output_path_list: Dict[int, Path] = {}
    spec_list: Dict[int, SpecInstance] = {}
    # Run the code over the mesh of param/values

    tmp_dir = Path(tempfile.gettempdir())

    with Progress(
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        console=CONSOLE,
    ) as progress:
        bench_progress = progress.add_task(
            "build-run",
            total=spec.mesh_size,
        )

        for i, s in enumerate(spec):
            output_dir = Path(f"{tmp_dir}/out_{i}")
            # Check for the failure here.
            try:
                output_file = output_dir / "output.csv"
                s.compile(output_file=output_file, verbose=verbose)
                spec_list[i] = s
                output_path_list[i] = output_file
            except:
                pass
            # When a benchmark is completed for one combination of parameters we advance progress by 1
            progress.update(bench_progress, advance=1)

    ########################################################
    # Retrieve, sort, and pick top choices
    valid_specs = []
    for i in spec_list.keys():
        try:
            df = pd.read_csv(output_path_list[i], index_col=None, header=0)
            df.insert(0, "mesh_idx", i)
            valid_specs.append(df)
        except:
            df = None

    print(f"Number of valid specs: {len(valid_specs)}")
    if valid_specs:
        merged_df = pd.concat(valid_specs, axis=0, ignore_index=False)
        ########################################################
        # Get the name of column 1 (met (ms))
        met_col = merged_df.columns[1]
        sorted_df = merged_df.sort_values([met_col], ascending=True)
        print(sorted_df)
        # Index to top spec after sort
        top_spec_idx = sorted_df.iloc[0].mesh_idx
        print("top_spec_idx", top_spec_idx)

        print(LINE)
        print("Best Measured Time:")
        print(LINE)
        print(spec_list[top_spec_idx].to_yaml())
        print(LINE)
        ########################################################


help_str = (
    "Grid-search all the params for a mojo benchmark and pick the top value"
)


@click.command(help=help_str)
@click.option("--yaml", "yaml_path", help="Path to config yaml.")
@click.option(
    "--output", "output_path", default=None, help="Path to output file."
)
@click.option("--force", "-f", is_flag=True, default=False, help="Force.")
@click.option(
    "--verbose", is_flag=True, default=False, help="Verbose printing."
)
def cli(
    yaml_path,
    output_path,
    force,
    verbose,
) -> bool:
    tune(yaml_path=yaml_path, output_path=output_path, verbose=verbose)


def main():
    cli()


if __name__ == "__main__":
    main()
