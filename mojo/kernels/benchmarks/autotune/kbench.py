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
import string
import sys
import numpy as np
import pandas as pd
import click
from time import sleep, time
from rich.progress import MofNCompleteColumn, Progress
from model.utils.logging import CONSOLE

from modular.utils import logging
from modular.utils.subprocess import (
    list2cmdline,
    run_shell_command,
)
import subprocess
from modular.utils.yaml import YAML
import csv

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

        # substitute env variables in the path
        file_abs_path = Path(string.Template(self.file).substitute(os.environ))
        try:
            assert file_abs_path.exists()
        except:
            print(f"ERROR: [{self.file}] doesn't exist!")
            return

        cmd = [
            MOJO_BINARY,
            *list(
                np.array([param.define() for param in self.params]).flatten()
            ),
            file_abs_path,
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
                    print(list2cmdline(cmd))
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

    def to_obj(self) -> str:
        obj = {}
        for param in self.params:
            obj[param.name] = param.value
        return obj


@dataclass(repr=True)
class Spec:
    name: str
    file: Path
    params: List[object] = field(default_factory=list)
    mesh_idx: int = 0
    mesh_size: int = 0
    instances: List[SpecInstance] = field(default_factory=list)

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

    def dump_yaml(self, out_path: Path):
        assert self.instances, "There are no instances to write to YAML!"
        obj = {
            "name": self.name,
            "file": self.file,
            "params": [s.to_obj() for s in self.instances],
        }
        with open(out_path, "w") as f:
            YAML(typ="safe").dump(obj, f, sort=False)
        print(f"dumped {len(self.instances)} instances to [{out_path}]")

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

        params = []
        if "params" in obj.keys():
            for cfg in obj["params"]:
                e = []
                for k, v in cfg.items():
                    e.append(ParamSpace(name=k, value=v))
                params.append(e)

        return Spec(
            name=obj["name"],
            file=obj["file"],
            params=params,
        )

    def __len__(self):
        assert self.instances
        return self.mesh_size

    def __post_init__(self):
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

        At the end, append the configs with fixed parameters, if any exists in YAML.

        Return the total size of mesh.
        """

        # params
        for cfg in self.params:
            name_list = [p.name for p in cfg]
            param_list = [p.value_list for p in cfg]
            param_mesh = list(product(*param_list))
            num_params = len(cfg)
            for idx in range(len(param_mesh)):
                s = SpecInstance(
                    name=self.name,
                    file=self.file,
                    params=[
                        Param(name=name_list[i], value=param_mesh[idx][i])
                        for i in range(num_params)
                    ],
                )
                self.instances.append(s)
        self.mesh_idx = 0
        self.mesh_size = len(self.instances)

        return self.mesh_size

    def __iter__(self):
        return self

    def __next__(self) -> "SpecInstance":
        assert (
            self.instances != None
        ), "Should call self.init_mesh after loading or in postinit."

        # Stop condition
        if self.mesh_idx == self.mesh_size:
            self.mesh_idx = 0
            raise StopIteration

        # Retrieve and update self.mesh_idx
        idx = self.mesh_idx
        self.mesh_idx = self.mesh_idx + 1
        return self.instances[idx]


SPEC_CONTENT = """
name: multistage_gemm
file: ./sample.mojo
params:
  - DTYPE: DType.float16
    M: [1024,512]
    N: [1024,512]
    STAGES: [4,8,12]

  - DTYPE1: DType.float32
    M1: [1024]
    N1: 768
    STAGES1: 12

  - DTYPE2: DType.float16
    M2: 132
    N2: 768
    STAGES2: 14

"""


def _get_tmp_path(file_path):
    base = os.path.basename(file_path).split(".")[0]
    tf = tempfile.NamedTemporaryFile(prefix=str(base) + "_").name + "/"
    return Path(tf)


def run(yaml_path, output_path=None, tune=True, verbose=False, tmp_path=None):
    spec = Spec.load_yaml(Path(yaml_path))
    # spec.dump_yaml(Path("rewrite.yaml"))
    # else:
    #     spec = Spec.loads(SPEC_CONTENT)

    output_path_list: Dict[int, Path] = {}
    spec_list: Dict[int, SpecInstance] = {}
    # Run the code over the mesh of param/values

    if not tmp_path:
        tmp_path = _get_tmp_path(spec.file)
    tmp_dir = Path(tmp_path)

    t_start = time()
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
            # "rm -rf {output_dir} && mkdir -p {output_dir}"
            if os.path.exists(output_dir) and os.path.isdir(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=False)
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

    t_elapsed = time() - t_start
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

    output_lines = []
    output_lines += [LINE]
    output_lines += [f"Tuning [{spec.name}] from [{spec.file}]"]
    output_lines += [LINE]
    output_lines += [f"Number of valid specs: {len(valid_specs)}"]

    if valid_specs:
        merged_df = pd.concat(valid_specs, axis=0, ignore_index=True)
        ########################################################
        # Get the name of column 2 (met (ms))
        met_col = merged_df.columns[2]

        if tune:
            sorted_df = merged_df.sort_values([met_col], ascending=True)
            output_lines += [sorted_df.to_string(index=False)]
            # Index to top spec after sort
            top_spec_idx = sorted_df.iloc[0].mesh_idx
            output_lines += [f"top_spec_idx: {top_spec_idx}"]

            output_lines += [LINE]
            output_lines += ["Best Measured Time:"]
            output_lines += [LINE]
            output_lines += [spec_list[top_spec_idx].to_obj()]
            output_lines += [LINE]
        else:
            sorted_df = merged_df.sort_values([met_col], ascending=True)
            output_lines += [merged_df.to_string(index=False)]
            output_lines += [LINE]
        ########################################################
        output_lines += ["Elapsed tuning time: %.1f (s)" % (t_elapsed)]
        output_str = "\n".join([str(x) for x in output_lines])
        print(output_str)
        if output_path:
            if tune:
                with open(output_path, "w") as f:
                    f.write(output_str + "\n")
            else:
                merged_df.drop(columns=["mesh_idx"]).to_csv(
                    output_path, index=False, quoting=csv.QUOTE_NONNUMERIC
                )
            print(f"wrote results to [{output_path}]")


help_str = (
    "Grid-search all the params for a mojo benchmark and pick the top value"
)


@click.command(help=help_str, no_args_is_help=True)
@click.option("--yaml", "yaml_path", help="Path to config yaml.")
@click.option(
    "--output", "-o", "output_path", default=None, help="Path to output file."
)
@click.option(
    "--tune",
    "-t",
    "tune",
    is_flag=True,
    default=False,
    help="Tune or just run.",
)
@click.option("--force", "-f", is_flag=True, default=False, help="Force.")
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Verbose printing."
)
def cli(
    yaml_path,
    output_path,
    tune,
    force,
    verbose,
) -> bool:
    run(
        yaml_path=yaml_path,
        output_path=output_path,
        tune=tune,
        verbose=verbose,
    )

    # TODO: add --expand and --factorize options to rewrite YAML files


def main():
    cli()


if __name__ == "__main__":
    main()
