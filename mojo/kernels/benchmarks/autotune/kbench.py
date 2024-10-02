# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import csv
import functools
import operator
import os
import shutil
import string
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import InitVar, dataclass, field
from enum import Enum
from itertools import chain, product
from pathlib import Path
from time import sleep, time
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import click
import numpy as np
import pandas as pd
from model.utils.logging import CONSOLE
from rich.progress import MofNCompleteColumn, Progress

from modular.utils import logging
from modular.utils.subprocess import list2cmdline, run_shell_command
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
    value_set: Set[object] = field(default_factory=set)
    length: int = 0

    def __post_init__(self):
        """Flatten the values in self.value and store them in a List
        Also, get the length of value list and store it in `length`.
        """
        self.value_set = set(flatten(self.value))
        self.value = None
        self.length = len(self.value_set)


@dataclass(frozen=True, repr=True)
class SpecInstance:
    name: str
    file: Path
    params: List[Param] = field(default_factory=list)

    def compile(
        self,
        *,
        output_file: Optional[Path] = None,
        build_opts: List[str] = [],
        dryrun: bool = False,
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

        cmd = [MOJO_BINARY]
        if build_opts:
            cmd.extend(build_opts)
        cmd.extend(
            [
                *list(
                    np.array(
                        [param.define() for param in self.params]
                    ).flatten()
                ),
                file_abs_path,
            ]
        )
        if not build_opts:
            cmd.extend(["-o", "%s" % (str(output_file))])

        # TODO: refactor the following into a separate function call, or invoke alias directly.
        # mojo-clear-cache
        # subprocess.call(
        #     (
        #         "rm -fr $MODULAR_DERIVED_PATH/.mojo_cache"
        #         " $HOME/.modular/.mojo_cache"
        #     ),
        #     shell=True,
        # )

        if verbose:
            print(f"[output_file: {output_file}")
        try:
            if dryrun:
                print(list2cmdline(cmd))
            else:
                # TODO: needs better error handling and error messages.
                if verbose:
                    print(list2cmdline(cmd))
                output = run_shell_command(
                    cmd, check=False, capture_output=True
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

    def __str__(self) -> str:
        tokens = [self.name]
        for param in self.params:
            tokens.append(f"{param.name}={param.value}")
        return "/".join(tokens)


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

    @staticmethod
    def load_yaml_list(yaml_path_list: List[str]) -> "Spec":
        spec = None
        for i, yaml_path in enumerate(yaml_path_list):
            spec_ld = Spec.load_yaml(Path(yaml_path))
            if i == 0:
                spec = spec_ld
            else:
                spec.join(spec_ld)
        return spec

    @staticmethod
    def parse_params(param_list):
        """
        Parse the parameters as (key,value) dictionary.
        The parameters can be defined as follows:
        - `PARAM_NAME:PARAM_VALUE` (single value)
        - `PARAM_NAME:[PARAM_VALUE0, PARAM_VALUE1]` (Pythonic list of values)

        Args:
            param_list (List): a list of param-value's as strings/

        Returns:
            Spec: Dictionary of with extra param names as keys and param values.
        """
        d = {}
        IFS = ":"
        for p in param_list:
            if IFS in p:
                name, val = p.split(IFS)
                break

            if name not in d.keys():
                d[name] = []

            # This supports list of params per one definition
            # The following works for parsing a single-value, or a Pythonic list of values.
            vals = val.split(",")
            vals[0] = vals[0].strip("[")
            vals[-1] = vals[-1].strip("]")
            for i, v in enumerate(vals):
                v = v.strip()
                try:
                    vals[i] = eval(v)
                except:
                    vals[i] = v
            d[name].extend(vals)
        return d

    def extend_params(self, param_list):
        # Expand with CLI params
        extra_params = self.parse_params(param_list)

        # For all params in each config either, update the exisiting `value_set`` with the new param value(s).
        for cfg in self.params:
            for k, v in extra_params.items():
                found = False
                for ps in cfg:
                    if ps.name == k:
                        ps.value_set.update(v)
                        found = True
                        break
                if not found:
                    cfg.append(ParamSpace(k, v))

        self.setup_mesh()

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
                    if k == "metadata":
                        continue
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
        self.setup_mesh()

    def setup_mesh(self):
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

        # clear instances just in case it is already populated
        # TODO: better way to separate post_init and mesh_Gen
        self.instances = []  # field(default_factory=list)

        # params
        for cfg in self.params:
            name_list = [p.name for p in cfg]
            param_list = [p.value_set for p in cfg]
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

    def join(self, other: "Spec"):
        assert self.name == other.name
        assert self.file == other.file
        assert other.mesh_size > 0

        self.mesh_idx = 0
        self.mesh_size += other.mesh_size
        self.params.extend(other.params)
        self.instances.extend(other.instances)

    def filter(self, filter_list: List):
        filters = {}
        for f in filter_list:
            if "=" in f:
                name, val = f.split("=")
            elif ":" in f:
                name, val = f.split(":")

            if name not in filters.keys():
                filters[name] = []
            filters[name].append(val)

        filtered_insts: List[SpecInstance] = []
        num_filters = len(filter_list)

        # Count the number of valid filters in each instance.
        # If the count==num_filters then add the instance to the result.
        valid_cnt = np.zeros(len(self.instances), dtype=np.int32)
        for k_filter, v_filter in filters.items():
            for i, s in enumerate(self.instances):
                for p in s.params:
                    if p.name == k_filter and str(p.value) in v_filter:
                        valid_cnt[i] += 1

        for i, idx in enumerate(valid_cnt):
            if idx == num_filters:
                filtered_insts.append(self.instances[i])

        self.instances = filtered_insts[:]
        self.mesh_idx = 0
        self.mesh_size = len(self.instances)

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

    def __repr__(self) -> str:
        rs = [f"[{i}] {str(s)}" for i, s in enumerate(self.instances)]
        rs += [LINE]
        rs += [f"Num Instances: {len(self.instances)}"]
        rs += [LINE]
        return "\n".join(rs)


class KBENCH_MODE(Enum):
    RUN = 0x1
    TUNE = 0x2
    BUILD = 0x4


def _get_tmp_path(file_path):
    base = os.path.basename(file_path).split(".")[0]
    tf = tempfile.NamedTemporaryFile(prefix=str(base) + "_").name + "/"
    return Path(tf)


def run(
    yaml_path_list,
    yaml_rewrite_path=None,
    output_path=None,
    mode=KBENCH_MODE.RUN,
    param_list=None,
    filter_list=None,
    dryrun=False,
    verbose=False,
    tmp_path=None,
):
    # Load specs from a list of YAML files and join them in 'spec'.
    assert len(yaml_path_list), "There should be at least 1 YAML as input."
    spec = Spec.load_yaml_list(yaml_path_list)

    # Expand with CLI params
    if param_list:
        spec.extend_params(param_list)

    # Apply the filters, if any.
    if filter_list:
        spec.filter(filter_list)

    # Rewrite the specs to yaml_rewrite_path, if any defined.
    if yaml_rewrite_path:
        spec.dump_yaml(Path(yaml_rewrite_path))

    if verbose:
        print(spec)

    output_path_list: Dict[int, Path] = {}
    spec_list: Dict[int, SpecInstance] = {}
    elapsed_time_list: Dict[int, float] = {}

    # Generate a tmp path for intermediate results.
    if not tmp_path:
        tmp_path = _get_tmp_path(spec.file)
    tmp_dir = Path(tmp_path)

    build_opts = ["build"] if mode == KBENCH_MODE.BUILD else []

    # Run the code over the mesh of param/values
    t_start_total = time()
    with Progress(
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        console=CONSOLE,
    ) as progress:
        bench_progress = progress.add_task(
            spec.name,
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
                t_start_item = time()

                s.compile(
                    output_file=output_file,
                    build_opts=build_opts,
                    dryrun=dryrun,
                    verbose=verbose,
                )
                elapsed_time_list[i] = (time() - t_start_item) * 1e3
                spec_list[i] = s
                output_path_list[i] = output_file

            except Exception as e:
                if e == KeyboardInterrupt:
                    sys.exit(0)
                else:
                    print(e)

            # When a benchmark is completed for one combination of parameters we advance progress by 1
            progress.update(bench_progress, advance=1)

    t_elapsed_total = time() - t_start_total
    output_lines = []
    ########################################################
    # Elapsed time per spec
    build_df = pd.DataFrame(
        {"name": [f"build/{str(s)}" for i, s in enumerate(spec)]}
    )
    build_df.insert(len(build_df.columns), "met (ms)", elapsed_time_list)
    build_df.insert(len(build_df.columns), "iters", 1)
    build_df["met (ms)"] = build_df["met (ms)"].fillna(0)

    output_lines += [LINE]
    output_lines += ["Build time stats:"]
    output_lines += [build_df.to_string(index=False)]

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

    output_lines += [LINE]
    output_lines += [f"Running [{spec.name}] from [{spec.file}]"]
    output_lines += [LINE]
    output_lines += [f"Number of valid executed specs: {len(valid_specs)}"]

    if valid_specs:
        merged_df = pd.concat(valid_specs, axis=0, ignore_index=True)
        ########################################################
        # Get the name of column 2 (met (ms))
        met_col = merged_df.columns[2]

        if mode == KBENCH_MODE.TUNE:
            sorted_df = merged_df.sort_values([met_col], ascending=True)
            output_lines += [sorted_df.to_string(index=False)]
            # Index to top spec after sort
            top_spec_idx = sorted_df.iloc[0].mesh_idx

            output_lines += [LINE]
            output_lines += ["Spec with the best measured time:"]
            output_lines += [LINE]
            output_lines += [f"mesh_idx: {top_spec_idx}"]
            output_lines += [spec_list[top_spec_idx].to_obj()]
            output_lines += [LINE]
        else:
            output_lines += [merged_df.to_string(index=False)]
            output_lines += [LINE]
        ########################################################

    output_lines += ["Total elapsed running time: %.3f (s)" % (t_elapsed_total)]
    output_str = "\n".join([str(x) for x in output_lines])
    print(output_str)

    if output_path:
        # KBENCH_MODE.RUN overrides everything else and just dumps the running results.
        # THIS IS CRITICAL FOR CI automated kernel benchmarks workflow.
        if mode == KBENCH_MODE.RUN and valid_specs:
            merged_df.drop(columns=["mesh_idx"]).to_csv(
                output_path, index=False, quoting=csv.QUOTE_NONNUMERIC
            )
        elif mode == KBENCH_MODE.BUILD:
            build_df.to_csv(
                output_path, index=False, quoting=csv.QUOTE_NONNUMERIC
            )
        else:
            with open(output_path, "w") as f:
                f.write(output_str + "\n")

        print(f"wrote results to [{output_path}]")


def check_gpu_clock():
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return
    output = run_shell_command(
        [nvidia_smi, "--query-gpu", "persistence_mode", "--format", "csv"],
        check=False,
        capture_output=True,
    )

    # We check for persistence here as a proxy to check if setup-gpu-benchmarking
    # has been run. This is not exact, but should cover most cases. Checking for
    # the clock frequency is more complicated since the frequencies changes per
    # GPU.
    if "Disabled" in output.stdout.decode("utf-8"):
        raise Exception(
            "the clock frequency for the GPU is not locked, please use"
            " `setup-gpu-benchmarking` to ensure that the frequencies and power"
            " of the GPU are locked to get consistent benchmarking behavior."
        )


help_str = (
    "Grid-search all the params for a mojo benchmark and pick the top value"
)


@click.command(help=help_str, no_args_is_help=True)
@click.option(
    "--yaml",
    "yaml_path",
    help="Path to a config yaml (can have multiple ones).",
    multiple=True,
)
@click.option(
    "--yaml-rewrite",
    "yaml_rewrite_path",
    help="Path to a rewrite the valid specs as a YAML file",
)
@click.option(
    "--filter",
    "filter",
    help=(
        "Define a single filter (should match a valid paramter, can have"
        " multiple ones). The filters should of the format `--filter"
        " PARAM=VALUE`, that is, the subset of parameters that satisfy this"
        " condition will be included."
    ),
    multiple=True,
)
@click.option(
    "--output", "-o", "output_path", default=None, help="Path to output file."
)
@click.option(
    "--tune",
    "-t",
    "tune",
    is_flag=True,
    default=False,
    help="Tune by running and finding the best running time.",
)
@click.option(
    "--build",
    "build",
    is_flag=True,
    default=False,
    help="Just measure the build time.",
)
@click.option(
    "--param", default=(), help="Set extra params from CLI.", multiple=True
)
@click.option("--force", "-f", is_flag=True, default=False, help="Force.")
@click.option(
    "--dryrun",
    "-dryrun",
    is_flag=True,
    default=False,
    help="Do not execute the config, just show the parameters.",
)
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Verbose printing."
)
def cli(
    yaml_path,
    yaml_rewrite_path,
    filter,
    output_path,
    tune,
    build,
    param,
    force,
    dryrun,
    verbose,
) -> bool:
    if not verbose:
        sys.tracebacklimit = 1

    mode = KBENCH_MODE.RUN

    assert (build == False) or (tune == False)
    if build:
        mode = KBENCH_MODE.BUILD
    elif tune:
        mode = KBENCH_MODE.TUNE

    check_gpu_clock()

    run(
        yaml_path_list=yaml_path,
        yaml_rewrite_path=yaml_rewrite_path,
        output_path=output_path,
        mode=mode,
        param_list=param,
        filter_list=filter,
        dryrun=dryrun,
        verbose=verbose,
    )


def main():
    cli()


if __name__ == "__main__":
    main()
