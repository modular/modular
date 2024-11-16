# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
import csv
import functools
import glob
import operator
import os
import shutil
import string
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from itertools import chain, product
from pathlib import Path
from time import sleep, time
from typing import Any, Dict, List, Optional, Set, Union

import click
import numpy as np
import pandas as pd
import rich
from model.utils.common_cli_options import CommonOptions
from model.utils.exceptions import CLIException, pretty_exception_handler
from model.utils.logging import CONSOLE
from rich import print, traceback
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import MofNCompleteColumn, Progress

from modular.utils import logging, yaml
from modular.utils.subprocess import list2cmdline, run_shell_command
from modular.utils.yaml import YAML

CONSOLE = Console()


CURRENT_FILE = Path(__file__).resolve()
LINE = 80 * "-"


def configure_logging(
    quiet: bool = False, verbose: bool = False, pretty_output: bool = True
) -> Console:
    global CONSOLE
    if pretty_output:
        debug_handler = RichHandler(
            show_path=False, show_time=False, console=CONSOLE
        )
        logging.basicConfig(format="%(message)s", handlers=[debug_handler])
    else:
        logging.basicConfig(format="%(levelname)s: %(message)s")
        CONSOLE = Console(force_terminal=False, color_system=None)

    if verbose:
        # Set logging level for this module's logger instance
        logging.getLogger().setLevel(logging.DEBUG)

        logging.debug("Enabled Verbose logging")
        if pretty_output:
            # Add the name of any module that we want to exclude
            # from traceback. This will reduce noise.
            traceback.install(suppress=[click, yaml, rich])
    else:
        # Set logging level for this module's logger instance
        level = logging.WARNING if quiet else logging.INFO
        logging.getLogger().setLevel(level)
        if pretty_output:
            sys.excepthook = pretty_exception_handler
    return CONSOLE


@dataclass(repr=True)
class Param:
    name: str
    value: object

    def define(self) -> list[str]:
        if self.name.startswith("$"):
            var_name = self.name.split("$")[1]
            return [f"--{var_name}={self.value}"]
        return ["-D", f"{self.name}={self.value}"]


def flatten(value: Union[int, object, Iterable]) -> List[Any]:
    """Flattens an iterable into a list.

    Supports nested lists, dictionaries, and tuples.

    Args:
        value: The iterable to flatten.

    Returns:
        A list of flattened values.
    """
    if not isinstance(value, Iterable) or isinstance(value, str):
        return [value]

    result = []
    for item in value:
        result.extend(flatten(item))
    return result


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


@dataclass(repr=True)
class ProcessOutput:
    stdout: Optional[str] = None
    stderr: Optional[str] = None


class KBENCH_MODE(Enum):
    RUN = 0x1
    TUNE = 0x2
    BUILD = 0x4


KBENCH_PARAM_CACHE: np.array = None


@dataclass(frozen=True, repr=True)
class SpecInstance:
    name: str
    file: Path
    executor: Optional[str] = None
    params: List[Param] = field(default_factory=list)

    @functools.cached_property
    def mojo_binary(self) -> str:
        mojo = shutil.which("mojo")
        if not mojo:
            raise FileNotFoundError(f"Could not find the `mojo` binary.")
        return mojo

    def get_executor(self) -> list[str]:
        if self.executor:
            return self.executor.split(" ")

        return [self.mojo_binary]

    def compile(
        self,
        *,
        output_file: Optional[Path] = None,
        build_opts: List[str] = [],
        mode: KBENCH_MODE,
        dryrun: bool = False,
        verbose: bool = False,
        check_kbench_cache: bool = False,
    ) -> ProcessOutput:
        if not output_file:
            output_file = Path(tempfile.gettempdir()) / Path(
                next(tempfile._get_candidate_names())  # type: ignore
            )

        # substitute env variables in the path
        file_abs_path = Path(
            string.Template(str(self.file)).substitute(os.environ)
        ).absolute()

        assert file_abs_path.exists()

        defines = []
        vars = []

        for param in self.params:
            if param.name.startswith("$"):
                vars += [param.define()]
            else:
                defines += [param.define()]

        defines = list(np.array(defines).flatten())
        vars = list(np.array(vars).flatten())

        if not check_kbench_cache:
            cmd = self.get_executor()
            if build_opts:
                cmd.extend(build_opts)
            cmd.extend(
                [
                    *defines,
                    str(file_abs_path),
                ]
            )
            if not mode == KBENCH_MODE.BUILD:
                cmd.extend(["-o", str(output_file), *vars])

        if check_kbench_cache:
            cmd = []

            global KBENCH_PARAM_CACHE
            found_in_cache = np.array_equal(
                np.array(defines), np.array(KBENCH_PARAM_CACHE)
            )
            if not found_in_cache:
                KBENCH_PARAM_CACHE = np.array(defines)
                cmd = self.get_executor()
                cmd.extend(["build", *defines, str(file_abs_path)])
                # TODO: how to handle the return from failing here?
                self._run(cmd, output_file, verbose, dryrun)

            # at this point the previous build is in the cache, a binary of the same name (without the extension) is created:
            binary_name = str(file_abs_path.with_suffix(""))
            cmd = [binary_name, "-o", str(output_file), *vars]

        return self._run(cmd, output_file, verbose, dryrun)
        # if mode == KBENCH_MODE.BUILD:
        # TODO: refactor the following into a separate function call, or invoke alias directly.
        # mojo-clear-cache
        # subprocess.call(
        #     (
        #         "rm -fr $MODULAR_DERIVED_PATH/.mojo_cache"
        #         " $HOME/.modular/.mojo_cache"
        #     ),
        #     shell=True,
        # )

    def _run(self, cmd, output_file, verbose, dryrun):
        if verbose:
            logging.info(f"[output_file: {output_file}")
        try:
            if dryrun:
                if verbose:
                    print(list2cmdline(cmd))
            else:
                # TODO: needs better error handling and error messages.
                logging.debug(list2cmdline(cmd))
                output = run_shell_command(
                    cmd, check=False, capture_output=True
                )
                return ProcessOutput(
                    output.stdout.decode("utf-8"), output.stderr.decode("utf-8")
                )

        except Exception as exc:
            raise CLIException(
                f"Unable to run the command {list2cmdline(cmd)}"
            ) from exc

        return ProcessOutput(None, None)

    def to_obj(self) -> dict[str, object]:
        obj = {}
        for param in self.params:
            obj[param.name] = param.value
        return obj

    def __str__(self) -> str:
        tokens = [self.name]
        for param in self.params:
            tokens.append(f"{param.name}={param.value}")
        return "/".join(tokens)


class GridSearchStrategy:
    def __init__(self, name, file, params):
        self.instances = []

        # params
        for cfg in params:
            name_list = [p.name for p in cfg]
            param_list = [p.value_set for p in cfg]
            param_mesh = list(product(*param_list))
            num_params = len(cfg)
            for idx in range(len(param_mesh)):
                s = SpecInstance(
                    name=name,
                    file=file,
                    params=[
                        Param(name=name_list[i], value=param_mesh[idx][i])
                        for i in range(num_params)
                    ],
                )
                self.instances.append(s)

    def __iter__(self):
        self.offset = 0
        return self

    def __next__(self):
        # Stop condition
        if self.offset == len(self.instances):
            raise StopIteration

        res = self.instances[self.offset]
        self.offset += 1
        return res

    def __getitem__(self, i):
        return self.instances[i]

    def __len__(self):
        return len(self.instances)

    def extend(self, other):
        self.instances.extend(other.instances)


@dataclass(repr=True)
class Spec:
    name: str
    file: Path
    params: list[object] = field(default_factory=list)
    mesh_idx: int = 0
    mesh_size: int = 0
    mesh: List[SpecInstance] = field(default_factory=list)

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
            raise ValueError(f"Could not load spec from {file}")

    @staticmethod
    def load_yaml_list(yaml_path_list: List[str]) -> "Spec":
        spec: "Spec" = None  # type: ignore
        for i, yaml_path in enumerate(yaml_path_list):
            spec_ld = Spec.load_yaml(Path(yaml_path))
            if i == 0:
                spec = spec_ld
            else:
                spec.join(spec_ld)
        return spec

    @staticmethod
    def parse_params(param_list: list[str]):
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
            name = ""
            val = ""
            if IFS in p:
                name, val = p.split(IFS)

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

        # For all params in each config either, update the exisiting `value_set``
        # with the new param value(s).
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
        assert self.mesh, "There are no instances to write to YAML!"
        obj = {
            "name": self.name,
            "file": self.file,
            "params": [s.to_obj() for s in self.mesh],
        }
        with open(out_path, "w") as f:
            YAML(typ="safe").dump(obj, f, sort=False)
        logging.debug(f"dumped {len(self.mesh)} instances to [{out_path}]")

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
        assert self.mesh
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
        self.mesh = GridSearchStrategy(self.name, self.file, self.params)

        return len(self.mesh)

    def join(self, other: "Spec"):
        assert self.name == other.name
        assert self.file == other.file
        assert other.mesh_size > 0

        self.mesh_idx = 0
        self.mesh_size += other.mesh_size
        self.params.extend(other.params)
        self.mesh.extend(other.mesh)

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
        valid_cnt = np.zeros(len(self.mesh), dtype=np.int32)

        for k_filter, v_filter in filters.items():
            for i, s in enumerate(self.mesh):
                for p in s.params:
                    if p.name == k_filter and str(p.value) in v_filter:
                        valid_cnt[i] += 1

        for i, idx in enumerate(valid_cnt):
            if idx == num_filters:
                filtered_insts.append(self.mesh[i])

        self.mesh.instances = filtered_insts[:]
        self.mesh_idx = 0
        self.mesh_size = len(self.mesh)

    def __iter__(self):
        self.iter_offset = 0
        return self

    def __next__(self) -> "SpecInstance":
        assert (
            self.mesh != None
        ), "Should call self.init_mesh after loading or in postinit."

        # Stop condition
        if self.iter_offset == len(self.mesh):
            raise StopIteration

        # Retrieve and update self.mesh_idx
        idx = self.iter_offset
        self.iter_offset += 1
        return self.mesh[idx]

    def __repr__(self) -> str:
        rs = [f"[{i}] {str(s)}" for i, s in enumerate(self.mesh)]
        rs += [LINE]
        rs += [f"Num Instances: {len(self.mesh)}"]
        rs += [LINE]
        return "\n".join(rs)


def _get_tmp_path(file_path):
    base = os.path.basename(file_path).split(".")[0]
    tf = tempfile.NamedTemporaryFile(prefix=str(base) + "_").name + "/"
    return Path(tf)


def run(
    yaml_path_list,
    output_path=None,
    mode=KBENCH_MODE.RUN,
    param_list=None,
    filter_list=None,
    debug_level=None,
    optimization_level=None,
    dryrun=False,
    cached=False,
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

    if verbose:
        print(spec)

    output_path_list: Dict[int, Path] = {}
    output_msg_list: Dict[int, ProcessOutput] = {}
    spec_list: Dict[int, SpecInstance] = {}
    elapsed_time_list: Dict[int, float] = {}

    # Generate a tmp path for intermediate results.
    if not tmp_path:
        tmp_path = _get_tmp_path(spec.file)
    tmp_dir = Path(tmp_path)

    build_opts = []
    if mode is KBENCH_MODE.BUILD:
        build_opts = ["build"]
    if debug_level:
        build_opts.extend(["--debug-level", debug_level])
    if optimization_level:
        build_opts.extend(["-O", optimization_level])

    # Run the code over the mesh of param/values
    t_start_total = time()
    with Progress(
        *Progress.get_default_columns(),
        MofNCompleteColumn(),
        console=CONSOLE,
        expand=True,
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

            progress.update(bench_progress, description=s)

            # Check for the failure here.
            try:
                output_file = output_dir / "output.csv"
                t_start_item = time()

                output_msg = s.compile(
                    output_file=output_file,
                    build_opts=build_opts,
                    mode=mode,
                    dryrun=dryrun,
                    verbose=verbose,
                    check_kbench_cache=cached,
                )
                elapsed_time_list[i] = (time() - t_start_item) * 1e3
                spec_list[i] = s
                output_path_list[i] = output_file
                output_msg_list[i] = output_msg

            except Exception as e:
                if e == KeyboardInterrupt:
                    sys.exit(0)
                else:
                    raise e

            # When a benchmark is completed for one combination of parameters
            # we advance progress by 1
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

    if verbose:
        output_lines += [LINE]
        output_lines += ["Build time stats:"]
        output_lines += [build_df.to_string(index=False)]

    ########################################################
    # Retrieve, sort, and pick top choices
    valid_specs = []
    invalid_specs = []
    for i in spec_list.keys():
        try:
            df = pd.read_csv(output_path_list[i], index_col=None, header=0)
            df.insert(0, "mesh_idx", i)
            df.insert(len(df.columns), "spec", str(spec.mesh[i]))
            valid_specs.append(df)

        except:
            invalid_specs.append([i, output_msg_list[i]])

    output_lines += [LINE]
    output_lines += [f"Running {spec.name} from '{spec.file}'"]

    if invalid_specs:
        output_lines += [LINE]
        output_lines += [f"Number of invalid specs: {len(invalid_specs)}"]
        for idx, msg in invalid_specs:
            output_lines += [LINE]
            output_lines += [f"mesh_idx: [{idx}][{spec_list[idx].to_obj()}]"]
            if msg.stdout:
                output_lines.append(msg.stdout)
            if msg.stderr:
                output_lines.append(msg.stderr)

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
            runtime = sorted_df.iloc[0][met_col]

            output_lines += [LINE]
            output_lines += ["Spec with the best measured time:"]
            output_lines += [LINE]
            output_lines += [f"mesh_idx: {top_spec_idx} Runtime: {runtime}"]
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


@functools.cache
def get_nvidia_smi():
    return shutil.which("nvidia-smi")


def reset_gpu():
    nvidia_smi = get_nvidia_smi()
    if not nvidia_smi:
        return
    run_shell_command([nvidia_smi, "-r"])


def check_gpu_clock():
    nvidia_smi = get_nvidia_smi()
    if not nvidia_smi:
        return
    output = run_shell_command(
        [
            nvidia_smi,
            "--query-gpu",
            "persistence_mode",
            "--format",
            "csv",
        ],
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


class FileGlobArg:
    def __init__(self, file: List[str]) -> None:
        self._files = file
        if not self._files:
            raise ValueError(
                f"Could not find any file that satisfies the glob {file}."
                " Check to ensure that the file exist or the glob matches one"
                " or more file."
            )

    def __iter__(self):
        return (Path(file).resolve() for file in self._files)

    def __len__(self):
        return len(self._files)


help_str = (
    "Grid-search all the params for a mojo benchmark and pick the top value"
)


@click.command(help=help_str, no_args_is_help=True)
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
@click.option(
    "--debug-level", default=None, help="The debug level used during the build."
)
@click.option(
    "-O",
    "--optimization-level",
    default=None,
    help="The optimization level used during the build.",
)
@click.option("--force", "-f", is_flag=True, default=False, help="Force.")
@click.option(
    "--cached", is_flag=True, default=False, help="Enable Kbench cache."
)
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
@click.argument("files", nargs=-1, type=click.UNPROCESSED)
def cli(
    files: click.UNPROCESSED,
    filter,
    output_path,
    tune,
    build,
    param,
    debug_level,
    optimization_level,
    force,
    cached,
    dryrun,
    verbose,
) -> bool:
    configure_logging(verbose=verbose)

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
        yaml_path_list=FileGlobArg(files),
        output_path=output_path,
        mode=mode,
        param_list=param,
        filter_list=filter,
        debug_level=debug_level,
        optimization_level=optimization_level,
        dryrun=dryrun,
        cached=cached,
        verbose=verbose,
    )
    return True


def main():
    try:
        cli()
    except Exception:
        CONSOLE.print_exception(suppress=[click, yaml, rich])


if __name__ == "__main__":
    main()
