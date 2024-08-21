#!/bin/echo This script should be run as: source
##===----------------------------------------------------------------------===##
#
# This file is Modular Inc proprietary.
#
##===----------------------------------------------------------------------===##

# NOTE: This is copied from utils/start-modular.sh
if [ -n "$MODULAR_HOME" ]; then
  echo "The MODULAR_HOME environment variable is incompatible with the monorepo. Please unset MODULAR_HOME and try again."
  return 1
fi

if [ -z "$_START_MODULAR_INCLUDED" ]; then
  export _START_MODULAR_INCLUDED=yes
  export PS1="[M] $PS1"
fi

if [ -n "$ZSH_VERSION" ]; then
  # shellcheck disable=SC2296
  CUR_DIR=${(%):-%N}
elif [ -n "$BASH_VERSION" ]; then
  CUR_DIR=${BASH_SOURCE[0]}
fi

if [ -n "$MODULAR_VENV_PATH" ]; then
  if [ ! -d "$MODULAR_VENV_PATH/bin" ]; then
    python3 -m venv $MODULAR_VENV_PATH
  fi
  source $MODULAR_VENV_PATH/bin/activate
fi

if [ -n "$DOCKER_MODULAR_DIR" ]; then
  export MODULAR_PATH=$DOCKER_MODULAR_DIR
else
  export MODULAR_PATH=$(python3 -c "from pathlib import Path; print(Path('$CUR_DIR').resolve().parent.parent)")
fi

source "$MODULAR_PATH/utils/start-modular.sh"

##===----------------------------------------------------------------------===##
# Mojo build helper
##===----------------------------------------------------------------------===##

BR="$MODULAR_PATH/bazelw run"
BT="$MODULAR_PATH/bazelw test"
BB="$MODULAR_PATH/bazelw build"

get_build_alias() {
    if [ "$1" = "kgen" ]; then
        echo "kgen-tool"
    elif [ "$1" = "check-kgen" ]; then
        echo "KGEN/test/kgen/..."
    elif [ "$1" = "check-mojo-integration" ]; then
        echo "KGEN/test/mojo-integration/..."
    elif [ "$1" = "check-mojo-isolated" ]; then
        echo "KGEN/test/mojo-isolated/..."
    elif [ "$1" = "check-mojo-parser" ]; then
        echo "KGEN/test/mojo-isolated/... KGEN/test/mojo-parser/..."
    elif [ "$1" = "check-mojo" ]; then
        echo "KGEN/test/..."
    elif [ "$1" = "check-genericml" ]; then
        echo "GenericML/test/... GenericML/unittests/..."
    elif [ "$1" = "check-graph-compiler-integration" ]; then
        echo "GenericML/graph-compiler/integration-test/..."
    fi
}

b() {
    test_targets=()
    run_targets=()
    for target in $@; do
        als=$(get_build_alias $target)
        if [ -z "$als" ]; then
            als="$target"
        fi
        if [[ "$target" == "check-"* ]]; then
            test_targets+=("$als")
        else
            run_targets+=("$als")
        fi
    done

    if [ "${#test_targets[@]}" -ne 0 ]; then
        test_targets=$(IFS=" "; echo "${test_targets[*]}")
        eval "$BT $test_targets"
    fi
    if [ "${#run_targets[@]}" -ne 0 ]; then
        run_targets_str=$(IFS=" "; echo "${run_targets[*]}")
        eval "$BB $run_targets_str"
        for target in "${run_targets[@]}"; do
            eval "$BR $target"
        done
    fi
}

bench() {
    cmd="$@"
    hyperfine --prepare='rm -rf .derived/.mojo_cache' "$cmd" --warmup 3
}
