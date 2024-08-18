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

##===----------------------------------------------------------------------===##
# Mojo build helper
##===----------------------------------------------------------------------===##

source $MODULAR_PATH/utils/start-modular.sh

b() {
    if [ "$1" = "kgen" ]; then
        br kgen-tool
    elif [ "$1" = "check-kgen" ]; then
        bt KGEN/test/kgen/...
    elif [ "$1" = "check-mojo-integration" ]; then
        bt KGEN/test/mojo-integration/...
    elif [ "$1" = "check-mojo-isolated" ]; then
        bt KGEN/test/mojo-isolated/...
    elif [ "$1" = "check-mojo-parser" ]; then
        bt KGEN/test/mojo-isolated/... KGEN/test/mojo-parser/...
    elif [ "$1" = "check-mojo" ]; then
        bt KGEN/test/...
    elif [ "$1" = "check-genericml" ]; then
        bt GenericML/test/... GenericML/unittests/...
    elif [ "$1" = "check-graph-compiler-integration" ]; then
        bt GenericML/graph-compiler/integration-test/...
    else
        br "$1"
    fi
}

c() {
    if [ "$1" = "default" ]; then
        echo "" > ./local.bazelrc
    elif [ "$1" = "production" ]; then
        echo "build --config=production" > ./local.bazelrc
    elif [ "$1" = "release" ]; then
        echo "build --config=release" > ./local.bazelrc
    elif [ "$1" = "asan" ]; then
        echo "build --config=asan" > ./local.bazelrc
    elif [ "$1" = "relwithdebinfo" ]; then
        echo "build:relwithdebinfo --cc_output_directory_tag=relwithdebinfo" > ./local.bazelrc
        echo "build:relwithdebinfo --compilation_mode=opt" >> ./local.bazelrc
        echo "build:relwithdebinfo --copt=-O3" >> ./local.bazelrc
        echo "build:relwithdebinfo --copt=-g" >> ./local.bazelrc
        echo "build:relwithdebinfo --strip=always" >> ./local.bazelrc
    elif [ "$1" = "relwithdebinfo-modular" ]; then
        echo "build:relwithdebinfo-modular --cc_output_directory_tag=relwithdebinfo-modular" > ./local.bazelrc
        echo "build:relwithdebinfo-modular --compilation_mode=opt" >> ./local.bazelrc
        echo "build:relwithdebinfo-modular --copt=-O3" >> ./local.bazelrc
        echo "build:relwithdebinfo-modular --copt=-g" >> ./local.bazelrc
        echo "build:relwithdebinfo-modular --strip=always" >> ./local.bazelrc
        echo "build:relwithdebinfo-modular --per_file_copt=external/llvm-project/.*@-g0" >> ./local.bazelrc
    else
        echo "invalid build config specified"
    fi
}
