# Mojo🔥 REPL

[TOC]

## Introduction

A Read Eval Print Loop, or REPL, is an effective tool for providing a powerful
interactive development experience. Mojo provides a powerful REPL experience
built on top of the [LLDB debugger](https://lldb.llvm.org/), which also provides
the debugging environment for the Mojo Language.

## Getting started

There are several entry points with which to experience the Mojo Repl, with the
main two being the [LLDB REPL](#lldb-command-line-repl), and a
[Jupyter Notebook](#jupyter-notebook). Each entry point contains specific setup
instructions, please refer to each section for more detailed information.

### LLDB Command Line REPL

In addition to providing the underpinning technology, LLDB can also be used as
a command line driver for interacting with the REPL. To start an interactive
REPL session within LLDB, Mojo provides a convenient utility with the necessary
setup:

```shell
# Build the Mojo REPL and all of the various dependencies.
$ build MojoLLDB

# Launch the REPL.
$ mojo-repl
```

Once run, you'll be provided with a REPL environment where you can immediately
start running expressions:

```shell
Welcome to Mojo.
Type :help for assistance.
  1>
```

### Jupyter Notebook

Jupyter notebooks are a common environment for interacting with REPLs of all
shapes and sizes. Mojo provides a custom kernel implementation for interacting
with the REPL in any jupyter environment.

```shell
# Ensure the Mojo Jupyter Kernel is installed in the local environment.
$ install_python_deps

# Build all of the necessary REPL functionality to run the jupyter kernel.
$ build MojoJupyter
```

#### VSCode Notebooks

VSCode provides a powerful suite of notebook functionality, which can be easily
integrated with the Mojo Kernel. To change the kernel within a notebook, simply
pick `Select Kernel` in the upper right of the notebook, and select Mojo.
Depending on your setup, you may need to find the kernel via:
 `> Select Another Kernel > Jupyter Kernel > Mojo`

#### JupyterLab Notebooks

JupyterLab is the latest web-based interactive development environment for
notebooks provided by the Jupyter Project. The kernel should be available
directly, but you may need to initialize Jupyter first if you haven't
already:

```shell
# Setup JupyterLab and the Mojo Jupyter extension.
$ jupyter-init

# Start a Jupyter server.
$ jupyter-lab
```

## Configuration

### Environment Variables
 * `MOJO_JUPYTER_LOG_FILE`: Setting this will cause the jupyter notebook kernel
     to log to the file specified. We recommend providing an absolute path
     here. If this is unspecified, the kernel simply logs to the stderr.

## Debugging Compiler Issues

Debugging compiler issues within the REPL environment is much different from
debugging issues with a single .mojo or .🔥 module. Given that the REPL executes
expressions across multiple invocations, it requires a different kind of mindset
when debugging a crash or miscompilation. This section contains useful tips and
tricks to make debugging issues within the REPL a bit easier.

### mojo-jupyter-executor

Jupyter notebooks generally involve a UI frontend component, with the backend
execution somewhat hidden and difficult to interact with outside of logs; not to
mention that the backend entry point is defined within python. This makes it
difficult to debug Mojo issues the traditional way, i.e. via a debugger. To make
this debugging flow a bit easier, Mojo provides a utility
`mojo-jupyter-executor` that can be used to execute a notebook in an environment
that is amenable to traditional debugging, e.g. via LLDB. To execute a notebook,
simply provide it to `mojo-jupyter-executor`. This will launch the Mojo Jupyter
kernel and execute each cell individually, as you would expect in a normal
jupyter environment.

```shell
$ mojo-jupyter-executor notebook.ipynb
```
