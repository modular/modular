# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import argv, exit


fn print_help():
    print(
        """NAME
        mojo-build-server — A build server for Mojo projects.

SYNOPSIS
        mojo-build-server [options]

DESCRIPTION
        A server that communicates via JSON-RPC in order to build
        Mojo projects.

OPTIONS
    Debugging options
        --debug
            Use delimited text for JSON input to the server, and
            print pretty JSON output from the server, for
            debugging and testing purposes.

    Common options
        --help, -h
            Displays help information.
"""
    )


fn main():
    # Parse command line arguments. If `-h` or `--help` appears anywhere in the
    # argument list, print help text and exit. Otherwise, reject unknown
    # arguments.
    var args = argv()
    var help = False
    var debug = False
    var unrecognized = List[StringRef]()
    for i in range(1, len(args)):
        var a = args[i]
        if a == "-h" or a == "--help":
            help = True
        elif a == "--debug":
            debug = True
        else:
            unrecognized.append(a)

    if help:
        print_help()
        exit(0)

    if len(unrecognized) > 0:
        for i in range(len(unrecognized)):
            print("mojo-build-server: error: unrecognized argument '", end="")
            print(unrecognized[i], end="")
            print("'")
        exit(1)

    # Launch the server by calling into the MojoBuild library.
    exit(external_call["mojoBuildServerMain", Int](debug))
