# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from sys import argv, exit


fn print_help():
    print("NAME")
    print("        mojo-build-server — A build server for Mojo projects.")
    print("")
    print("SYNOPSIS")
    print("        mojo-build-server [options]")
    print("")
    print("DESCRIPTION")
    print("        A server that communicates via JSON-RPC in order to build")
    print("        Mojo projects.")
    print("")
    print("OPTIONS")
    print("    Common options")
    print("        --help, -h")
    print("            Displays help information.")
    print("")


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
    exit(external_call["mojoBuildServerMain", Int]())
