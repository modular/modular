"Example macro wrapping the mocha CLI"

load("@mojo-lsp-server-node-tests//KGEN/test/mojo-lsp-server-node:mocha/package_json.bzl", "bin")
load("//bazel/internal:mojo_test_environment.bzl", "mojo_test_environment")  # buildifier: disable=bzl-visibility

def mocha_test(name, srcs, args = [], data = [], env = {}, **kwargs):
    mojo_test_environment(name = "mojo_test_env", data = ["@mojo//:stdlib"], testonly = True)
    mojo_libs = [
        "//AsyncRT:RuntimeGlobals",
        "//AsyncRT:DeviceContext",
        "//Support:Globals",
    ]

    bin.mocha_test(
        name = name,
        args = [
            "--reporter",
            "mocha-multi-reporters",
            "--reporter-options",
            "configFile=$(location //KGEN/test/mojo-lsp-server-node:mocha-reporters.json)",
            "--config=$(location //KGEN/test/mojo-lsp-server-node:.mocharc.json)",
            native.package_name() + "/*.spec.js",
        ] + args,
        data = data + srcs + [
            "//KGEN/test/mojo-lsp-server-node:node_modules/mocha-junit-reporter",
            "//KGEN/test/mojo-lsp-server-node:node_modules/mocha-multi-reporters",
            "//KGEN/test/mojo-lsp-server-node:mocha-reporters.json",
            "//KGEN/test/mojo-lsp-server-node:.mocharc.json",
            "@mojo//:stdlib",
        ] + mojo_libs,
        env = env | {
            # Add environment variable so that mocha writes its test xml
            # to the location Bazel expects.
            "MOCHA_FILE": "$$XML_OUTPUT_FILE",
            "MODULAR_MOJO_MAX_IMPORT_PATH": "$(COMPUTED_IMPORT_PATH)",
            "MODULAR_MOJO_MAX_SHARED_LIBS": "$(COMPUTED_LIBS)",
        },
        toolchains = [":mojo_test_env"],
        **kwargs
    )
