"Example macro wrapping the mocha CLI"

load("@mojo-lsp-server-node-tests//KGEN/test/mojo-lsp-server-node:mocha/package_json.bzl", "bin")

def mocha_test(name, srcs, args = [], data = [], env = {}, **kwargs):
    bin.mocha_test(
        name = name,
        args = [
            "--reporter",
            "mocha-junit-reporter",
            native.package_name() + "/*.spec.js",
        ] + args,
        data = data + srcs + [
            "//KGEN/test/mojo-lsp-server-node:node_modules/mocha-junit-reporter",
        ],
        env = env | {
            # Add environment variable so that mocha writes its test xml
            # to the location Bazel expects.
            "MOCHA_FILE": "$$XML_OUTPUT_FILE",
        },
        **kwargs
    )
