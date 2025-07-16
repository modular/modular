#!/usr/bin/env python3
# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

import argparse
import json
import os
import sys
from pathlib import Path

import jinja2


def addImplicitConversionDecorator(mojo_json) -> None:  # noqa: ANN001
    """Show @implicit on implicit constructors. We reuse the "convention"
    field which is also used for marking structs with the register-passable
    decorator. The isImplicitConversion flag should only appear on constructors,
    but check just in case."""
    for struct in mojo_json["structs"] + mojo_json["traits"]:
        for overload_set in struct["functions"]:
            for function in overload_set["overloads"]:
                if function["isImplicitConversion"]:
                    if function["name"] == "__init__":
                        function["convention"] = "@implicit"
                    else:
                        print(
                            f"Error: {struct['name']}.{function['name']} "
                            + "declared with @implicit but is not a constructor.",
                            file=sys.stderr,
                        )
                        exit(1)


def copyFieldTypesToValue(mojo_json) -> None:  # noqa: ANN001
    """The type of struct fields is expected by the doc generator to be in a
    field called 'value'."""
    for struct in mojo_json["structs"] + mojo_json["traits"]:
        for field in struct["fields"]:
            field["value"] = field["type"]


def processStructConvention(mojo_json) -> None:  # noqa: ANN001
    """We want to show the decorators for register-passable types; don't display
    anything for the default case (memory-only)."""
    for struct in mojo_json["structs"]:
        if "convention" in struct:
            if struct["convention"] == "register_passable":
                struct["convention"] = "@register_passable"
            elif struct["convention"] == "register_passable_trivial":
                struct["convention"] = "@register_passable(trivial)"
            elif struct["convention"] == "memory_only":
                del struct["convention"]
            else:
                print(
                    f"Unknown struct convention: {struct['convention']}",
                    file=sys.stderr,
                )
                exit(1)


def removeSelfArgumentFromStructMethods(mojo_json) -> None:  # noqa: ANN001
    """If we are in a non-static struct method, we don't want to show the first argument (self) as an
    argument in the documentation. So we remove it from all the functions that are child of structs.
    """
    for struct in mojo_json["structs"] + mojo_json["traits"]:
        for overload_set in struct["functions"]:
            for function in overload_set["overloads"]:
                if (
                    not function["isStatic"]
                    and function["args"]
                    and function["args"][0]["type"] == "Self"
                    and function["args"][0]["name"] == "self"
                ):
                    function["args"].pop(0)


def removeArgumentsWithoutDocumentation(mojo_json) -> None:  # noqa: ANN001
    """We've been omitting function arguments without documentation from docstring, so we remove them
    from top-level functions and struct methods."""

    def process_decl_with_functions(decl) -> None:  # noqa: ANN001
        for overloadSet in decl["functions"]:
            for function in overloadSet["overloads"]:
                function["args"] = [
                    arg for arg in function["args"] if arg["description"]
                ]

    for struct in mojo_json["structs"] + mojo_json["traits"]:
        process_decl_with_functions(struct)
    process_decl_with_functions(mojo_json)


def removeParametersWithoutDocumentation(mojo_json) -> None:  # noqa: ANN001
    """We've been omitting parameters without documentation from docstring, so we remove them
    from top-level functions, struct methods and structs"""

    def process_decl_with_parameters(decl) -> None:  # noqa: ANN001
        decl["parameters"] = [
            param for param in decl["parameters"] if param["description"]
        ]

    def process_decl_with_functions(decl) -> None:  # noqa: ANN001
        for overloadSet in decl["functions"]:
            for function in overloadSet["overloads"]:
                process_decl_with_parameters(function)

    for struct in mojo_json["structs"]:
        process_decl_with_functions(struct)
        process_decl_with_parameters(struct)

    for trait in mojo_json["traits"]:
        process_decl_with_functions(trait)

    process_decl_with_functions(mojo_json)


def removeStaticFromInitializers(mojo_json) -> None:  # noqa: ANN001
    """Removes 'isStatic' from struct initializer functions.

    The "isStatic" attribute is set to `true` for all `FnOp` for which
    `isStatic` is true. This is confusing for readers of the documentation, who
    would see that the `__init__` method of a struct with value semantics (that
    is, one decorated with `@value`) is "static," but a similar `__init__`
    method for a struct not decorated as such it not "static."

    At a high level, the concern is that users will not always understand why
    the LIT dialect treats certain functions as static or not, and we munge the
    data here to be simpler."""

    def process_decl_with_functions(decl) -> None:  # noqa: ANN001
        for overloadSet in decl["functions"]:
            for function in overloadSet["overloads"]:
                name = function["name"]
                isInitializer = name.startswith("__") and name.endswith(
                    "init__"
                )
                function["isStatic"] = (
                    function["isStatic"] and not isInitializer
                )

    for struct in mojo_json["structs"] + mojo_json["traits"]:
        process_decl_with_functions(struct)
    process_decl_with_functions(mojo_json)


# The slug is the name of the module, except for the index
# module, which is named "index_" to avoid a name conflict with
# the index file.
def nameToSlug(name):  # noqa: ANN001
    return "index_" if name == "index" else name


def generateMarkdown(
    mojo_json,  # noqa: ANN001
    version: str,
    output: Path,
    environment: jinja2.Environment,
    template: jinja2.Template,
    parent_json=None,  # noqa: ANN001
    is_nested=False,  # noqa: ANN001
    namespace=None,  # noqa: ANN001
) -> None:
    name = mojo_json["name"]

    # Skip private modules.
    if name != "__init__" and name.startswith("_"):
        return

    # If the json is a package, we recurse into the nested modules.
    if mojo_json["kind"] == "package":
        # If the package is nested, we need to add the package name to the
        # output path.
        if is_nested:
            output = output / name

        namespace = namespace + "." + name if namespace else name

        for module in mojo_json["modules"] + mojo_json["packages"]:
            generateMarkdown(
                module,
                version,
                output,
                environment,
                template,
                parent_json=mojo_json,
                is_nested=True,
                namespace=namespace,
            )
        return
    else:
        mojo_json["version"] = version
        mojo_json["slug"] = nameToSlug(mojo_json["name"])
        mojo_json["namespace"] = namespace

    # If its a module, we apply separate templates for struct/trait or function
    if mojo_json["kind"] == "module":
        for transformation in [
            addImplicitConversionDecorator,
            copyFieldTypesToValue,
            processStructConvention,
            removeParametersWithoutDocumentation,
            removeArgumentsWithoutDocumentation,
            removeSelfArgumentFromStructMethods,
            removeStaticFromInitializers,
        ]:
            transformation(mojo_json)

        # If we don't have an output path, we use the slug for the module.
        output = output / Path(mojo_json["slug"])
        struct_template = environment.get_template("mojodoc_struct.md")
        function_template = environment.get_template("mojodoc_function.md")

        namespace = namespace + "." + name if namespace else name

        # Save list of all struct names to compare to function names below
        struct_names = []
        for struct in mojo_json["structs"]:
            struct_names.append(struct["name"])
            generateMarkdown(
                struct,
                version,
                output,
                environment,
                struct_template,
                parent_json=mojo_json,
                is_nested=True,
                namespace=namespace,
            )

        for trait in mojo_json["traits"]:
            generateMarkdown(
                trait,
                version,
                output,
                environment,
                struct_template,
                parent_json=mojo_json,
                is_nested=True,
                namespace=namespace,
            )

        for function in mojo_json["functions"]:
            # Account for function names that match sibling struct names.
            # URL paths are case-sensitive but the macOS filesystem is not, so
            # create unique filenames for these functions so we don't clobber
            # files when building on mac. Also create unique filenames for any
            # functions called index or Index for similar reasons.
            function["filename"] = function["name"]
            if (
                function["name"].capitalize() in struct_names
                or function["name"].capitalize() == "Index"
            ):
                function["filename"] = function["name"] + "-function"
            generateMarkdown(
                function,
                version,
                output,
                environment,
                function_template,
                parent_json=mojo_json,
                is_nested=True,
                namespace=namespace,
            )

        # Handle the init module.
        if name == "__init__" and parent_json:
            # The init module is generated as the index file in the output
            # directory.
            output = output.with_name("index.md")

            # Add links to the public modules and packages in the parent.
            mojo_json["modules"] = [
                {
                    "name": module["name"],
                    "slug": nameToSlug(module["name"]),
                    "kind": "module_link",
                    "summary": module["summary"],
                }
                for module in parent_json["modules"]
                if not module["name"].startswith("_")
            ]
            mojo_json["packages"] = [
                {
                    "name": package["name"],
                    "kind": "package_link",
                    "summary": package["summary"],
                }
                for package in parent_json["packages"]
                if not package["name"].startswith("_")
            ]

            # We want to display the name of the parent module in the title.
            mojo_json["name"] = parent_json["name"]
        else:
            output = output / Path("index.md")
        mojo_json["slug"] = " "
    elif mojo_json["kind"] == "struct":
        output = output / Path(name + ".md")
    elif mojo_json["kind"] == "trait":
        output = output / Path(name + ".md")
    elif mojo_json["kind"] == "function":
        # Account for function names that match sibling struct names
        name = mojo_json["filename"]
        mojo_json["slug"] = name
        output = output / Path(name + ".md")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as output_file:
        # The first line must start with front matter, not mdlint comment
        markdown = template.render(decls=[mojo_json])
        lines = markdown.splitlines()
        if lines and "markdownlint-disable" in lines[0]:
            lines.pop(0)
        output_file.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", help="the input Mojo documentation json file name"
    )
    parser.add_argument("-o", "--output", type=Path, help="the output path")
    args = parser.parse_args()

    with open(args.filename) as jsonFile:
        template_dir = os.path.join(
            os.path.dirname(__file__), "mojodoc-templates"
        )
        environment = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        template = environment.get_template("mojodoc_module.md")
        docJson = json.load(jsonFile)

        version = docJson["version"]
        decl = docJson["decl"]
        generateMarkdown(decl, version, args.output, environment, template)
        os.remove(args.filename)


if __name__ == "__main__":
    main()
