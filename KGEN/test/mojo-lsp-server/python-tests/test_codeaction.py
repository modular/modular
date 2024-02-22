# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lib.utils import Document, Requests, fail_if_none, mojo_lsp_client
from lsprotocol.types import Diagnostic, DiagnosticSeverity, Position, Range
from pytest_lsp import LanguageClient

# This variable is required by the test runner.
# pyright: reportUnknownVariableType=false
client = mojo_lsp_client


async def test_codeaction_generate_documentation(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        '''""""""

fn function[value: Int](self: Int) -> Int:
    """"""
    return 10

struct EmptyStruct:
    """"""
    ...

struct ParameterStruct[value: Int]:
    """"""
    ...
''',
    )
    requests = Requests(client)
    requests.open_document(doc)

    # Build an expected diagnostic for a missing doc string given a start line
    # and position.
    build_expected_diag = lambda line, start_pos: Diagnostic(
        Range(Position(line, start_pos), Position(line, start_pos + 6)),
        "Unexpected empty documentation string",
        severity=DiagnosticSeverity.Warning,
        source="mojo",
    )
    actions = fail_if_none(
        await requests.code_action(
            doc,
            doc.get_full_range(),
            [
                build_expected_diag(0, 0),
                build_expected_diag(3, 4),
                build_expected_diag(7, 4),
                build_expected_diag(11, 4),
            ],
        )
    )
    assert len(actions) == 4

    # Check the resultant template for a given action index.
    def check_template(idx: int, expected: str):
        action = actions[idx]
        assert (
            action.title == "Generate documentation"
            and action.edit
            and len(action.edit.changes) == 1
            and list(action.edit.changes.values())[0][0].new_text == expected
        )

    # Module template.
    check_template(0, "[summary].")

    # Function template.
    check_template(
        1,
        """[summary].

    Parameters:
        value: [description].

    Args:
        self: [description].

    Returns:
        [description].
    """,
    )

    # Empty struct template.
    check_template(2, """[summary].""")

    # Parameterized struct template.
    check_template(
        3,
        """[summary].

    Parameters:
        value: [description].
    """,
    )
