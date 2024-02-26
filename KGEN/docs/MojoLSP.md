# Mojo🔥 LSP

## Introduction

The Mojo Language server is a productivity tool that enhances the authoring
experience of Mojo programs in editors that support the [Language Server
Protocol](https://en.wikipedia.org/wiki/Language_Server_Protocol). It provides
[editing features](https://code.visualstudio.com/api/language-extensions/programmatic-language-features),
such as code completion, diagnostics, quick fixes, hover dialogs, jump to
definition, refactoring utilities, etc.

## Getting Started with VSCode

Just run the `vscode-init` command on your terminal and the **Mojo LSP VSCode**
extension will be built and installed on VSCode. It will be automatically
launched whenever a `.mojo` or `.🔥` file is opened.

## Development

### Testing mojo-lsp-server

For testing, we are using the `pytest_lsp` python package, which has a set of
utilities that make interacting with LSP server and testing easier. However, its
official [documentation](https://swyddfa.github.io/lsp-devtools/docs/latest/en/pytest-lsp/guide/getting-started.html)
lacks enough documentation, which the rest of this document tries to make up
for.

#### Set Up

All test files require first setting up the connection with the
`mojo-lsp-server`, like in the following snippet:

```python
import pytest_lsp
from pytest_lsp import ClientServerConfig, LanguageClient

@pytest_lsp.fixture(
    config=ClientServerConfig(
        server_command=[os.environ["MOJO_LSP_SERVER"]],
    ),
)
async def client(lsp_client: LanguageClient):
    # Setup
    await initialize(lsp_client)
    yield
    # Teardown
    await lsp_client.shutdown_session()
```

The `yield` action is used to return temporarily to the test runner, which then
executes all the test methods in the test file using the initialized
`LanguageClient`, and eventually returns to shutdown the server.

#### LSP API in Tests Methods

All tests are asynchronous and receive the initialized `LanguageClient` as
argument. This client is able to communicate with the server and can be used to
dispatch requests and notifications. The list of requests and notifications can
be found in `site-packages/pytest_lsp/gen.py`.

As a general rule of thumb, if a request has a name of `"textDocument/request"`,
then there should be a method `LanguageClient.text_document_request_async()`
available. Likewise, a notification `"textDocument"/notification"` should have a
corresponding `LanguageClient.text_document_notification()` method.

It's worth mentioning that `LanguageClient` doesn't issue any LSP messages
behind the scenes, in fact, you need to mimic the exact communication traffic
with `mojo-lsp-server` that the LSP spec dictates.

Regarding LSP structures, the input and result types of these LSP methods have
the same naming and shapes as in the official
[LSP spec](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/).
They are defined in python in the `lspprotocol` library, in this path
`site-packages/lsprotocol/types.py`.

Finally, A test method would look like this:

```python
from lsprotocol.types import DidOpenTextDocumentParams, HoverParams

async def test_lsp_request(client: LanguageClient)
  # Notify that a file was opened
  client.text_document_did_open(DidOpenTextDocumentParams(...))
  # Request hover on a certain position
  results = client.text_document_hover_async(HoverParams(...))
  assert results.contents.value == "foo
```

#### Utils

The package `lib/utils` contains a set of helpers that make asserting and
issuing LSP messages less verbose. Please keep improving them.
