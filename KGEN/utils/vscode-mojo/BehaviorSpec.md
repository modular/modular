# Mojo VS Code Extension Behavior Spec

## Introduction

This document outlines the most important behaviors of the extension, especially
under different SDK conditions. The extension is already complex enough to
require a document like this one to serve as a communication means to help
extension developers modify the extension in a coherent way.

Anything that you'd like to be expressed in an integration test should be
written here. If there's a behavior that for which writing an integration test
would be really hard, then it's imperative to outline it here.

On the other hand, integration tests can't cover all the expectations we might want
for the extension, but they should try to cover as much as possible the
specification of this document.

Finally, updates to the extension's code must be reflected in this document accordingly.

## Terms

- `Mojo SDK`: the `mojo` section of the Max SDK.
- `enableMagicSDK`: setting that indicates the extension to use `magic`
  to look for a Max SDK instead of the `modular cli`. In this mode, a vendored
  version of `magic` and the Max SDK are automatically downloaded and managed by
  the extension.
- `activeSDK`: at all points, the extension only uses one single Mojo SDK for
  the entirety of its features, we call this SDK the `activeSDK`. The
  `activeSDK` can be undefined, if no SDKs have been seen. Otherwise, it
  indicates an SDK somewhere in the file system, which might also be invalid.
  The invalid state indicates that the extension identified that this SDK is
  corrupt and failed to load properly.
- `initializationSDK`: an SDK that is forcefully used upon initialization of
  the extension.
- `modular cli SDK`: an SDK gotten via the modular cli tool.
- `magic SDK`: an SDK gotten via `magic`.
- `dev SDK`: an SDK gotten from a modular repo.

## Top-level behaviors

### Switching the `activeSDK`

If the `activeSDK` is switched to another one, the extension is reloaded using the
new SDK as `initializationSDK`.

### Initialization and restart of the extension

Initialization can happen in the following cases:

- The extension is initialized by VS Code by an activation event or an extension
  update. In this case, there will be no `initializationSDK` to guarantee a
  clean state.
- The extension is restarted via the `Mojo: Restart the extension command`, which
  should respect the current `initializationSDK`.
- The extension is restarted because of an SDK switch, in which
  case there's a new `initializationSDK`.
- The extension is restarted because the value of `enableMagicSDK` changed, in
  which case `initializationSDK` is unset.

#### Initialization steps

1. The extension initializes synchronously the SDK Manager and the logging service.
   No language feature is activated yet.
1. The extension notifies the user if both the nightly and stable versions of
   the extension are enabled, in which case it prompts the user to disable one
   of them and restart the extension. No language feature is activated in this case.
1. If only one extension is enabled, then it performs its async activation,
   where most language features are activated, which include

   - Test Manager
   - RPC Debug Server
   - Decoration Manager
   - LSP Manager
   - Debug Manager

   This step is asynchronous given the nature of finding the SDK, which is used
   by almost all language features. The interesting bits are in the
   `SDK management` section.

#### Restarts

Whenever there's a restart, the extension waits some seconds for the previous
activation to finish to prevent race conditions.

## SDK management

At some point during the activation of any of the language features, such
features will get access to an SDK via the `findSDK` method. This method doesn't
throw, but returns an optional. If an SDK is returned, this means that the SDK
has passed a minimum level of validation, but it might still fail when invoking
specific SDK tools. In the case of `undefined` being returned, the feature
should not show an error. It's worth noticing that `undefined` maybe returned
even if there's an actual `activeSDK`.

The language feature shouldn't show an error message if `undefined` is returned.
Error messaging is handled inside `activeSDK`.

### SDK validation and error messages

Whenever the `activeSDK` is set, some basic SDK validation will happen. This
validation looks for errors in the `modular.cfg` file. It doesn't validate that
the underlying tools exist on disk, as corresponding issues would be thrown when
using these tools from language features. The validation error message is stored
and it's evicted if the `modular.cfg` file changes. If the validation passed,
`undefined` is stored as error.

Additionally, whenever `findSDK` is invoked, the same SDK validation process is
executed, its error message is stored and displayed to the user. This is useful
to remind users of SDK errors when executing important actions, like launching
the debugger.

However, there are cases like formatting on save, on which you
don't want to show the same error message over and over again. In this case,
`findSDK` should be invoked with the flag `hideRepeatedErrors`, in which the
error message is not shown only if it's the same as the last saved one.

### `dev SDK`

A `dev SDK` will be identified given a path within the modular repo. The
identification logic will traverse this path up until all of the following
sub-paths exist `.git`, `.derived` and `WORKSPACE.bazel`. This is enough for the
extension to consider the ancestor directory valid `.derived/modular.cfg` config
file.

### `modular cli SDK`

This logic is based on querying the `modular cli` tool.

### `magic SDK`

The magic SDK is gotten by downloading magic and installing Max in a directory
private to the extension.

Additionally, a binary `vscode[-insiders]-mojo-[stable|nightly]` is installed in
the users's path following the process done by the VS Code command
`Install 'code' command in path`. Failures are not reported, as this is a rather
obscure feature. However, a command for reinstalling it manually is provided.

### SDK identification during initialization

The following steps describe what happens when `findSDK` is invoked for the
first time. Besides that, it has to be guaranteed that all terminal nodes invoke
`set as activeSDK`.

- `initialization` with `initializationSDK`:

  - The `activeSDK` remains the `initializationSDK` and it is `set as activeSDK`
    even if it fails SDK validation. This is done to guarantee that there are
    no unexpected SDK changes.
  - The `activeSDK` is returned.

- `initialization` (`enableMagicSDK` is `false`):

  - The extension will look for a `modular cli SDK` without doing version
    matching. It will also look for all the `dev SDK` from workspace directories
    and open files.
    - If there are no SDKs, then the user is prompted to install the
      `modular cli SDK` via a web link, and then asked to restart the IDE after
      that. `set as activeSDK` is invoked with `undefined`.
    - If there's one SDK, that one is `set as activeSDK`.
    - IF there are multiple SDKs, the user is then prompted to pick one of
      these SDKs and that one will be `set as activeSDK`.
  - The `activeSDK` is returned.

- `initialization` (`enableMagicSDK` is `true`):
  - The extension will look for a `magic SDK` without doing version
    matching. If the SDK is not present, it will be automatically downloaded
    using commands in an output channel , which will be convenient for error reporting.
    The extension will also look for all the `dev SDK` from workspace directories
    and open files.
    - If there are errors downloading the `magic SDK`, the user is prompted to
      click an action to reinstall it, this time without file locking mechanisms.
      - If this action failed, the user is asked to file an issue.
    - After the previous steps, if there are no SDKs, `set as activeSDK` is
      invoked with `undefined`.
    - If there's one SDK, that one is `set as activeSDK`.
    - If there are multiple SDKs, the user is then prompted to pick one of
      these SDKs and that one will be `set as activeSDK`.
  - The `activeSDK` is returned.

### SDK identification after initialization

At any point during the program execution, whenever a workspace folder is loaded
or a file is open, `dev SDK`s will be attempted to be identified. If that
happens, then the user will be prompted to pass it to `set as activeSDK`.

Whenever any SDK is identified via a custom debug configuration, the SDK
validation step will be run but the user won't be prompted to use that SDK as
active.

### `set as activeSDK`

This step runs `SDK validation` and prompts the user of a corresponding
troubleshooting message upon failures.

- For `modular cli SDK`, the message prompts reinstalling the SDK via a
  web link and then indicates that the IDE should be reloaded after that.
- For `dev SDK`, the message asks running the `://install` target, and
  also mentions that the extension will pick this change up eventually. It also
  shows a button to restart the extension manually.
- For `magic SDK`, the message shows a button to run the reinstallation
  of the `magic SDK` without locking, or file an issue. Upon a successful
  reinstallation, the extension is reloaded automatically using it as
  `initializationSDK`.

Additionally, a file listener is installed to look for changes to `modular.cfg`
that will evict saved validation error message.

### Manual SDK selection

A command is provided for users to manually select an SDK from the list of all
identified SDKs.

## Tool-specific behaviors

### Debugger

#### SDK for RPC debug sessions

Unlike the `activeSDK` managed by the extension, an RPC call may originate from a
different SDK, because of which the debug config must contain the path to the
originating SDK in the `modularHomePath` field, and the SDK manager should
resolve this one. The `activeSDK` is only used if the debug config doesn't
contain this field. In case of failures at resolving this specific SDK, an error
should be displayed to the user and returned to the RPC client.

#### Failure at launching the debugger

Basic launching failures should be shown to the user on the IDE
and on the RPC client if applicable. This includes, dynamic linker errors,
missing executable error, wrong permissions errors, etc.

Initialization errors after launching, like missing lldb-server errors, should
be displayed to the user on the IDE but not to RPC clients.

### LSP

There's only one LSP server for all Mojo files, including notebooks. Besides
that, the LSP is not launched unless a Mojo file is open.

## Future work

- Enabling auto updates of the magic installation
- Enabling multiple concurrent SDKs based on max projects
