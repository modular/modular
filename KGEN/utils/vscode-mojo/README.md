# Mojo for Visual Studio Code

This VS Code extension from the Modular team adds support for the
[Mojo programming language](https://www.modular.com/mojo).

## Features

- Syntax highlighting for `.mojo` and `.🔥` files
- Code completion
- Code diagnostics and quick fixes
- Full LSP experience for doc string code blocks
- Go to symbol
- API docs on hover
- Code formatting
- Run Mojo file

## Get started

1. Install the Mojo SDK.
2. Install the Mojo VS Code extension.
3. Open any `.mojo` or `.🔥` file.

## Configuration

The extension will attempt to find the path of the Mojo SDK installation using
the `MODULAR_HOME` environment variable. If `MODULAR_HOME` is not set within
the environment, the path can be explicitly set via the `mojo.modularHomePath`
extension setting.

```json
{
    "mojo.modularHomePath": "/absolute/path/to/.modular"
}
```

## Code Completion

To trigger a completion press `ctrl + space`, pressing `ctrl + space` again will
bring up doc hints:

![completion](https://github.com/modularml/mojo/assets/77730378/51af7c47-8c39-449b-a759-8351c543208a)

Rebind the hotkey in Preferences: Open Keyboard Shortcuts > `Trigger Suggest`

## Hover and Doc Hints

Hover over a symbol with your cursor for doc hints. The default hotkey
to trigger it in macOS is `⌘ + k`, `⌘ + i` or `ctrl + k`, `ctrl + i` in Linux
and Windows:

![hover](https://github.com/modularml/mojo/assets/77730378/59881310-d2ec-481f-975a-d69d5e6c7ae3)

Rebind the hotkey in Preferences: Open Keyboard Shortcuts >
`Show or Focus Hover`

## Signature Help

Mojo provides function overloading, so you need a way to scroll through the
multiple signatures available. You can bring this up with the hotkey
`⌘ + shift + space` in macOS or `ctrl + shift + space` in Linux or Windows.

![signature-help](https://github.com/modularml/mojo/assets/77730378/3994ab6d-ae4b-43af-9ddf-0d979c51330f)

Rebind related hotkeys in Preferences: Open Keyboard Shortcuts >
`Trigger Parameter Hints`

## Code Diagnostics

Code diagnostics are indicated with an underline on the code and details appear
when you hover. You can also see them in the `PROBLEMS` tab and use
`Go to Next Problem in Files` to quickly cycle through them:

![diagnostics2](https://github.com/modularml/mojo/assets/77730378/b9d4c570-62da-4e82-981d-6d95ea8f34a2)

Rebind related hotkeys in Preferences: Open Keyboard Shortcuts >
`Go to Next Problem...`

**Tip:** Also try the `Error Lens` extension (not associated with Modular),
which will display the first line of the diagnostic inline, making it easier
to quickly fix problems.

## Doc String Code Blocks

Unique to Mojo, you get a full LSP experience for code blocks inside doc
strings, with all the features mentioned here including completions and
diagnostics:

![doc-lsp](https://github.com/modularml/mojo/assets/77730378/c2d73fd0-66de-44e7-8125-511bf0237396)

## Go to Symbol

You can quickly jump to a symbol in the file with `⌘ + shift + o` in macOS or
`ctrl + shift + o` in Linux and Windows.

![go-to-symbol](https://github.com/modularml/mojo/assets/77730378/1972e611-4a01-4a7f-945d-a3b5f10034a9)

This also enables the outline view in the explorer window.

Rebind the hotkey in Preferences: Open Keyboard Shortcuts >
`Go to Symbol in Editor`

## Quick Fix

If there is an available quick fix with the code diagnostic, click
the lightbulb icon or use the default hotkey `ctrl + .` for a list of options:

![quick-fix](https://github.com/modularml/mojo/assets/77730378/b9bb1122-9fdc-4fbc-b3a8-28a54cd78704)

Rebind the hotkey in Preferences: Open Keyboard Shortcuts >
`Quick Fix...`

## Run Mojo File

The extension provides a set of actions on the top-right of a Mojo file to run
the active file, which by default are under a small `▶️` button up the
top-right of the editor:

![run-file](https://github.com/modularml/mojo/assets/77730378/22ef37cf-154a-430b-9ef3-427dbab411fc)

These actions are also available in the command palette and under the `Mojo`
submenu in the File Explorer when right-clicking on Mojo files:

![right-click-menu](https://github.com/modularml/mojo/assets/77730378/b267a44c-fa2c-425d-bada-7360cd338351)

You may bind hotkeys to any of the actions listed here. For example, to bind a
hotkey for the "Run Mojo File" action, open preferences, then select
`Keyboard Shortcuts > Mojo: Run Mojo File`.

### Run Mojo File

This executes the current Mojo file in a terminal that is reused by other
invocations of this same action, even if they run a different file.

### Run Mojo File in Dedicated Terminal

This executes the current Mojo file in a dedicated terminal that is reused only
by subsequent runs of this very same file.

## Code Formatting

From the command palette run `Format Document` or tick the setting
`Format on Save`:

![format](https://github.com/modularml/mojo/assets/77730378/4e0e22c4-0216-41d7-b5a5-7f48a018fd81)

## Restarting Mojo Extension

The extension may crash and produce incorrect results periodically, to fix this
from the command palette search for `Mojo: Restart the extension`

![restart](https://github.com/modularml/mojo/assets/77730378/c65bf84b-5c9b-4151-8176-2b098533dbe3)

Bind a hotkey in Preferences: Open Keyboard Shortcuts >
`Mojo: Restart the extension`
