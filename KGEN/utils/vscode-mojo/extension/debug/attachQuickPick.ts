//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import { ProcessDescriptor, psList } from '../external/psList';

class RefreshButton implements vscode.QuickInputButton {
  get iconPath(): vscode.ThemeIcon {
    return new vscode.ThemeIcon('extensions-refresh');
  }

  get tooltip(): string {
    return 'Refresh process list';
  }
}

interface ProcessItem extends vscode.QuickPickItem {
  id: number;
}

async function getProcessItems(
  context: vscode.ExtensionContext,
): Promise<ProcessItem[]> {
  let processes: ProcessDescriptor[] = await psList(context);
  processes.filter((p) => p.pid !== undefined);

  processes.sort((a, b) => {
    if (a.name === undefined) {
      if (b.name === undefined) {
        return 0;
      }
      return 1;
    }
    if (b.name === undefined) {
      return -1;
    }
    const aLower: string = a.name.toLowerCase();
    const bLower: string = b.name.toLowerCase();
    if (aLower === bLower) {
      return 0;
    }
    return aLower < bLower ? -1 : 1;
  });
  return processes.map((process): ProcessItem => {
    return {
      label: process.name,
      description: `${process.pid}`,
      detail: process.cmd,
      id: process.pid,
    };
  });
}

/**
 * Show a QuickPick that selects a process to attach.
 *
 * @param context
 * @param debugConfig The debug config this action originates from. Its name is
 *     used as cache key for persisting the filter the user used to find the
 *     process to attach.
 * @returns The pid of the selected process as string. If the user cancelled, an
 *     exception is thrown.
 */
async function showQuickPick(
  extensionContext: vscode.ExtensionContext,
  debugConfig: any,
): Promise<Optional<string>> {
  const processItems: ProcessItem[] = await getProcessItems(extensionContext);
  const memento = extensionContext.workspaceState;
  const filterMementoKey = 'searchProgramToAttach' + debugConfig.name;
  const previousFilter = memento.get<string>(filterMementoKey);

  return new Promise<Optional<string>>((resolve, reject) => {
    const quickPick: vscode.QuickPick<ProcessItem> =
      vscode.window.createQuickPick<ProcessItem>();
    quickPick.value = previousFilter || '';
    quickPick.title = 'Attach to process';
    quickPick.canSelectMany = false;
    quickPick.matchOnDescription = true;
    quickPick.matchOnDetail = true;
    quickPick.placeholder = 'Select the process to attach to';
    quickPick.buttons = [new RefreshButton()];
    quickPick.items = processItems;
    let textFilter = '';
    const disposables: vscode.Disposable[] = [];

    quickPick.onDidTriggerButton(
      async () => {
        quickPick.items = await getProcessItems(extensionContext);
      },
      undefined,
      disposables,
    );

    quickPick.onDidChangeValue((e: string) => {
      textFilter = e;
    });

    quickPick.onDidAccept(
      () => {
        if (quickPick.selectedItems.length !== 1) {
          reject(new Error('Process not selected.'));
        }

        const selectedId: string = `${quickPick.selectedItems[0].id}`;

        disposables.forEach((item) => item.dispose());
        quickPick.dispose();
        memento.update(filterMementoKey, textFilter);

        resolve(selectedId);
      },
      undefined,
      disposables,
    );

    quickPick.onDidHide(
      () => {
        disposables.forEach((item) => item.dispose());
        quickPick.dispose();

        resolve(undefined);
      },
      undefined,
      disposables,
    );

    quickPick.show();
  });
}

export function activatePickProcessToAttachCommand(
  extensionContext: vscode.ExtensionContext,
): vscode.Disposable {
  return vscode.commands.registerCommand(
    'mojo.pickProcessToAttach',
    async (debugConfig: any) => showQuickPick(extensionContext, debugConfig),
  );
}
