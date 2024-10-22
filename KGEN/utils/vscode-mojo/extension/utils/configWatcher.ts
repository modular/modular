//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as chokidar from 'chokidar';
import * as vscode from 'vscode';

import * as config from './config';
import { DisposableContext } from './disposableContext';

/**
 *  Prompt the user to see if we should restart the server.
 */
async function promptRestart(settingName: string, promptMessage: string) {
  switch (config.get<string>(settingName, /*workspaceFolder=*/ undefined)) {
    case 'restart':
      vscode.commands.executeCommand('mojo.extension.restart');
      break;
    case 'ignore':
      break;
    case 'prompt':
    default:
      switch (
        await vscode.window.showInformationMessage(
          promptMessage,
          'Yes',
          'Yes, always',
          'No, never',
        )
      ) {
        case 'Yes':
          vscode.commands.executeCommand('mojo.extension.restart');
          break;
        case 'Yes, always':
          vscode.commands.executeCommand('mojo.extension.restart');
          config.update<string>(
            settingName,
            'restart',
            vscode.ConfigurationTarget.Global,
          );
          break;
        case 'No, never':
          config.update<string>(
            settingName,
            'ignore',
            vscode.ConfigurationTarget.Global,
          );
          break;
        default:
          break;
      }
      break;
  }
}

/**
 *  Activate watchers that track configuration changes for the given workspace
 *  folder, or undefined if the workspace is top-level.
 */
export async function activate({
  workspaceFolder,
  settings,
  paths,
}: {
  workspaceFolder?: Optional<vscode.WorkspaceFolder>;
  settings?: Optional<string[]>;
  paths?: Optional<string[]>;
}): Promise<DisposableContext> {
  // Flag that controls whether a restart event was issued. This is used to
  // prevent multiple simultaneous restarts caused by, for example, multiple
  // watchers being triggered at once.
  let restartIssued = false;
  const promptRestartOnce = (promptMessage: string) => {
    if (restartIssued) {
      return;
    }
    restartIssued = true;
    promptRestart('onSettingsChanged', promptMessage);
  };

  const disposables = new DisposableContext();
  // When a configuration change happens, check to see if we should restart.
  disposables.pushSubscription(
    vscode.workspace.onDidChangeConfiguration((event) => {
      for (const setting of settings || []) {
        const expandedSetting = `mojo.${setting}`;
        if (event.affectsConfiguration(expandedSetting, workspaceFolder)) {
          promptRestartOnce(
            `setting '${
              expandedSetting
            }' has changed. Do you want to reload the server?`,
          );
        }
      }
    }),
  );

  // Setup watchers for the provided paths.
  const fileWatcherConfig = {
    disableGlobbing: true,
    followSymlinks: true,
    ignoreInitial: true,
    awaitWriteFinish: true,
  };
  for (const serverPath of paths || []) {
    // If the path actually exists, track it in case it changes.
    const fileWatcher = chokidar.watch(serverPath, fileWatcherConfig);
    fileWatcher.on('all', (event, _filename, _details) => {
      if (event != 'unlink') {
        promptRestartOnce(
          'mojo language server file has changed. Do you want to reload the server?',
        );
      }
    });
    disposables.pushSubscription(
      new vscode.Disposable(() => {
        fileWatcher.close();
      }),
    );
  }
  return disposables;
}
