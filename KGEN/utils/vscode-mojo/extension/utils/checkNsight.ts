//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';
import { Logger } from '../logging';

/**
 * Check if user has NVIDIA Nsight extension installed.
 * Prompts user to install if not installed.
 * Returns undefined if extension is installed and enabled.
 * Returns an error message string if the extension is not enabled.
 */
export async function checkNsightInstall(logger: Logger) {
  const nsight = vscode.extensions.getExtension('nvidia.nsight-vscode-edition');
  if (!nsight) {
    // Tell the user to install the nsight extension.
    const message =
      'Unable to start the cuda-gdb debug session. You first need to install and enablethe NVIDIA Nsight extension (nsight-vscode-edition).';
    logger.info(message);
    const response = await vscode.window.showInformationMessage(
      'Unable to debug with cuda-gdb mode without the NVIDIA Nsight extension.',
      'Find NVIDIA Nsight extension',
    );
    if (response) {
      vscode.commands.executeCommand(
        'workbench.extensions.search',
        '@id:nvidia.nsight-vscode-edition',
      );
    }
    return message;
  }
  return false;
}
