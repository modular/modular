//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';
import { MojoSDKSpec } from './types';
import * as config from '../utils/config';
import * as path from 'path';
import { directoryExists, mkdirp } from '../utils/files';
import * as util from 'util';
import * as fs from 'fs';
import { lock } from 'proper-lockfile';
import axios from 'axios';
import { Logger } from '../logging';
const execFile = util.promisify(require('child_process').execFile);
import { MojoSDKVersion } from './sdkVersion';
const chmod = util.promisify(require('fs').chmod);

async function downloadFile(url: string, outputPath: string) {
  const writer = fs.createWriteStream(outputPath);

  const response = await axios({
    url,
    method: 'GET',
    responseType: 'stream',
  });

  response.data.pipe(writer);

  return new Promise((resolve, reject) => {
    writer.on('finish', resolve);
    writer.on('error', reject);
  });
}

function getMagicUrl(): Optional<string> {
  let platform: string;
  if (process.platform === 'linux') {
    platform = 'unknown-linux-musl';
  } else if (process.platform === 'darwin') {
    platform = 'apple-darwin';
  } else if (process.platform === 'win32') {
    platform = 'pc-windows-msvc';
  } else {
    vscode.window.showErrorMessage(
      `The MAX SDK is not supported in this platform: ${process.platform}`
    );
    return undefined;
  }
  let arch: string;
  if (process.arch === 'x64') {
    arch = 'x86_64';
  } else if (process.arch === 'arm64') {
    arch = 'aarch64';
  } else {
    arch = process.arch;
  }
  return `https://dl.modular.com/public/magic/raw/versions/latest/magic-${arch}-${platform}`;
}

type DownloadSpec = {
  privateDir: string;
  magicDataHome: string;
  magicPath: string;
  doneDirectory: string;
  versionDoneDir: string;
  magicUrl: string;
  version: string;
  major: string;
  minor: string;
  patch: string;
};

function createDownloadSpec(
  context: vscode.ExtensionContext
): Optional<DownloadSpec> {
  const privateDir = context.globalStorageUri.fsPath;
  const magicDataHome = path.join(privateDir, 'magic-data-home');
  const magicPath = path.join(privateDir, 'magic');
  const doneDirectory = path.join(privateDir, 'done');
  const magicUrl = getMagicUrl();
  if (!magicUrl) {
    return undefined;
  }
  const major = '24';
  const minor = '4';
  const patch = '0dev7';
  const version = `${major}.${minor}.${patch}`;
  return {
    privateDir,
    magicDataHome,
    magicPath,
    doneDirectory,
    versionDoneDir: path.join(privateDir, version),
    magicUrl,
    version,
    major,
    minor,
    patch,
  };
}

async function doInstallMagicSDK(
  downloadSpec: DownloadSpec,
  logger: Logger
): Promise<void> {
  try {
    fs.rmdirSync(downloadSpec.doneDirectory);
  } catch {}

  logger.main.logInfo(
    `Will download ${downloadSpec.magicUrl} into ${downloadSpec.magicPath}`
  );
  await downloadFile(downloadSpec.magicUrl, downloadSpec.magicPath);
  logger.main.logInfo('Successfully downloaded');
  await chmod(downloadSpec.magicPath, 0o755);
  logger.main.logInfo(
    `The permissions for ${downloadSpec.magicPath} have been changed.`
  );

  logger.main.logInfo(`Will install MAX`);
  const env = { ...process.env };
  env['MAGIC_DATA_HOME'] = downloadSpec.magicDataHome;

  const args = [
    'global',
    'install',
    '-c',
    'https://conda.modular.com/max',
    '-c',
    'conda-forge',
    `max==${downloadSpec.version}`,
    'python>=3.11,<3.12',
  ];
  await execFile(downloadSpec.magicPath, args, { env });
  logger.main.logInfo(`Successfully installed MAX`);

  await mkdirp(downloadSpec.doneDirectory);
  await mkdirp(downloadSpec.versionDoneDir);
}

async function installMagicSDKWithProgress(
  downloadSpec: DownloadSpec,
  logger: Logger
): Promise<boolean> {
  if (
    (await directoryExists(downloadSpec.doneDirectory)) &&
    (await directoryExists(downloadSpec.versionDoneDir))
  ) {
    logger.main.logInfo('Magic SDK present. Skipping installation.');
    return true;
  }
  return await vscode.window.withProgress(
    {
      title: 'Installing the MAX SDK for VS Code',
      location: vscode.ProgressLocation.Notification,
    },
    async () => {
      try {
        await doInstallMagicSDK(downloadSpec, logger);
        return true;
      } catch (e) {
        logger.main.logError("Couldn't install the MAX SDK for VS Code", e);
        return false;
      }
    }
  );
}

export async function findMagicSDKSpec(
  withLock: boolean,
  context: vscode.ExtensionContext,
  logger: Logger
): Promise<Optional<MojoSDKSpec>> {
  const downloadSpec = createDownloadSpec(context);
  if (downloadSpec === undefined) {
    return undefined;
  }
  await mkdirp(downloadSpec.magicDataHome);

  let success = false;
  try {
    logger.main.logInfo('Trying to acquire installation lock...');
    const release = withLock
      ? await lock(downloadSpec.privateDir, { retries: 10 })
      : async () => {};
    logger.main.logInfo('Lock acquired...');
    success = await installMagicSDKWithProgress(downloadSpec, logger);
    await release();
  } catch (e) {
    logger.main.logError(
      'Error while handling the lock for the MAX SDK for VS Code',
      e
    );
  }
  if (!success) {
    vscode.window.showErrorMessage("Couldn't install the MAX SDK for VS Code");
    return undefined;
  }
  const modularHomePath = path.join(
    downloadSpec.magicDataHome,
    'envs',
    'max',
    'share',
    'max'
  );

  // Consider nightly here
  return {
    kind: 'magic',
    modularHomePath,
    section: 'mojo-max',
    version: new MojoSDKVersion(
      'MAX SDK for VS Code',
      downloadSpec.major,
      downloadSpec.minor,
      downloadSpec.patch,
      modularHomePath
    ),
  };
}
