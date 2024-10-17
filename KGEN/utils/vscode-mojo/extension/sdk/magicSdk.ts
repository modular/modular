//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';
import { MojoSDKSpec } from './types';
import * as path from 'path';
import { mkdirp, chmod, rm, createWriteStream } from 'fs-extra';
import { directoryExists, readFile } from '../utils/files';
import * as util from 'util';
import { lock } from 'proper-lockfile';
import axios from 'axios';
import { Logger } from '../logging';
const execFile = util.promisify(require('child_process').execFile);
import { MojoSDKVersion as MaxSDKVersion } from './sdkVersion';
import { quote } from 'shell-quote';

async function downloadFile(
  url: string,
  outputPath: string,
  timeoutMins: number,
  errorMessage: string,
  logger: Logger
): Promise<boolean> {
  const writer = createWriteStream(outputPath);

  const response = await axios({
    url,
    method: 'GET',
    responseType: 'stream',
    timeout: timeoutMins * 60 * 1000,
  });

  response.data.pipe(writer);

  try {
    await new Promise((resolve, reject) => {
      writer.on('finish', resolve);
      writer.on('error', reject);
    });
    return true;
  } catch (ex: any) {
    vscode.window.showErrorMessage(errorMessage, ex.message);
    logger.main.logError(errorMessage, ex);
    return false;
  }
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
  versionDoneDirParent: string;
  versionDoneDir: string;
  magicUrl: string;
  version: string;
  major: string;
  minor: string;
  patch: string;
};

/**
 * @returns 1 if version1 > version2, -1 if version1 < version2, 0 otherwise.
 */
function compareNightlyMAXVersions(version1: string, version2: string): number {
  const [M1_, m1_, p1_, d1] = version1.split('.');
  const [M2_, m2_, p2_, d2] = version2.split('.');
  const M1 = parseInt(M1_);
  const m1 = parseInt(m1_);
  const p1 = parseInt(p1_);
  const M2 = parseInt(M2_);
  const m2 = parseInt(m2_);
  const p2 = parseInt(p2_);
  const compare = (a: any, b: any) => {
    return a === b ? 0 : a < b ? -1 : 1;
  };
  for (const [a, b] of [
    [M1, M2],
    [m1, m2],
    [p1, p2],
    [d1, d2],
  ]) {
    const comp = compare(a, b);
    if (comp !== 0) {
      return comp;
    }
  }
  return 0;
}

/**
 * @returns undefined or a non-empty list of versions sorted ascending semantically.
 */
async function getAllNightlyMAXVersions(
  logger: Logger,
  privateDir: string
): Promise<Optional<string[]>> {
  const repodataDir = path.join(privateDir, 'repodata');
  const now = new Date();
  const repodataFile = path.join(
    privateDir,
    'repodata',
    `${now.getFullYear()}-${now.getMonth()}-${now.getDay()}`
  );
  let contents = await readFile(repodataFile);
  // If the repodata for today is not present, then we download it and we delete any previous repodata files.
  if (!contents) {
    await rm(repodataDir, { recursive: true, force: true });
    await mkdirp(repodataDir);

    const repodataUrl =
      'https://conda.modular.com/max-nightly/noarch/repodata.json';
    logger.main.logInfo(`Will download ${repodataUrl} into ${repodataFile}`);
    if (
      !(await downloadFile(
        repodataUrl,
        repodataFile,
        /*timeoutMins=*/ 1,
        "Couldn't download " + repodataUrl,
        logger
      ))
    ) {
      return undefined;
    }
    logger.main.logInfo('Successfully downloaded');
  }
  contents = await readFile(repodataFile);
  if (!contents) {
    return undefined;
  }
  const jsonContents = JSON.parse(contents);
  const packages = jsonContents.packages;
  const versions: string[] = [];
  for (const packageName in packages) {
    if (packageName.startsWith('max-')) {
      versions.push(packages[packageName].version);
    }
  }
  versions.sort(compareNightlyMAXVersions);
  return versions.length === 0 ? undefined : versions;
}

async function getLatestNightlyMAXVersion(
  logger: Logger,
  privateDir: string
): Promise<Optional<string>> {
  const versions = await getAllNightlyMAXVersions(logger, privateDir);
  if (versions === undefined) {
    return undefined;
  }
  return versions[versions.length - 1];
}

async function findVersionToDownload(
  context: vscode.ExtensionContext,
  extVersion: string,
  isNightly: boolean,
  logger: Logger,
  privateDir: string
): Promise<Optional<[string, string, string]>> {
  const nightlyMaxVersionToComponents = (
    nightlyVersion: Optional<string>
  ): Optional<[string, string, string]> => {
    if (nightlyVersion === undefined) {
      return undefined;
    }
    let [major, minor, patch, dev] = nightlyVersion.split('.');
    return [major, minor, patch + `.${dev}`];
  };

  if (extVersion === '0.0.0') {
    if (!isNightly) {
      vscode.window.showErrorMessage(
        'Invalid extension version: ' + extVersion
      );
    }
    // If this is a dev version of the extension, we can figure out dynamically
    // what's the latest version of the sdk.
    return nightlyMaxVersionToComponents(
      await getLatestNightlyMAXVersion(logger, privateDir)
    );
  }
  if (isNightly) {
    return nightlyMaxVersionToComponents(
      context.extension.packageJSON.sdkVersion
    );
  }
  // stable
  const [major, minor, patch] =
    context.extension.packageJSON.sdkVersion.split('.');
  return [major, minor, patch];
}

async function createDownloadSpec(
  context: vscode.ExtensionContext,
  isNightly: boolean,
  logger: Logger
): Promise<Optional<DownloadSpec>> {
  const privateDir = context.globalStorageUri.fsPath;
  const magicDataHome = path.join(privateDir, 'magic-data-home');
  const magicPath = path.join(privateDir, 'magic');
  const doneDirectory = path.join(privateDir, 'done');
  const magicUrl = getMagicUrl();
  if (!magicUrl) {
    return undefined;
  }
  const extVersion = context.extension.packageJSON.version as string;
  const versionToDownload = await findVersionToDownload(
    context,
    extVersion,
    isNightly,
    logger,
    privateDir
  );
  if (!versionToDownload) {
    return undefined;
  }
  const [major, minor, patch] = versionToDownload;
  const version = `${major}.${minor}.${patch}`;
  const versionDoneDirParent = path.join(privateDir, 'versionDone');
  const versionDoneDir = path.join(versionDoneDirParent, version);
  return {
    privateDir,
    magicDataHome,
    magicPath,
    doneDirectory,
    versionDoneDir,
    versionDoneDirParent,
    magicUrl,
    version,
    major,
    minor,
    patch,
  };
}

async function doInstallMagicAndMAXSDK(
  downloadSpec: DownloadSpec,
  logger: Logger,
  isNightly: boolean
): Promise<void> {
  await rm(downloadSpec.doneDirectory, { recursive: true, force: true });

  logger.main.logInfo(
    `Will download ${downloadSpec.magicUrl} into ${downloadSpec.magicPath}`
  );
  await downloadFile(
    downloadSpec.magicUrl,
    downloadSpec.magicPath,
    /*timeoutMins=*/ 5,
    "Couldn't download magic",
    logger
  );
  logger.main.logInfo('Successfully downloaded magic.');
  await chmod(downloadSpec.magicPath, 0o755);
  logger.main.logInfo(
    `The permissions for ${downloadSpec.magicPath} have been changed and it's now executable.`
  );

  logger.main.logInfo(`Will prepare the MAX SDK installation.`);
  const env = { ...process.env };
  env['MAGIC_DATA_HOME'] = downloadSpec.magicDataHome;
  // We remove data home before installing again in case another process is
  // trying to write to it for some weird reason.
  logger.main.logInfo(`Removing magic-data-home`);
  await rm(downloadSpec.magicDataHome, {
    recursive: true,
    force: true,
  });

  const args = [
    'global',
    'install',
    '-c',
    'https://conda.modular.com/max' + (isNightly ? '-nightly' : ''),
    '-c',
    'conda-forge',
    `max==${downloadSpec.version}`,
    'python>=3.11,<3.12',
  ];
  logger.main.logInfo(`Installing the MAX SDK.`);
  await execFile(downloadSpec.magicPath, args, { env });
  logger.main.logInfo(`Successfully installed MAX.`);

  await mkdirp(downloadSpec.doneDirectory);
  await mkdirp(downloadSpec.versionDoneDir);
}

async function installMagicAndMAXSDKWithProgress(
  downloadSpec: DownloadSpec,
  logger: Logger,
  isNightly: boolean,
  reinstall: boolean
): Promise<boolean> {
  if (!reinstall && (await directoryExists(downloadSpec.versionDoneDir))) {
    logger.main.logInfo('Magic SDK present. Skipping installation.');
    return true;
  }
  await rm(downloadSpec.versionDoneDirParent, {
    recursive: true,
    force: true,
  });
  return await vscode.window.withProgress(
    {
      title: 'Installing the MAX SDK for VS Code',
      location: vscode.ProgressLocation.Notification,
    },
    async () => {
      try {
        await doInstallMagicAndMAXSDK(downloadSpec, logger, isNightly);
        return true;
      } catch (e) {
        logger.main.logError("Couldn't install the MAX SDK for VS Code", e);
        return false;
      }
    }
  );
}

async function acquireLockIfNeeded(
  logger: Logger,
  useLock: boolean,
  downloadSpec: DownloadSpec
): Promise<() => Promise<void>> {
  if (!useLock) {
    return async () => {};
  }
  logger.main.logInfo('Trying to acquire installation lock...');
  const releaseLock = await lock(downloadSpec.privateDir, { retries: 10 });
  logger.main.logInfo('Lock acquired...');
  return releaseLock;
}

export async function findMagicSDKSpec(
  withLock: boolean,
  context: vscode.ExtensionContext,
  logger: Logger,
  isNightly: boolean,
  reinstall: boolean = false
): Promise<Optional<MojoSDKSpec>> {
  const downloadSpec = await createDownloadSpec(context, isNightly, logger);
  if (downloadSpec === undefined) {
    return undefined;
  }
  await mkdirp(downloadSpec.magicDataHome);

  let success = false;
  try {
    logger.main.logInfo('Trying to acquire installation lock...');
    const releaseLock = await acquireLockIfNeeded(
      logger,
      withLock,
      downloadSpec
    );
    success = await installMagicAndMAXSDKWithProgress(
      downloadSpec,
      logger,
      isNightly,
      reinstall
    );
    await releaseLock();
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

  return {
    kind: 'magic',
    modularHomePath,
    section: 'mojo-max' + (isNightly ? '-nightly' : ''),
    version: new MaxSDKVersion(
      'MAX SDK ' + (isNightly ? '(nightly) ' : '(stable)'),
      downloadSpec.major,
      downloadSpec.minor,
      downloadSpec.patch,
      modularHomePath
    ),
  };
}
