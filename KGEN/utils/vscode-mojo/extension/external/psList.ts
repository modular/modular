//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// This file has been copied from https://github.com/sindresorhus/ps-list
// using commit 6dbe8d6.
// We have to copy paste it instead of adding it as a regular dependency
// because the npm package `ps-list` is an ES module, which is not
// compatible with Electron, the runtime for VS Code, which means that
// we can't import it once our extension is loaded as a vsix package.

// The only modification to this file is changing the line
//   const __dirname = path.dirname(fileURLToPath(import.meta.url));
// to using vscode.ExtensionContext.extensionPath to find the current
// dirname.

import * as childProcess from 'child_process';
import * as path from 'path';
import * as process from 'process';
import { promisify } from 'util';
import * as vscode from 'vscode';

const TEN_MEGABYTES = 1000 * 1000 * 10;
const execFile = promisify(childProcess.execFile);

const windows = async (context: vscode.ExtensionContext) => {
  // Source: https://github.com/MarkTiedemann/fastlist
  let binary;
  switch (process.arch) {
    case 'x64':
      binary = 'fastlist-0.3.0-x64.exe';
      break;
    case 'ia32':
      binary = 'fastlist-0.3.0-x86.exe';
      break;
    default:
      throw new Error(`Unsupported architecture: ${process.arch}`);
  }

  const binaryPath = path.join(context.extensionPath, 'bin', binary);
  const { stdout } = await execFile(binaryPath, {
    maxBuffer: TEN_MEGABYTES,
    windowsHide: true,
  });

  return stdout
    .trim()
    .split('\r\n')
    .map((line) => line.split('\t'))
    .map(([pid, ppid, name]) => ({
      pid: Number.parseInt(pid, 10),
      ppid: Number.parseInt(ppid, 10),
      name,
    }));
};

const nonWindowsMultipleCalls = async (options: any = {}) => {
  const flags = (options.all === false ? '' : 'a') + 'wwxo';
  const returnValue: { [index: string]: any } = {};

  await Promise.all(
    ['comm', 'args', 'ppid', 'uid', '%cpu', '%mem'].map(async (cmd) => {
      const { stdout } = await execFile('ps', [flags, `pid,${cmd}`], {
        maxBuffer: TEN_MEGABYTES,
      });

      for (let line of stdout.trim().split('\n').slice(1)) {
        line = line.trim();
        const [pid] = line.split(' ', 1);
        const value = line.slice(pid.length + 1).trim();

        if (returnValue[pid] === undefined) {
          returnValue[pid] = {};
        }

        returnValue[pid][cmd] = value;
      }
    }),
  );

  // Filter out inconsistencies as there might be race
  // issues due to differences in `ps` between the spawns
  return Object.entries(returnValue)
    .filter(
      ([, value]: any) =>
        value.comm &&
        value.args &&
        value.ppid &&
        value.uid &&
        value['%cpu'] &&
        value['%mem'],
    )
    .map(([key, value]: any) => ({
      pid: Number.parseInt(key, 10),
      name: path.basename(value.comm),
      cmd: value.args,
      ppid: Number.parseInt(value.ppid, 10),
      uid: Number.parseInt(value.uid, 10),
      cpu: Number.parseFloat(value['%cpu']),
      memory: Number.parseFloat(value['%mem']),
    }));
};

const ERROR_MESSAGE_PARSING_FAILED = 'ps output parsing failed';

const psOutputRegex =
  /^[ \t]*(?<pid>\d+)[ \t]+(?<ppid>\d+)[ \t]+(?<uid>[-\d]+)[ \t]+(?<cpu>\d+\.\d+)[ \t]+(?<memory>\d+\.\d+)[ \t]+(?<comm>.*)?/;

const nonWindowsCall = async (options: any = {}) => {
  const flags = options.all === false ? 'wwxo' : 'awwxo';

  const psPromises = [
    execFile('ps', [flags, 'pid,ppid,uid,%cpu,%mem,comm'], {
      maxBuffer: TEN_MEGABYTES,
    }),
    execFile('ps', [flags, 'pid,args'], { maxBuffer: TEN_MEGABYTES }),
  ];

  const [psLines, psArgsLines] = (await Promise.all(psPromises)).map(
    ({ stdout }) => stdout.trim().split('\n'),
  );

  const psPids = new Set(psPromises.map((promise) => promise.child.pid));

  psLines.shift();
  psArgsLines.shift();

  const processCmds: { [index: string]: any } = {};
  for (const line of psArgsLines) {
    const [pid, cmds]: any = line.trim().split(' ');
    processCmds[pid] = cmds.join(' ');
  }

  const processes = psLines
    .map((line) => {
      const match = psOutputRegex.exec(line);

      if (match === null) {
        throw new Error(ERROR_MESSAGE_PARSING_FAILED);
      }

      const { pid, ppid, uid, cpu, memory, comm }: any = match.groups;

      const processInfo = {
        pid: Number.parseInt(pid, 10),
        ppid: Number.parseInt(ppid, 10),
        uid: Number.parseInt(uid, 10),
        cpu: Number.parseFloat(cpu),
        memory: Number.parseFloat(memory),
        name: path.basename(comm),
        cmd: processCmds[pid],
      };

      return processInfo;
    })
    .filter((processInfo) => !psPids.has(processInfo.pid));

  return processes;
};

const nonWindows = async (options = {}) => {
  try {
    return await nonWindowsCall(options);
  } catch {
    // If the error is not a parsing error, it should manifest itself in
    // multicall version too.
    return nonWindowsMultipleCalls(options);
  }
};

export interface Options {
  /**
  Include other users' processes as well as your own.

  On Windows this has no effect and will always be the users' own processes.

  @default true
  */
  readonly all?: boolean;
}
export interface ProcessDescriptor {
  readonly pid: number;
  readonly name: string;
  readonly ppid: number;

  /**
  Not supported on Windows.
  */
  readonly cmd?: string;

  /**
  Not supported on Windows.
  */
  readonly cpu?: number;

  /**
  Not supported on Windows.
  */
  readonly memory?: number;

  /**
  Not supported on Windows.
  */
  readonly uid?: number;
}

export function psList(
  context: vscode.ExtensionContext,
  options?: Options,
): Promise<ProcessDescriptor[]> {
  return process.platform === 'win32' ? windows(context) : nonWindows(options);
}
