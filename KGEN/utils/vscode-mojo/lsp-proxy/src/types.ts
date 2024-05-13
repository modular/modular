//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

export interface InitializationOptions {
  serverArgs: string[];
  serverPath: string;
  serverEnv: {[env: string]: string|undefined}
}

export type JSONObject = {
  [key: string]: any
};
export type RequestId = number;

export type ExitStatus = {
  code: number|null; signal : NodeJS.Signals | null;
}
