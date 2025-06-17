//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import { ChildProcess } from 'child_process';

import { ExitStatus, JSONObject, Optional } from './types';

/**
 * A stream reader that reports whenever a line ending with `\n` is found.
 */
export class LineSeparatedStream {
  private enabled = true;

  constructor(
    rawStream: NodeJS.ReadableStream,
    onLine: (line: string) => void,
  ) {
    let buffer = '';
    rawStream.on('data', (chunk: any) => {
      if (!this.enabled) {
        return;
      }

      buffer += chunk;

      let newLinePos = -1;
      while ((newLinePos = buffer.indexOf('\n')) !== -1) {
        const line = buffer.substring(0, newLinePos);
        buffer = buffer.substring(newLinePos + 1);
        onLine(line);
      }
    });
  }

  public dispose() {
    this.enabled = false;
  }
}

/**
 * A stream reader based on the JSON-RPC protocol that reports whenever a
 * notification or the response to a request is found.
 */
export class JSONRPCStream {
  static protocolHeader = 'Content-Length: ';
  static protocolLineSeparator = '\r\n\r\n';
  private buffer = '';
  private enabled = true;

  constructor(
    rawStream: NodeJS.ReadableStream,
    onResponse: (response: JSONObject) => void,
    onNotification: (notification: JSONObject) => void,
    onOutgoingRequest: (request: JSONObject) => void,
  ) {
    rawStream.on('data', (chunk: any) => {
      if (!this.enabled) {
        return;
      }

      this.buffer += chunk;

      let packet: Optional<JSONObject>;
      while ((packet = this.tryProcessPacket()) != undefined) {
        if ('id' in packet) {
          // Differentiate between a response to a client request or a request from the server to the client.
          if ('method' in packet) {
            onOutgoingRequest(packet);
          } else {
            onResponse(packet);
          }
        } else {
          onNotification(packet);
        }
      }
      return true;
    });
  }

  /**
   * Tries to read a packet from the buffer and update that buffer if found.
   */
  private tryProcessPacket(): Optional<JSONObject> {
    // We process first the protocol header.
    if (!this.buffer.startsWith(JSONRPCStream.protocolHeader)) {
      return undefined;
    }
    // Then we parse the content length.
    let index = JSONRPCStream.protocolHeader.length;
    let contentLength = 0;
    for (; index < this.buffer.length; index++) {
      const c = this.buffer[index];

      if (c < '0' || c > '9') {
        break;
      }
      contentLength = contentLength * 10 + parseInt(c);
    }
    // Then we parse the line separator.
    if (
      !this.buffer
        .substring(index)
        .startsWith(JSONRPCStream.protocolLineSeparator)
    ) {
      return undefined;
    }

    // Then we extract the contents of the packet.
    const contentBegPos = index + JSONRPCStream.protocolLineSeparator.length;
    const contentBytes = Buffer.from(this.buffer.substring(contentBegPos));

    if (contentBytes.length < contentLength) {
      return undefined;
    }
    const contents = contentBytes.subarray(0, contentLength).toString();

    // We update the buffer to point past this packet.
    this.buffer = this.buffer.substring(contentBegPos + contents.length);
    return JSON.parse(contents);
  }

  public dispose() {
    this.enabled = false;
  }
}

/**
 * A stream reader that reports whenever a given process exists. Its underlying
 * callback will be invoked at most once.
 */
export class ProcessExitStream {
  private enabled = true;

  constructor(process: ChildProcess, onExit: (status: ExitStatus) => void) {
    process.on('exit', (code: number | null, signal: NodeJS.Signals | null) => {
      if (!this.enabled) {
        return;
      }
      onExit({ code, signal });
    });
  }

  public dispose() {
    this.enabled = false;
  }
}
