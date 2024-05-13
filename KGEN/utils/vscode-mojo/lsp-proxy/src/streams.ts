//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import {JSONObject} from "./types";

/**
 * A stream reader that reports whenever a line ending with `\n` is found.
 */
export class LineSeparatedStream {
  constructor(rawStream: NodeJS.ReadableStream,
              onLine: (line: string) => void) {
    let buffer = "";
    rawStream.on("data", (chunk: any) => {
      buffer += chunk;

      let newLinePos = -1;
      while ((newLinePos = buffer.indexOf('\n')) !== -1) {
        const line = buffer.substring(0, newLinePos);
        buffer = buffer.substring(newLinePos + 1);
        onLine(line);
      }
    });
  }
}

/**
 * A stream reader based on the JSON-RPC protocol that reports whenever a
 * notification or the response to a request is found.
 */
export class JSONRPCStream {
  static protocolHeader = "Content-Length: ";
  static protocolLineSeparator = "\r\n\r\n";
  private buffer = "";

  constructor(rawStream: NodeJS.ReadableStream,
              onResponse: (response: JSONObject) => void,
              onNotification: (notification: JSONObject) => void) {

    rawStream.on("data", (chunk: any) => {
      this.buffer += chunk;

      let packet: JSONObject|undefined;
      while ((packet = this.tryProcessPacket()) != undefined) {
        if ("id" in packet)
          onResponse(packet);
        else
          onNotification(packet);
      }
      return true;
    });
  }

  /**
   * Tries to read a packet from the buffer and update that buffer if found.
   */
  private tryProcessPacket(): JSONObject|undefined {
    // We process first the protocol header.
    if (!this.buffer.startsWith(JSONRPCStream.protocolHeader))
      return undefined;
    // Then we parse the content length.
    let index = JSONRPCStream.protocolHeader.length;
    let contentLength = 0;
    for (; index < this.buffer.length; index++) {
      const c = this.buffer[index];
      if (c < '0' || c > '9')
        break;
      contentLength = contentLength * 10 + parseInt(c);
    }
    // Then we parse the line separator.
    if (!this.buffer.substring(index).startsWith(
            JSONRPCStream.protocolLineSeparator))
      return undefined;

    // Then we extract the contents of the packet.
    const contentBegPos = index + JSONRPCStream.protocolLineSeparator.length;
    const contents =
        this.buffer.substring(contentBegPos, contentBegPos + contentLength);
    if (contents.length != contentLength)
      return undefined;

    // We update the buffer to point past this packet.
    this.buffer = this.buffer.substring(contentBegPos + contentLength);
    return JSON.parse(contents);
  }
}
