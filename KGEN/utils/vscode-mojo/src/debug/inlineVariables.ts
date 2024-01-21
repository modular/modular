//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as vscode from 'vscode';

import {MOJOContext} from '../mojoContext';
import {DisposableContext} from '../utils/disposableContext';

import {DEBUG_TYPE} from './constants';

/**
 *  Variables grouped by evaluate name. Multiple entries per key represent
 * shadowed variables.
 */
type VariablesGroups = Map<VariableEvaluateName, Variable[]>;

/**
 * Class that tracks the local variables of every frame by inspecting the DAP
 * messages.
 *
 * The only interesting detail is that the "variables" request doesn't have a
 * `frameId`. Instead, this request is followed by the "scopes" request, which
 * does have a `frameId`, so we keep an eye on this successive pair of requests
 * to produce the appropriate mapping.
 */
export class LocalVariablesTracker implements vscode.DebugAdapterTracker {
  /**
   * The current `frameId` gotten from the last "scopes" request.
   */
  private currentFrameId: FrameId = -1;
  /**
   * A mapping from frameId to a grouped list of variables. These groups
   * represent shadowed variables.
   */
  public frameToVariables = new Map<FrameId, VariablesGroups>();
  /**
   * A mapping that helps us identify which frameId corresponds to a given
   * variables request.
   */
  public variablesRequestIdToFrameId = new Map<RequestId, FrameId>();
  /**
   * This is a hardcoded value in lldb-dap that represents the list of local
   * variables.
   */
  private static LOCAL_SCOPE_ID = 1;
  public onFrameGotVariables =
      new vscode.EventEmitter<[ FrameId, VariablesGroups ]>();

  async waitForFrameVariables(frameId: FrameId): Promise<VariablesGroups> {
    const result = this.frameToVariables.get(frameId);
    if (result !== undefined)
      return result;

    return new Promise<VariablesGroups>((resolve, reject) => {
      this.onFrameGotVariables.event(([ eventFrameId, variables ]) => {
        if (eventFrameId == frameId)
          resolve(variables);
      });
    });
  }

  onWillReceiveMessage(message: {[key: string]: unknown}): void {
    if (message.command === "scopes") {
      this.currentFrameId = (message as DAPScopesRequest).arguments.frameId;
    } else if (message.command === "variables") {
      const request = message as DAPVariablesRequest;
      if (request.arguments.variablesReference ===
          LocalVariablesTracker.LOCAL_SCOPE_ID) {
        this.variablesRequestIdToFrameId.set(request.seq, this.currentFrameId);
      }
    }
  }

  onDidSendMessage(message: {[key: string]: unknown}): void {
    if (message.event === "stopped") {
      this.currentFrameId = -1;
      this.frameToVariables.clear();
      this.variablesRequestIdToFrameId.clear();
    }

    if (message.command === "variables") {
      const response = message as DAPVariablesResponse;
      const variablesMap: VariablesGroups = new Map();

      for (const variable of response.body.variables) {
        if (!variablesMap.has(variable.evaluateName))
          variablesMap.set(variable.evaluateName, []);
        variablesMap.get(variable.evaluateName)!.push(variable);
      }

      const frameId =
          this.variablesRequestIdToFrameId.get(response.request_seq)!;
      this.frameToVariables.set(frameId, variablesMap);
      this.onFrameGotVariables.fire([ frameId, variablesMap ]);
    }
  }
}

/**
 * Provides inline local variables during a debug session.
 */
export class InlineLocalVariablesProvider implements
    vscode.InlineValuesProvider {
  private localVariablesTrackers: Map<SessionId, LocalVariablesTracker>;
  private context: MOJOContext;

  constructor(context: MOJOContext,
              localVariablesTrackers: Map<SessionId, LocalVariablesTracker>) {
    this.context = context;
    this.localVariablesTrackers = localVariablesTrackers;
  }

  /**
   * Create the inline text to show for the given variable.
   */
  private createInlineVariableValue(line: number, column: number,
                                    variable: Variable):
      vscode.InlineValueText {
    let displayName = variable.evaluateName;
    const range = new vscode.Range(line, column, line,
                                   column + variable.evaluateName.length);
    return new vscode.InlineValueText(range,
                                      `${displayName} = ${variable.value}`);
  }

  /**
   * Find the column in the document where the given variable is declared.
   * Currently DWARF doesn't have columns (#29230), so we have to look for the
   * declaration column using text search in the document.
   */
  private findDeclColumn(document: vscode.TextDocument, line: number,
                         variable: Variable): number|undefined {
    const text = document.lineAt(line).text;
    let index = -1;

    // This is used to verify that a candidate declaration for our variable
    // cannot be expanded into a larger variable name.
    const forbiddenBoundary = (char?: string) =>
        char !== undefined && /^[a-zA-Z0-9_]$/.test(char);

    do {
      index = text.indexOf(variable.evaluateName, index + 1);
      if (index == -1)
        break;

      const prev = text[index - 1];
      const next = text[index + variable.evaluateName.length];
      if (!forbiddenBoundary(prev) && !forbiddenBoundary(next))
        return index;
    } while (true);

    return undefined;
  }

  /**
   * Create the list of inline values for a given variable using the LSP's index
   * of references.
   */
  async getInlineValuesForVariable(document: vscode.TextDocument,
                                   stoppedLocation: vscode.Range,
                                   variable: Variable):
      Promise<vscode.InlineValue[]> {
    const decl = variable.$__lldb_extensions.declaration;
    const error = variable.$__lldb_extensions.error || "";
    const path = decl?.path || "";
    if (decl?.line === undefined || path.length == 0 || error.length > 0)
      return [];
    const line = decl.line - 1;
    // If the decl line is where we are stopped or later, we don't inline the
    // variable to prevent printing dirty memory.
    if (line >= stoppedLocation.start.line)
      return [];
    let column = this.findDeclColumn(document, line, variable);

    // If there's no column information, we can at least show the variable in
    // the decl line.
    if (column === undefined) {
      return [ this.createInlineVariableValue(line, 0, variable) ];
    }

    const uri = vscode.Uri.file(path);
    const lspServer = await this.context.getOrActivateLanguageClient(
        uri, /*launchLanguageServerSuspended=*/ false);
    if (lspServer === undefined)
      return [];

    const references: undefined|any[] =
        await lspServer.sendRequest("textDocument/references", {
          textDocument : {
            uri : uri.toString(),
          },
          context : {includeDeclaration : true},
          position : {
            line : line,
            character : column,
          }
        });
    return (references || [])
        .map(ref => this.createInlineVariableValue(
                 ref.range.start.line, ref.range.start.character, variable))
        .filter(
            // We only keep the references that are on the stop line or above.
            inlineVar =>
                inlineVar.range.start.line <= stoppedLocation.start.line);
  }

  async provideInlineValues(document: vscode.TextDocument,
                            _viewport: vscode.Range,
                            context: vscode.InlineValueContext):
      Promise<vscode.InlineValue[]> {
    const tracker = this.localVariablesTrackers.get(
        vscode.debug.activeDebugSession?.id || "");
    if (tracker === undefined) {
      // This could be a non-bug if there are two simultaneous debug sessions
      // with different debuggers.
      this.context.loggingService.logError(
          `Couldn't find the local variable tracker for sessionId ${
              vscode.debug.activeDebugSession?.id} and frameId ${
              context.frameId}.`);
      return [];
    }

    const variableGroups = await tracker.waitForFrameVariables(context.frameId);

    const allValues: vscode.InlineValue[] = [];
    for (const variables of variableGroups.values()) {
      for (const variable of variables) {
        allValues.push(...await this.getInlineValuesForVariable(
            document, context.stoppedLocation, variable));
      }
    }
    return allValues;
  }
}

export function initializeInlineLocalVariablesProvider(context: MOJOContext):
    DisposableContext {
  const localVariablesTrackers: Map<SessionId, LocalVariablesTracker> =
      new Map();
  const disposables = new DisposableContext();

  disposables.pushSubscription(vscode.debug.registerDebugAdapterTrackerFactory(
      DEBUG_TYPE, <vscode.DebugAdapterTrackerFactory>{
        createDebugAdapterTracker(session: vscode.DebugSession) :
            vscode.ProviderResult<vscode.DebugAdapterTracker> {
              const tracker = new LocalVariablesTracker();
              localVariablesTrackers.set(session.id, tracker);
              return tracker;
            }
      }));
  disposables.pushSubscription(vscode.debug.onDidTerminateDebugSession(
      (session: vscode
           .DebugSession) => { localVariablesTrackers.delete(session.id); }));

  disposables.pushSubscription(vscode.languages.registerInlineValuesProvider(
      "*", new InlineLocalVariablesProvider(context, localVariablesTrackers)));
  return disposables;
}