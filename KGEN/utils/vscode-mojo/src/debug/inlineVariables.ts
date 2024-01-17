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
 *  Variables grouped by evaluate name.
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
   * represent shadowed variables and they are sorted by declaration. Higher
   * columns come first in the group, whereas variables without declaration line
   * come last.
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

      for (const variables of variablesMap.values()) {
        variables.sort((v1: Variable, v2: Variable): number => {
          // If v1 has no decl, it comes last.
          if (v1.declaration === undefined || v1.declaration.line === undefined)
            return 1;

          // If v2 has no decl, it comes last.
          if (v2.declaration === undefined || v2.declaration.line === undefined)
            return -1;

          // The one with the largest line number comes first.
          return v2.declaration.line - v1.declaration.line;
        });
      }
      const frameId =
          this.variablesRequestIdToFrameId.get(response.request_seq)!
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
                                    variable: Variable,
                                    shadowed: boolean): vscode.InlineValueText {
    let displayName = variable.evaluateName;
    if (shadowed) {
      if (variable?.declaration?.line !== undefined)
        displayName += ` @ ${variable.declaration.line}`;
      else
        displayName = variable.name;
    }
    const range = new vscode.Range(line, column, line,
                                   column + variable.evaluateName.length);
    return new vscode.InlineValueText(range,
                                      `${displayName} = ${variable.value}`);
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
      this.context.getLoggingService().logError(
          `Couldn't find the local variable tracker for sessionId ${
              vscode.debug.activeDebugSession?.id} and frameId ${
              context.frameId}.`);
      return [];
    }

    const variableGroups = await tracker.waitForFrameVariables(context.frameId);

    const allValues: vscode.InlineValue[] = [];
    for (const variables of variableGroups.values()) {
      const shadowed = variables.length > 1;
      let prevBeginLine = Number.MAX_SAFE_INTEGER;

      // This list is sorted decrementally in terms of line number
      for (const variable of variables) {
        if (variable.declaration?.line === undefined ||
            variable.failedValueError !== undefined)
          continue;
        // We perform a text search of the variable name within a range that
        // goes from the declaration line up to where the previous shadowed
        // variable was declared, or the current breakpoint stop.

        const searchBeginLine = variable.declaration.line - 1;
        const searchEndLine =
            Math.min(prevBeginLine - 1, context.stoppedLocation.end.line);
        for (let line = searchBeginLine; line <= searchEndLine; line++) {
          const text = document.lineAt(line).text;
          const re = RegExp(variable.evaluateName, "g");
          do {
            var match = re.exec(text);
            if (match)
              allValues.push(this.createInlineVariableValue(
                  line, match.index, variable, shadowed));
          } while (match);
        }
        prevBeginLine = searchBeginLine;
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