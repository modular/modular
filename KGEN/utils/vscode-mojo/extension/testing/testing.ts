//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import { execFile } from 'child_process';
import * as vscode from 'vscode';
import { MAXSDK } from '../sdk/sdk';
import * as config from '../utils/config';
import { DisposableContext } from '../utils/disposableContext';
import * as path from 'path';
import { MAXSDKManager } from '../sdk/sdkManager';
import { Logger } from '../logging';

/**
 * An interface defining a source range for a mojo test.
 */
interface MojoSourceRange {
  startLine: number;
  startColumn: number;
  endLine: number;
  endColumn: number;
}

/**
 * An interface defining a mojo test.
 */
interface MojoTest {
  id: string;
  location: Optional<MojoSourceRange>;
  children: Optional<MojoTest[]>;
}

/**
 * An interface defining the result of a mojo test execution.
 */
interface MojoTestExecutionResult {
  kind: string;
  testID: string;
  duration_ms: number;

  error: string;
  stdOut: string;
  stdErr: string;

  children: Optional<MojoTestExecutionResult[]>;
}

/**
 * Class used to register and manage all the necessary constructs to support
 * mojo testing.
 */
export class MojoTestManager extends DisposableContext {
  private sdkManager: MAXSDKManager;
  private controller: vscode.TestController;
  private logger: Logger;

  // A tag used to mark doc tests.
  private docTestTag = new vscode.TestTag('docTest');
  private unitTestTag = new vscode.TestTag('unitTest');

  constructor(sdkManager: MAXSDKManager, logger: Logger) {
    super();
    this.sdkManager = sdkManager;
    this.logger = logger;

    // Register the mojo test controller.
    this.controller = vscode.tests.createTestController(
      'mojoTests',
      'Mojo Tests',
    );
    this.pushSubscription(this.controller);

    // Create the different test profiles.
    this.controller.createRunProfile(
      'Run',
      vscode.TestRunProfileKind.Run,
      (request, token) => {
        this.runHandler(/*shouldDebug=*/ false, request, token);
      },
    );
  }

  /**
   * Activate the mojo test context.
   */
  async activate() {
    this.pushSubscription(
      vscode.workspace.onDidOpenTextDocument((event) => {
        this.discoverTestsInDocument(event);
      }),
    );
    this.pushSubscription(
      vscode.workspace.onDidSaveTextDocument((event) => {
        this.discoverTestsInDocument(event);
      }),
    );
    this.pushSubscription(
      vscode.workspace.onDidCloseTextDocument((event) => {
        this.controller.items.delete(event.uri.fsPath);
      }),
    );
    // Process any existing documents.

    // Process any existing documents.
    for (const textDoc of vscode.workspace.textDocuments) {
      this.discoverTestsInDocument(textDoc);
    }
  }

  /**
   * Handle the given run request.
   *
   * @param shouldDebug Whether to run the tests in debug mode.
   * @param request The test run request.
   * @param token The cancellation token.
   */
  async runHandler(
    _shouldDebug: boolean,
    request: vscode.TestRunRequest,
    token: vscode.CancellationToken,
  ) {
    const queue: vscode.TestItem[] = [];

    // Loop through all included tests, or all known tests, and add them to our
    // queue.

    // Loop through all included tests, or all known tests, and add them to our
    // queue.
    if (request.include) {
      request.include.forEach((test) => queue.push(test));
    } else {
      this.controller.items.forEach((test) => queue.push(test));
    }

    // A set of doc tests that we have seen, mapped to the final code block
    // to test.

    // A set of doc tests that we have seen, mapped to the final code block
    // to test.
    const docTests = new Map<vscode.TestItem, number>();
    const unitTests = new Set<vscode.TestItem>();

    // Process the queue to collect the tests we want to run.
    const includedTests: Set<vscode.TestItem> = new Set();
    while (queue.length > 0 && !token.isCancellationRequested) {
      const test = queue.pop()!;

      // Skip tests the user asked to exclude

      // Skip tests the user asked to exclude
      if (request.exclude?.includes(test)) {
        continue;
      }

      // Doc tests are a bit special because each code block is treated as a
      // separate test, but we want to process them all together in the same
      // run.

      // Doc tests are a bit special because each code block is treated as a
      // separate test, but we want to process them all together in the same
      // run.
      if (test.tags.includes(this.docTestTag)) {
        // Track the latest code block we actually want to test. We can find
        // this by inspecting the label, which is the index of the code block.
        let testIndex = parseInt(test.label);

        if (testIndex > (docTests.get(test.parent!) ?? -1)) {
          docTests.set(test.parent!, testIndex);
        }
      } else if (test.tags.includes(this.unitTestTag)) {
        unitTests.add(test);
        includedTests.add(test);
      } else {
        includedTests.add(test);
        test.children.forEach((test) => queue.push(test));
      }
    }

    // Include any additional doc tests we found that are dependencies of
    // included tests.
    for (const [test, latestCodeBlockIdx] of docTests) {
      if (token.isCancellationRequested) {
        break;
      }

      for (let i = 0; i <= latestCodeBlockIdx; ++i) {
        includedTests.add(test.children.get(test.id + '::' + i.toString())!);
      }
    }

    // Build a new request that contains the expanded set of included tests.
    let excludedTests: vscode.TestItem[] = [];
    if (request.exclude) {
      request.exclude.forEach((test) => {
        if (!includedTests.has(test)) {
          excludedTests.push(test);
        }
      });
    }
    request = new vscode.TestRunRequest(
      Array.from(includedTests.keys()),
      excludedTests,
      request.profile,
    );
    const run = this.controller.createTestRun(request);

    // Process the doc tests collected so far.
    let testPromises = [];
    for (const [test, childIdx] of docTests) {
      if (token.isCancellationRequested) {
        break;
      }

      // The predecessor doc tests are implicit-dependencies, and will be run
      // automatically.

      // The predecessor doc tests are implicit-dependencies, and will be run
      // automatically.
      const dependencies: vscode.TestItem[] = [];

      for (let i = 0; i < childIdx; ++i) {
        dependencies.push(test.children.get(test.id + '::' + i.toString())!);
      }
      testPromises.push(
        this.executeTest(
          run,
          test.children.get(test.id + '::' + childIdx.toString())!,
          dependencies,
        ),
      );
    }

    // Process the unit tests collected so far.
    unitTests.forEach(async (test) => {
      if (!token.isCancellationRequested) {
        testPromises.push(this.executeTest(run, test));
      }
    });

    // Wait for the tests to finish executing.
    await Promise.all(testPromises);
    run.end();
  }

  /**
   * Execute the given test, updating its status (and the status of its
   * dependencies) as we go.
   */
  async executeTest(
    run: vscode.TestRun,
    test: vscode.TestItem,
    dependencies: vscode.TestItem[] = [],
  ) {
    let allTests = [...dependencies, test];
    for (const test of allTests) {
      run.enqueued(test);
      run.started(test);
    }

    // A utility functor to mark all of the tests as errored.
    const markAllTestsErrored = (message: string) => {
      for (const test of allTests) {
        run.errored(test, new vscode.TestMessage(message));
      }
    };

    // Grab the sdk for the execution context.
    let sdk = await this.sdkManager.findSDK(/*hideRepeatedErrors=*/ false);
    if (!sdk) {
      this.controller.items.delete(test.uri!.fsPath);
      return;
    }

    // Invoke the `test` subcommand of the mojo tool to discover tests in the
    // document.
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(test.uri!);
    let result = await this.runMojoTestCommand<MojoTestExecutionResult>(
      sdk,
      test.id,
      workspaceFolder,
      /*args=*/ [],
      /*withTelemetry=*/ true,
    );
    if (!result) {
      markAllTestsErrored('fatal error: unable to process test execution');
      return;
    }

    // Build a map of the results keyed by the id of the test.
    let resultsPerTest = new Map<string, MojoTestExecutionResult>();
    resultsPerTest.set(result.testID, result);

    for (const child of result.children ?? []) {
      resultsPerTest.set(child.testID, child);
    }

    // Process the tests.

    // Process the tests.
    for (const test of allTests) {
      let result = resultsPerTest.get(test.id);
      if (!result) {
        run.errored(
          test,
          new vscode.TestMessage('fatal error: test not found'),
        );
        continue;
      }

      if (result.kind === 'success') {
        run.passed(test, result.duration_ms);
      } else if (result.kind === 'skipped') {
        run.skipped(test);
      } else {
        let message = result.error;

        if (result.stdErr.length > 0) {
          message += '\n' + result.stdErr + '\n';
        }

        // TODO: We only add stdout right now because we don't have a nice
        // way of printing exceptions to stderr.

        // TODO: We only add stdout right now because we don't have a nice
        // way of printing exceptions to stderr.

        // TODO: We only add stdout right now because we don't have a nice
        // way of printing exceptions to stderr.
        if (result.stdOut.length > 0) {
          message += '\n' + result.stdOut + '\n';
        }

        run.failed(test, [new vscode.TestMessage(message)], result.duration_ms);
      }
    }
  }

  /**
   * Invoke the `test` subcommand of the mojo tool with the given
   * arguments. Returns the json output of running the command.
   */
  async runMojoTestCommand<Result>(
    sdk: MAXSDK,
    testId: string,
    workspaceFolder: Optional<vscode.WorkspaceFolder>,
    args: string[] = [],
    withTelemetry: boolean,
  ): Promise<Optional<Result>> {
    // Grab any additional include directories from the workspace settings.
    const includeDirs =
      config.get<Optional<string[]>>('lsp.includeDirs', workspaceFolder) || [];

    for (const includeDir of includeDirs) {
      args.push('-I', includeDir);
    }

    let env = sdk.getProcessEnv(withTelemetry);
    const logger = this.logger;

    return new Promise<Optional<Result>>(function (resolve, _reject) {
      execFile(
        sdk.config.mojoDriverPath,
        ['test', '--diagnostic-format', 'json', testId, ...args],
        { env },
        (_error, stdout, _stderr) => {
          // Parse the json output from the stdout.
          try {
            resolve(JSON.parse(stdout));
          } catch (e) {
            logger.error(
              `Received invalid JSON response from mojo CLI\n${stdout}`,
            );
            resolve(undefined);
          }
        },
      );
    });
  }

  /**
   * Process the given document for tests.
   */
  async discoverTestsInDocument(document: vscode.TextDocument) {
    if (document.languageId !== 'mojo' || document.isDirty) {
      return;
    }

    this.logger.debug(`Discovering tests in ${document.uri}`);

    // Invoke the mojo tool to discover tests in the document.
    // We use 'hideRepeatedErrors' because this action is automated.
    let sdk = await this.sdkManager.findSDK(/*hideRepeatedErrors=*/ true);
    if (!sdk) {
      this.controller.items.delete(document.uri.fsPath);
      this.logger.debug(`No SDK present, clearing tests for ${document.uri}`);
      return;
    }

    // Invoke the `test` subcommand of the mojo tool to discover tests in the
    // document.
    const workspaceFolder = vscode.workspace.getWorkspaceFolder(document.uri);
    let mojoTestSuite = await this.runMojoTestCommand<MojoTest>(
      sdk,
      document.uri.fsPath,
      workspaceFolder,
      ['--co'],
      /*withTelemetry=*/ false,
    );
    if (!mojoTestSuite || !mojoTestSuite.children) {
      this.controller.items.delete(document.uri.fsPath);
      return;
    }

    // Create a new test suite for the file if necessary.
    let file = this.controller.items.get(document.uri.fsPath);
    if (!file) {
      file = this.controller.createTestItem(
        document.uri.fsPath,
        document.uri.fsPath.split(path.sep).pop()!,
        document.uri,
      );
      this.controller.items.add(file);
    }

    // Add the tests to the file.
    this.populateTests(mojoTestSuite, file, document);
  }

  /**
   * Recursively populate the tests in the given mojo test suite.
   */
  populateTests(
    parent: MojoTest,
    parentVSTest: vscode.TestItem,
    document: vscode.TextDocument,
  ) {
    if (!parent.children) {
      return;
    }
    let vsChildren: vscode.TestItem[] = [];
    for (let test of parent.children) {
      let label = test.id.substring(parent.id.length);
      let tags: vscode.TestTag[] = [];
      if (label.startsWith('@')) {
        label = label.substring(1);
      } else if (label.startsWith('::')) {
        label = label.substring(2);

        // Add the proper tag if this is a doc test.

        // Add the proper tag if this is a doc test.
        if (test.id.includes('__doc__::')) {
          tags.push(this.docTestTag);
        } else {
          tags.push(this.unitTestTag);
        }
      }

      let vsTest = this.controller.createTestItem(test.id, label, document.uri);
      if (test.location) {
        vsTest.range = new vscode.Range(
          new vscode.Position(
            test.location.startLine - 1,
            test.location.startColumn - 1,
          ),
          new vscode.Position(
            test.location.endLine - 1,
            test.location.endColumn - 1,
          ),
        );
      }
      vsTest.tags = tags;
      this.populateTests(test, vsTest, document);

      vsChildren.push(vsTest);
    }
    parentVSTest.children.replace(vsChildren);
  }
}
