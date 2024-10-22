//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import { reporters, Runner, Test } from 'mocha';

import { setLogHook } from '../extension';

const {
  EVENT_TEST_BEGIN,
  EVENT_SUITE_BEGIN,
  EVENT_SUITE_END,
  EVENT_TEST_PASS,
  EVENT_TEST_PENDING,
  EVENT_TEST_FAIL,
  EVENT_RUN_END,
} = Runner.constants;
const Base = reporters.Base;

class VsCodeReporter extends Base {
  constructor(runner: Runner) {
    super(runner);

    const testLogs: { [id: string]: string[] } = {};
    let indentDepth = 0;

    const indent = (str: string) => {
      let result = '';
      for (const line of str.split('\n')) {
        result += ' '.repeat(indentDepth) + line + '\n';
      }
      return result.trimEnd();
    };

    runner.on(EVENT_SUITE_BEGIN, (suite) => {
      console.log(indent(Base.color('suite', suite.title)));
      indentDepth += 2;
    });

    runner.on(EVENT_SUITE_END, () => {
      indentDepth -= 2;
    });

    runner.on(EVENT_TEST_BEGIN, (test: Test) => {
      const testId = test.fullTitle();
      testLogs[testId] = [];

      const callback = (level: string, message: string) =>
        testLogs[testId].push(`[${level.padEnd(5, ' ')}]: ${message}`);
      setLogHook(callback);
    });

    runner.on(EVENT_TEST_PASS, (test: Test) => {
      console.log(
        indent(
          `${Base.color('checkmark', Base.symbols.ok)} ${Base.color('pass', test.title)}`,
        ),
      );
    });

    runner.on(EVENT_TEST_FAIL, (test: Test, err) => {
      console.log(
        indent(
          `${Base.color('fail', Base.symbols.err)} ${Base.color('fail', test.title)}`,
        ),
      );
      indentDepth += 2;

      if (err.stack) {
        console.log(indent(`${Base.color('error stack', err.stack)}`));
      } else {
        console.log(indent(`${Base.color('error title', err.name)}`));
        console.log(indent(`${Base.color('error message', err.message)}`));
      }

      console.log(indent('\nExtension logs:\n'));

      const logs = testLogs[test.fullTitle()];
      for (const log of logs) {
        console.log(indent(log));
      }

      indentDepth -= 2;
    });

    runner.on(EVENT_TEST_PENDING, (test: Test) => {
      console.log(indent(`${Base.color('pending', test.title)}`));
    });

    runner.on(EVENT_RUN_END, () => {
      this.epilogue();
    });
  }
}

module.exports = VsCodeReporter;
