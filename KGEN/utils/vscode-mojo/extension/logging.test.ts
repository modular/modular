//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import * as assert from 'assert';
import { LogChannel, LogLevel } from './logging';

function createLogSpy(): [string[], (level: string, message: string) => void] {
  let lines: string[] = [];
  return [
    lines,
    (_level: string, message: string) => {
      lines.push(message);
    },
  ];
}

suite('Logging', () => {
  test('logs should respect output levels', () => {
    const channel = new LogChannel('Test Channel');
    const [lines, callback] = createLogSpy();
    channel.logCallback = callback;

    channel.setOutputLevel(LogLevel.None);
    channel.error('error');
    channel.warn('warn');
    channel.info('info');
    channel.debug('debug');
    channel.trace('trace');
    assert.deepStrictEqual(lines, []);
    lines.length = 0;

    channel.setOutputLevel(LogLevel.Error);
    channel.error('error');
    channel.warn('warn');
    channel.info('info');
    channel.debug('debug');
    channel.trace('trace');
    assert.deepStrictEqual(lines, ['error']);
    lines.length = 0;

    channel.setOutputLevel(LogLevel.Warn);
    channel.error('error');
    channel.warn('warn');
    channel.info('info');
    channel.debug('debug');
    channel.trace('trace');
    assert.deepStrictEqual(lines, ['error', 'warn']);
    lines.length = 0;

    channel.setOutputLevel(LogLevel.Info);
    channel.error('error');
    channel.warn('warn');
    channel.info('info');
    channel.debug('debug');
    channel.trace('trace');
    assert.deepStrictEqual(lines, ['error', 'warn', 'info']);
    lines.length = 0;

    channel.setOutputLevel(LogLevel.Debug);
    channel.error('error');
    channel.warn('warn');
    channel.info('info');
    channel.debug('debug');
    channel.trace('trace');
    assert.deepStrictEqual(lines, ['error', 'warn', 'info', 'debug']);
    lines.length = 0;

    channel.setOutputLevel(LogLevel.Trace);
    channel.error('error');
    channel.warn('warn');
    channel.info('info');
    channel.debug('debug');
    channel.trace('trace');
    assert.deepStrictEqual(lines, ['error', 'warn', 'info', 'debug', 'trace']);
    lines.length = 0;
  });

  test('data should be logged as JSON', () => {
    const channel = new LogChannel('Test Channel');
    const [lines, callback] = createLogSpy();
    channel.logCallback = callback;

    channel.setOutputLevel(LogLevel.Info);

    channel.info('message', { foo: 123, bar: true, baz: [1, 2, 3] });
    assert.equal(lines.length, 2);
    assert.equal(lines[0], 'message');

    const json = JSON.parse(lines[1]);
    assert.deepStrictEqual(json, {
      foo: 123,
      bar: true,
      baz: [1, 2, 3],
    });
  });

  test('data should respect log level', () => {
    const channel = new LogChannel('Test Channel');
    const [lines, callback] = createLogSpy();
    channel.logCallback = callback;
    channel.setOutputLevel(LogLevel.Info);

    channel.debug('message', { foo: 123 });
    assert.deepStrictEqual(lines, []);
  });
});
