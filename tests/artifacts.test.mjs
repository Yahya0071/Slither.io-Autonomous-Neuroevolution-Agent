/**
 * Build-artifact tests.
 *
 * The same 60KB agent ships in five places. These tests make silent drift a
 * failure instead of a surprise: if you edit console_inject.js and forget to run
 * `npm run build`, or if you hand-edit a generated copy, this goes red.
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { projectRoot, sourceCode, sourcePath } from '../agent-loader.mjs';

const read = (rel) => readFileSync(join(projectRoot, rel), 'utf8');
const pkg = JSON.parse(read('package.json'));

test('console_inject.js is the single source of truth and can be inlined', () => {
  assert.ok(existsSync(sourcePath));
  assert.ok(sourceCode.length > 1000);
  assert.ok(!sourceCode.includes('</script>'), 'a literal </script> would break HTML inlining');
  assert.doesNotThrow(() => {
    new Function(sourceCode);
  }, 'console_inject.js must parse with zero syntax errors');
});

test('extension/agent.js is byte-identical to the source', () => {
  assert.equal(read('extension/agent.js'), sourceCode);
});

test('the userscript is the source plus a userscript header only', () => {
  const user = read('slither_ml_agent.user.js');
  assert.ok(user.endsWith(sourceCode), 'userscript body must equal the source verbatim');

  const header = user.slice(0, user.length - sourceCode.length);
  assert.match(header, /^\/\/ ==UserScript==\n/, 'header must open the userscript block');
  assert.match(header, /\/\/ ==\/UserScript==\n\n$/, 'header must close the userscript block and end with a blank line');
  assert.doesNotMatch(header, /console\.log/, 'no agent code may live in the header');
});

test('quick_start.html inlines both variants exactly', () => {
  const html = read('quick_start.html');

  const consoleBlock = html.match(/<script type="text\/plain" id="codeConsole">([\s\S]*?)<\/script>/);
  assert.ok(consoleBlock, '#codeConsole block must exist');
  assert.equal(consoleBlock[1], sourceCode, 'inlined console copy is stale - run `npm run build`');

  const userBlock = html.match(/<script type="text\/plain" id="codeUserscript">([\s\S]*?)<\/script>/);
  assert.ok(userBlock, '#codeUserscript block must exist');
  assert.equal(userBlock[1], read('slither_ml_agent.user.js'), 'inlined userscript copy is stale - run `npm run build`');
});

test('all four artifacts are identical to each other', () => {
  const html = read('quick_start.html');
  const consoleBlock = html.match(/<script type="text\/plain" id="codeConsole">([\s\S]*?)<\/script>/)[1];
  const userBlock = html.match(/<script type="text\/plain" id="codeUserscript">([\s\S]*?)<\/script>/)[1];
  const userFile = read('slither_ml_agent.user.js');

  assert.equal(read('extension/agent.js'), consoleBlock);
  assert.equal(userFile, userBlock);
  assert.equal(consoleBlock.length, sourceCode.length);
});

test('the version is consistent across every manifest', () => {
  const manifest = JSON.parse(read('extension/manifest.json'));
  const user = read('slither_ml_agent.user.js');
  const declared = user.match(/^\/\/ @version\s+(\S+)$/m);

  assert.ok(declared, 'userscript must declare @version');
  assert.equal(declared[1], pkg.version, 'userscript @version must match package.json');
  assert.equal(manifest.version, pkg.version, 'extension manifest version must match package.json');
  assert.ok(
    sourceCode.includes(`v${pkg.version}`),
    `console_inject.js banner should mention v${pkg.version}`,
  );
});

test('the extension manifest still targets slither.io only', () => {
  const manifest = JSON.parse(read('extension/manifest.json'));
  assert.equal(manifest.manifest_version, 3);
  for (const pattern of manifest.host_permissions) {
    assert.match(pattern, /slither\.(io|com)/, `unexpected host permission: ${pattern}`);
  }
  assert.equal(manifest.content_scripts[0].js[0], 'content.js');
  assert.deepEqual(manifest.web_accessible_resources[0].resources, ['agent.js']);
});
