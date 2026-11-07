#!/usr/bin/env node
/**
 * Cross-platform dashboard launcher for quick_start.html (zero dependencies).
 */
import { spawn } from 'node:child_process';
import { platform } from 'node:os';
import { resolve } from 'node:path';

const target = resolve('quick_start.html');
const os = platform();

let cmd;
let args;

if (os === 'win32') {
  cmd = 'cmd.exe';
  args = ['/c', 'start', '""', target];
} else if (os === 'darwin') {
  cmd = 'open';
  args = [target];
} else {
  cmd = 'xdg-open';
  args = [target];
}

const child = spawn(cmd, args, { stdio: 'ignore', detached: true });
child.unref();
