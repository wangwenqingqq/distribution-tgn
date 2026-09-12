#!/usr/bin/env python3
"""Check staged vault content for common accidental disclosures and large files."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 5 * 1024 * 1024
PATTERNS = [
    rb'-----BEGIN [A-Z ]*PRIVATE KEY-----',
    rb'gh[pousr]_[A-Za-z0-9_]{20,}',
    rb'github_pat_[A-Za-z0-9_]{20,}',
    rb'AKIA[A-Z0-9]{16}',
    rb'sk-[A-Za-z0-9_-]{32,}',
    rb'xox[baprs]-[A-Za-z0-9-]{20,}',
]

def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])

def history_check():
    errors = []
    seen = set()
    commits = git('rev-list', 'HEAD').decode().splitlines()
    attribution = re.compile('cla' + 'ude|anthro' + 'pic|co-' + 'authored', re.I)
    metadata = git('log', '--format=%an|%ae|%cn|%ce|%B', 'HEAD').decode('utf-8', errors='replace')
    if attribution.search(metadata):
        errors.append(('commit metadata', 'unexpected attribution; inspect before publishing'))
    for commit in commits:
        for entry in filter(None, git('ls-tree', '-r', '-z', commit).split(b'\0')):
            header, raw = entry.split(b'\t', 1)
            mode, kind, oid = header.split()
            name = raw.decode('utf-8')
            if (name, mode, oid) in seen:
                continue
            seen.add((name, mode, oid))
            parts = Path(name).parts
            base = Path(name).name.lower()
            if mode in {b'120000', b'160000'}:
                errors.append((name, 'symlink or submodule in history'))
                continue
            if any(p in {'.obsidian', '_local', '.ssh', '.git', 'datasets', 'checkpoints'} for p in parts) or base == '.env' or base.startswith(('.env.', 'credentials', 'cookies', 'secrets', 'id_rsa', 'id_ed25519')) or Path(name).suffix.lower() in {'.pem', '.key', '.pt', '.pth', '.ckpt', '.safetensors'}:
                errors.append((name, 'restricted path in history'))
                continue
            obj = oid.decode()
            if int(git('cat-file', '-s', obj)) > LIMIT:
                errors.append((name, 'file exceeds 5 MiB in history'))
                continue
            data = git('cat-file', 'blob', obj)
            if any(re.search(pattern, data) for pattern in PATTERNS):
                errors.append((name, 'possible secret in history; inspect locally'))
    for name, reason in errors:
        print('BLOCKED:', name, '-', reason)
    if not errors:
        print('History checks passed:', len(commits), 'commits and', len(seen), 'file versions.')
    return int(bool(errors))

def main():
    paths = git('diff', '--cached', '--name-only', '--diff-filter=ACMRT', '-z').split(b'\0')
    errors = []
    for raw in filter(None, paths):
        name = raw.decode('utf-8')
        parts = Path(name).parts
        if any(p in {'.obsidian', '_local', '.ssh', '.git', 'datasets', 'checkpoints'} for p in parts):
            errors.append((name, 'device-local or restricted path'))
            continue
        base = Path(name).name.lower()
        if base == '.env' or base.startswith(('.env.', 'credentials', 'cookies', 'secrets', 'id_rsa', 'id_ed25519')) or Path(name).suffix.lower() in {'.pem', '.key', '.pt', '.pth', '.ckpt', '.safetensors'}:
            errors.append((name, 'credential or model file name'))
            continue
        mode = git('ls-files', '--stage', '--', name).split(b' ', 1)[0]
        if mode in {b'120000', b'160000'}:
            errors.append((name, 'symlink or submodule needs manual review'))
            continue
        size = int(git('cat-file', '-s', ':' + name))
        if size > LIMIT:
            errors.append((name, 'file exceeds 5 MiB; store outside this vault'))
            continue
        data = git('show', ':' + name)
        if any(re.search(pattern, data) for pattern in PATTERNS):
            errors.append((name, 'possible secret; inspect locally, do not paste it into chat'))
    if errors:
        for name, reason in errors:
            print('BLOCKED:', name, '-', reason)
        return 1
    print('Staged checks passed. This is a heuristic check, not a guarantee; review the diff.')
    return 0

if __name__ == '__main__':
    sys.exit(history_check() if '--history' in sys.argv else main())
