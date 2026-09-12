#!/usr/bin/env python3
"""Interactively synchronize only this vault; stop safely on conflicting histories."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, GIT_SSH_COMMAND='ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=yes')

def git(*args, capture=False, check=True):
    result = subprocess.run(['git', '-C', str(ROOT), *args], env=ENV,
                            text=True, capture_output=capture, check=check)
    return result.stdout.strip() if capture else result.returncode

def stop(message):
    raise SystemExit(message)

def main():
    config = json.loads((ROOT / '99-System/repository.json').read_text())
    if Path(git('rev-parse', '--show-toplevel', capture=True)).resolve() != ROOT:
        stop('Wrong repository root; no changes made.')
    if git('branch', '--show-current', capture=True) != 'main':
        stop('Switch to the intended main branch manually; no changes made.')
    if git('remote', 'get-url', '--all', 'origin', capture=True).splitlines() != [config['ssh_url']] or git('remote', 'get-url', '--push', '--all', 'origin', capture=True).splitlines() != [config['ssh_url']]:
        stop('Unexpected remote URL; inspect it manually before syncing.')
    if git('diff', '--cached', '--name-only', capture=True):
        stop('Existing staged work found. Commit or unstage it yourself before syncing.')
    if git('ls-files', '--unmerged', capture=True):
        stop('Unresolved conflicts found. Resolve them manually first.')
    git_dir = Path(git('rev-parse', '--absolute-git-dir', capture=True))
    if any((git_dir / p).exists() for p in ('MERGE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD', 'rebase-merge', 'rebase-apply')):
        stop('Another Git operation is in progress. Finish it manually first.')
    print('Vault:', ROOT.name, '\nDestination:', config['web_url'])
    git('status', '--short')
    if not sys.stdin.isatty():
        stop('Preview only. Run Sync.command interactively to approve synchronization.')
    if input('Commit the listed changes and synchronize this vault? [y/N] ').strip().lower() != 'y':
        stop('Cancelled; no changes made.')
    git('fetch', '--prune', 'origin')
    if git('show-ref', '--verify', '--quiet', 'refs/remotes/origin/main', check=False) == 0:
        ahead, behind = map(int, git('rev-list', '--left-right', '--count', 'HEAD...origin/main', capture=True).split())
        if behind and (ahead or git('status', '--porcelain', capture=True)):
            stop('Remote changes overlap with local work. Nothing was staged or discarded; ask for a reviewed merge.')
        if behind:
            git('merge', '--ff-only', 'origin/main')
    git('add', '-A')
    if subprocess.run([sys.executable, str(ROOT / 'scripts/check.py')]).returncode:
        stop('Check failed. Changes remain staged for inspection; nothing was pushed.')
    git('diff', '--cached', '--check')
    if git('diff', '--cached', '--quiet', check=False) != 0:
        git('commit', '-m', 'docs(notes): update research notes')
    if subprocess.run([sys.executable, str(ROOT / 'scripts/check.py'), '--history']).returncode:
        stop('History check failed. Nothing was pushed; inspect the indicated file history.')
    git('push', '-u', 'origin', 'HEAD:main')
    local = git('rev-parse', 'HEAD', capture=True)
    remote = git('ls-remote', 'origin', 'refs/heads/main', capture=True).split()[0]
    if local != remote:
        stop('Remote changed during verification. Do not force push; inspect status.')
    print('Verified: local and remote main match at', local[:12])

if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as exc:
        stop('Synchronization stopped safely (exit %s). Inspect Git status before retrying.' % exc.returncode)
