# Git Workflow and Advanced Usage

## Branching Strategy

Use trunk-based development for fast-moving teams:

```bash
# Create feature branch from main
git checkout -b feature/user-auth main

# Keep branch up to date with main
git fetch origin
git rebase origin/main

# Squash commits before merging
git rebase -i HEAD~3
```

Branch naming conventions: `feature/`, `fix/`, `chore/`, `docs/`. Keep branch names short and descriptive.

## Interactive Rebase

Clean up commit history before merging:

```bash
git rebase -i HEAD~5
```

Commands in the rebase editor:
- `pick` — keep the commit as-is
- `squash` — merge into previous commit
- `fixup` — merge into previous, discard message
- `reword` — keep commit, edit message
- `drop` — remove the commit

## Git Bisect

Find the commit that introduced a bug:

```bash
git bisect start
git bisect bad           # current commit is broken
git bisect good v1.2.0   # known good commit

# Git checks out a commit — test it
git bisect good   # or git bisect bad

# Automate with a script
git bisect run ./test_script.sh

git bisect reset  # return to original branch
```

## Useful Aliases

Add to `~/.gitconfig`:

```ini
[alias]
    st = status -sb
    lg = log --oneline --graph --decorate -20
    amend = commit --amend --no-edit
    unstage = reset HEAD --
    last = log -1 HEAD --stat
    branches = branch -a --sort=-committerdate
```

## Git Hooks

Automate checks with hooks in `.git/hooks/` or use frameworks like `pre-commit`:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
```
