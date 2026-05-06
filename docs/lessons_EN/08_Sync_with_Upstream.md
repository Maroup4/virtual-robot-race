# 8. Syncing with Upstream

The official aira repository is updated regularly with new features and bug fixes.
The repository you forked in [Lesson 00](00_Preparation.md) is a snapshot from the time of forking — it does not automatically receive updates from the official repo.

In this lesson, you will learn how to **pull in the latest official changes while keeping your own algorithm safe**.

**Learning objectives:**
- Understand the difference between `upstream` and `origin`
- Learn the safe workflow for pulling in official updates using `git fetch` + `git merge`
- Use `git stash` to shelve and restore work in progress
- Know how to handle conflicts if they arise

---

## What is Upstream?

In Git, remote repositories are managed by name.

| Name | Points to | Purpose |
|------|-----------|---------|
| `origin` | Your fork (on GitHub) | Where you save and share your work |
| `upstream` | The official aira repository | Where you fetch the latest simulator updates |

Here is how the three relate:

```
  Official aira repository (upstream)
        │
        │  git fetch upstream
        │  git merge upstream/main
        ▼
  Your local PC
        │
        │  git push origin main
        ▼
  Your GitHub fork (origin)
```

Right after forking, only `origin` exists. By manually adding `upstream`, you can pull in updates from the official repository.

---

## Steps

### Step 1: Add upstream (first time only)

```bash
git remote add upstream https://github.com/aira-race/virtual-robot-race.git
```

Verify it was added:

```bash
git remote -v
```

You should see something like:

```
origin    https://github.com/YOUR_USERNAME/virtual-robot-race.git (fetch)
origin    https://github.com/YOUR_USERNAME/virtual-robot-race.git (push)
upstream  https://github.com/aira-race/virtual-robot-race.git (fetch)
upstream  https://github.com/aira-race/virtual-robot-race.git (push)
```

---

### Step 2: Stash your local changes

If you have uncommitted changes (e.g., in `config.txt`), the merge will be blocked:

```
error: Your local changes to the following files would be overwritten by merge:
        config.txt
```

If you see this error, stash your changes first:

```bash
git stash
```

> **💡 What is `git stash`?** It temporarily shelves your uncommitted changes. After the merge, run `git stash pop` to restore them. **Your changes are not lost.**
>
> **Alternative**: You can also `git add` → `git commit` your changes first, then merge. Use `git stash` when you don't want to commit unfinished work yet.
>
> **Safety net**: VS Code has a **Timeline** feature that automatically saves local file history. Open a file, then check the Timeline panel at the bottom-left to browse past versions — a useful backup before running `git stash` or `git merge`.

---

### Step 3: Fetch and merge the latest official changes

```bash
git fetch upstream
git merge upstream/main
```

- `git fetch upstream` — Downloads the official changes locally (nothing in your code changes yet)
- `git merge upstream/main` — Integrates those changes into your branch

> **💡 Why not just `git pull upstream main`?** `pull` runs fetch and merge in one step. Keeping them separate lets you review what changed (`git log upstream/main`) before committing to the merge — a safer habit.
>
> **💡 Your code is safe**: `merge` combines the official changes with yours. A conflict only occurs if both you and the official repo edited the same line of the same file (see below).

Once the merge is complete, restore your stashed changes:

```bash
git stash pop
```

---

### Step 4: Push to your origin

Push the merged result to your fork on GitHub:

```bash
git push origin main
```

---

## Resolving Conflicts

If the official update changed the same part of a file you also edited, a conflict will occur.

> **💡 Want to start over?** Before resolving any conflicts, you can run `git merge --abort` to return to the state before the merge. Use this if things get confusing.

A conflicted file looks like this:

```
<<<<<<< HEAD
# Your change
=======
# Official change
>>>>>>> upstream/main
```

**How to resolve:**

1. Open the conflicting file in VSCode
2. Choose which version to keep (or combine both)
   - The easiest way is to click the **"Accept Current Change", "Accept Incoming Change",** or **"Accept Both Changes"** buttons that appear at the top of the conflict
   - Or manually delete the `<<<<<<`, `=======`, and `>>>>>>>` marker lines and save
3. Run `git add` → `git commit` to finalize

> **💡 Ask your AI assistant**: Paste the conflict directly and ask which version to keep:
> ```
> I have the following conflict.
> My goal is to keep my algorithm while taking in the official update.
> Please tell me which version to keep, or how to combine both.
>
> (paste the conflicting file content here)
> ```
> This prevents accidentally leaving in the `<<<<<<` markers or deleting the wrong lines.

---

## Summary

| Action | Command | When |
|--------|---------|------|
| Register upstream | `git remote add upstream <URL>` | First time only |
| Stash local changes | `git stash` | Before merge (if you have local edits) |
| Fetch official updates | `git fetch upstream` | When updates are available |
| Merge into your branch | `git merge upstream/main` | When updates are available |
| Restore stashed changes | `git stash pop` | After merge |
| Push to your fork | `git push origin main` | After merge |
| Abort a merge | `git merge --abort` | Before resolving conflicts |

aira version updates are announced on [GitHub Releases](https://github.com/aira-race/virtual-robot-race/releases) and [X (@RaceYourAlgo)](https://x.com/RaceYourAlgo).

---

> **❓ Having trouble?**
> Paste your error message directly into [NotebookLM](https://notebooklm.google.com/notebook/e16cd6bf-f642-459f-91fd-1146aa75398e) and ask for help.

---

⬅️ [Previous lesson: 07_How_to_Join_Race.md (How to Join the Race)](07_How_to_Join_Race.md) ｜ [Glossary](99_Glossary.md)
