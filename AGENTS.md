# lsface-raspberry-pi Agent Instructions

## Base Branch Invariant (CRITICAL)

- **Default & Active Branch**: The active development branch for this repository is **`stage-1-demo`**, NOT `master`.
- **Branching Rule**: All agent tasks, worktrees, feature branches, and bug fixes **MUST** be created from or rebased onto `stage-1-demo`.
- **Targeting Rule**: Pull requests and merges must target `stage-1-demo`.
- **Legacy `master`**: `master` is a legacy branch (last updated Sept 10, 2026). Do NOT branch from or target `master` unless explicitly instructed by the repository owner.
- `origin/HEAD` points to `refs/remotes/origin/stage-1-demo`.

## Git LFS Requirement

- This repository uses Git LFS for binary models, databases, and assets (e.g. `db/lasalledb_lbph.yml`).
- Git LFS binary is installed at `/home/kyle/.local/bin/git-lfs`.
- Ensure `/home/kyle/.local/bin` is in `$PATH` when executing shell commands.
- Never commit large binaries directly without Git LFS tracking.

## Architecture and Layout

- **Platform**: Raspberry Pi face recognition application using OpenCV LBPH + Head Pose detection + Kivy UI.
- **`app/`**: Main Kivy application entry point (`app/main.py`), views (`app/src/views/`), engine (`app/src/engine/`), and UI definitions (`app/src/ui/`).
- **`app/src/engine/camera/manager.py`**: Centralized camera manager introduced in `stage-1-demo` to fix loading delays and prevent resource conflicts between live recognition and face enrollment. Always use this manager rather than creating independent camera capture instances.
- **`docs/`**: Documentation directory containing `IMPLEMENTATION.md` and `README.md`.
- **`config/`**: Configuration YAML files for camera, cascade, and LBPH parameters.
- **`pose-detection/`**: Head pose estimation scripts and configs.
- **`db/`**: LBPH training databases and metadata.

## Agent Coordination & Shared State

- Use `hub-coord` (`/home/kyle/.local/bin/hub-coord`) for project radar, file claims, and persistent memories:
  - Check active claims: `hub-coord --project lsface-raspberry-pi radar`
  - Claim files before editing: `hub-coord claim <files...>`
  - Release claims upon completion: `hub-coord release`
- Do not perform destructive git operations (`git reset --hard`, `git clean -fd`, `git checkout --`) in shared trees that may hold work from sibling agents.
