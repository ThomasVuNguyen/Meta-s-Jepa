---
description: Default workflow for running scripts and commands in this project
---

// turbo-all

## Rules
1. Always write scripts inside the project folder `/home/thomas/development/experiments/Meta-s-Jepa/`
2. Never write scripts to `/tmp/` or other external locations
3. All `run_command` calls should use `SafeToAutoRun: true`
4. SSH commands, pip installs, python script execution — all auto-run
5. Pod creation/termination — auto-run (user trusts the process)
