# Plan - Track: Quality Sweep (src/)

## Phase 1: Tool Validation & Baseline
Establish that all required quality tools are available and record the initial state.

- [x] Task: Verify tool availability (black, isort, ruff, mypy, pytest)
- [ ] Task: Run baseline tests to ensure current state is functional
- [ ] Task: Conductor - User Manual Verification 'Tool Validation & Baseline' (Protocol in workflow.md)

## Phase 2: Automated Formatting & Linting
Apply non-destructive automated fixes to the `src/` directory.

- [ ] Task: Execute `isort src/` to standardize import ordering
- [ ] Task: Execute `black src/` to enforce PEP 8 style consistency
- [ ] Task: Execute `ruff check src/ --fix` to resolve automated linting violations
- [ ] Task: Conductor - User Manual Verification 'Automated Formatting & Linting' (Protocol in workflow.md)

## Phase 3: Static Type Analysis & Manual Fixes
Identify and resolve type-related issues that require human intervention.

- [ ] Task: Run `mypy src/` and analyze the error report
- [ ] Task: Resolve high-priority type errors in core modules
- [ ] Task: Re-run `mypy src/` to verify zero-error state
- [ ] Task: Conductor - User Manual Verification 'Static Type Analysis & Manual Fixes' (Protocol in workflow.md)

## Phase 4: Regression Testing & Final Approval
Ensure that the quality sweep has not introduced any functional regressions.

- [ ] Task: Run `pytest` on all relevant tests in `tests/`
- [ ] Task: Verify that code coverage for `src/` remains >80%
- [ ] Task: Conductor - User Manual Verification 'Regression Testing & Final Approval' (Protocol in workflow.md)
