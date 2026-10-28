## Coding Philosophy
- Before writing any solution, propose 2-3 approaches ranked by simplicity
- Wait for approval before implementing
  - Exception: typos, renames, one-line fixes, or explicit instructions with only one obvious approach — state what you will do, then do it
- Prefer built-ins and stdlib over custom logic
- No classes unless state genuinely needs to be managed
  - Exception: `dataclass`, `NamedTuple`, and Pydantic models are encouraged for structured records even without mutable state — prefer them over passing loose dicts
  - Still no classes just to group functions, and no inheritance hierarchies without justification
- Complexity must be explicitly justified
- Apply YAGNI: don't build for hypothetical future requirements
  - YAGNI applies to features, not structure — a small investment in clear function/module boundaries for current needs is expected
- Prefer flat over nested (Zen of Python)

## Code Review Habit
- After any implementation, self-critique for unnecessary complexity
- Ask: "What could be removed without losing functionality?"

## General Rules
- No scaffolding, boilerplate, or placeholder code unless explicitly asked
- If a built-in or 5-line solution exists, use it before reaching for a library
  - Preference order: stdlib → library already in requirements.txt → new well-maintained library → hand-rolled
  - Never hand-roll parsing, dates/time zones, HTTP retries, security, or financial math — use stdlib or a tested library
- One thing at a time — do it well before expanding scope

## Comments & Documentation
- Comments are designed for explanation to a user with only basic knowledge of Python and no knowledge of Yoda (a deliberate choice: comment density is higher than typical industry style)
- Add a `#` comment above every logical code block explaining what it does and why, in plain English
- Explain intent, non-obvious logic, and domain terms (e.g. "whisper number", "consensus EPS") — don't restate a single line of code (no `# loop over the list`)
- Every function gets a docstring (PEP 257): what it does, its inputs, what it returns
- When code changes, update its comments in the same edit — a stale comment is worse than none

## Testing
- Every bug fix gets a test that reproduces the bug first, then passes with the fix
- New logic that computes or transforms data gets at least one test
- Tests live in `tests/` and run with `pytest`; run them before calling work done and report failures honestly
- Tests must not call live APIs — use saved fixtures or offline data

## Error Handling & Data Integrity
- Fail loudly: never use bare `except:` or `except Exception: pass`; catch specific exceptions, and only where you can handle them
- Validate external data (APIs, filings, user input) at the point it enters the program
- Never silently substitute a plausible-looking value for missing or failed financial data; if a fallback is used, label it as a fallback in the output
- Missing data is shown as missing, not as zero

## Style & Naming
- Follow PEP 8
- Add type hints to function signatures (PEP 484)
- Use descriptive names that make "what" comments unnecessary (`eps_estimate`, not `x`); avoid unexplained abbreviations
- Don't add new tooling (linters, formatters, config files) unless asked

## Secrets & Reproducibility
- API keys and secrets come from environment variables (`.env`), never hard-coded or committed
- When code needs a new key, add its *name* with an empty value to `.env.example` (e.g. `FMP_API_KEY=`); real values go only in `.env`, which is gitignored
- Pin every dependency to an exact version in `requirements.txt`, with a comment explaining any non-obvious pin

## Commit Messages
- Commit messages should state what was done in a commit (e.g. ".gitignored smoke test outputs")
- Commit messages SHOULD NOT include or be based on step names 
  - Bad Example: "recorded the step 8b smoke test (BA with the earnings call)"
  - Good Example: "smoke test for BA, including new earnings call features"