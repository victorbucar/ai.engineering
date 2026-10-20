# 03 — Structured outputs

## Contents

- `code/main.py`: schema validation and generation demos plus live Claude contact extraction.
- `code/structured.py`: Anthropic client and structured response schema.
- `code/main.ts`: standalone offline TypeScript implementation with simulated extraction.
- `docs/en.md`: original lesson explanation and exercises.
- `outputs/`: reusable prompt and decision guide.
- `quiz.json`: self-assessment questions.
- `linkedin-article.md`: locally authored learning article.

## Python setup (PowerShell)

From the repository root, with Python 3.10 or newer installed:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r lessons/03-structured-outputs/requirements.txt
$env:ANTHROPIC_API_KEY = 'your-api-key'
.\.venv\Scripts\python.exe lessons/03-structured-outputs/code/main.py
```

The key is read from the environment by the Anthropic SDK. Do not save credentials in tracked files. Dependencies are not locked yet.

## Offline TypeScript demo

With Node.js and npm installed, from the repository root:

```powershell
npx --yes tsx lessons/03-structured-outputs/code/main.ts
```

This command can download `tsx`; the demo itself uses simulated output and needs no API key.

## Starting points for further learning

The imported code is preserved as it was in the local source project:

- `claude_call(message)` currently sends the fixed `message1` example instead of its argument. Update this when experimenting with different inputs.
- The configured model is `claude-haiku-5-5`; confirm availability in your Anthropic account before running live calls.
- The handwritten validator implements only part of JSON Schema. It does not fully validate the Pydantic schema's nullable/date-time fields.
- Python and TypeScript currently demonstrate different extraction subjects: contacts and products respectively.

Record experiments and findings in `notes.md` as this lesson evolves.
