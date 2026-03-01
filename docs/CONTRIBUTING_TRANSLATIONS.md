# Contributing Translations

Thank you for your interest in making aira accessible to learners around the world!

The training materials in this repository were originally written in Japanese (`lessons_JP/`) and translated into English (`lessons_EN/`).
If you would like to contribute a translation in your language, **we warmly welcome your Pull Request.**
Every translation helps grow the aira community globally.

---

## How to Contribute a Translation

### 1. Create your lessons folder

Create a new folder under `docs/` named `lessons_XX`, where `XX` is the [ISO 639-1 two-letter language code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) for your language.

Examples:

| Language | Folder name |
|----------|-------------|
| French | `docs/lessons_FR/` |
| Spanish | `docs/lessons_ES/` |
| German | `docs/lessons_DE/` |
| Korean | `docs/lessons_KO/` |
| Chinese (Simplified) | `docs/lessons_ZH/` |
| Portuguese | `docs/lessons_PT/` |
| Arabic | `docs/lessons_AR/` |

### 2. Translate the lesson files

Use the English versions in [`docs/lessons_EN/`](lessons_EN/) as your source.
Translate any or all of the following files:

| File | Contents |
|------|----------|
| `00_Preparation.md` | Environment setup |
| `01_Foundation.md` | Core philosophy and overview |
| `02_Live_QA_NotebookLM.md` | How to use the AI Q&A system |
| `03_Manual_Control.md` | Keyboard driving |
| `04_Log_and_Table_Mode.md` | Log data and table-mode driving |
| `05_Rule_Based_Control.md` | Rule-based algorithm development |
| `06_AI_Mode.md` | Imitation learning and neural networks |
| `07_How_to_Join_Race.md` | How to enter an official race |
| `99_Glossary.md` | Glossary of terms |

You do not have to translate all files at once — partial contributions are welcome too.

### 3. Add a README in your folder (optional but appreciated)

If you like, add a short `README.md` inside your `lessons_XX/` folder with a note like:

```md
# aira Training Materials — [Your Language]

Translated by [your name / GitHub handle].
Based on lessons_EN (English).
```

### 4. Open a Pull Request

1. Fork the repository and create a branch (e.g., `translation/lessons-FR`).
2. Add your `docs/lessons_XX/` folder with the translated files.
3. Open a Pull Request against the `main` branch.
4. In the PR description, briefly describe the language and which files you translated.

We will review and merge your contribution as quickly as possible.

---

## Guidelines

- **Use `lessons_EN/` as the source**, not `lessons_JP/`, so all translations stay consistent with each other.
- Keep code blocks, file names, parameter names, and URLs **unchanged** — only translate the surrounding text.
- It is fine to add culturally relevant notes or adjust examples to better fit your audience, as long as the technical content remains accurate.
- If you find errors or outdated content in the English source while translating, please open a separate issue or PR for that fix.

---

## Current Translations

| Language | Folder | Status |
|----------|--------|--------|
| Japanese (original) | [`lessons_JP/`](lessons_JP/) | Complete |
| English | [`lessons_EN/`](lessons_EN/) | Complete |
| *Your language here* | `lessons_XX/` | *You could be next!* |

---

## Questions?

If you have any questions or need help, feel free to open a [GitHub Issue](https://github.com/aira-race/virtual-robot-race/issues).

Thank you for helping the aira community grow!
**Together, let's bring autonomous racing education to everyone.**
