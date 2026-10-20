---
name: self-hosted-search-agent
description: A command line tool with no interface of its own. The surface is the terminal: a rule, a panel, a list of claims and a prompt.
colors:
  source: terminal-default
  accent: "#ffcc00"
  bracket: "#888888"
  answer-panel: "green"
  research-panel: "magenta"
  metadata: "dim"
typography:
  source: terminal-default
  body:
    fontFamily: "the terminal's own face"
    fontSize: "the terminal's own size"
    fontWeight: 400
    lineHeight: 1.2
rounded:
  source: rich-default
  panel: "rounded"
spacing:
  source: rich-default
  sm: "1 blank line"
---

# Design system: self-hosted-search-agent

This project has no visual identity of its own. It is a command line tool:
the whole interface is what it writes to a terminal, and every pixel of it
belongs to the terminal and to Rich, the library that draws it.

What follows records the small vocabulary the code does set, because that is
the whole of it: two hex values, a handful of Rich style names, and the order
in which the output appears. Product truth: [PRODUCT.md](PRODUCT.md). The
machine behind it: [the architecture](docs/architecture.md).

## Overview

**Creative North Star: "The Desk"**

The output is a report laid on the desk of the person who asked the question.
The question goes on a rule at the top. The line under it says which profile
ran and which search backend answered. Then the claims, one per line, with
their verdict and their number of independent sources. Then the answer, in a
panel, because it is the one thing that is long. Then the receipt, in the
quietest ink there is, because it is a path and not a message.

The one thing the tool is careful about is the difference between what the
model said and what the tool knows. The claims carry a verdict, the panel
carries the answer, and the receipt points at the audit trail. The reader can
tell which of the three is a fact of the run.

The confirmed rejections:

- No colour for decoration. Every colour in the output carries a meaning:
  accent for the prompt, green for the answer, magenta for the research
  mode, dim for metadata, red for a failure.
- No box around anything but the answer, the help and the settings.
- No progress bar. The work is announced with a status line and then it is
  over.

**Key characteristics:**

- English in the interface, including the headings of the panels.
- One typeface: the terminal's own.
- The answer is the only long text, and it is the only thing in a box.
- A citation in the answer is a number in square brackets, and the number is
  positional.

## Colors

Rich resolves the names below against the terminal's own palette, so the
meaning is stable and the exact shade is the terminal's business. Two values
are set as hex, in `search_agent/cli.py`, because they are the prompt and the
prompt is the one element a person looks at for hours.

- **Accent** (`#ffcc00`): the arrow of the prompt, `search] > `. It is bold.
- **Bracket** (`#888888`): the square brackets around the word `search` in
  the prompt.
- **Green**: the panel of the answer, and the title `Answer` in bold green.
- **Magenta**: the research mode, the rule `arXiv Pipeline Research` and the
  panel of the techniques.
- **Cyan**: the profile name, the identifiers of the claims, the command names
  in the help panel, and the values in the settings panel.
- **Dim**: the metadata. The profile line, the description of a profile, the
  verdict and the source count of a claim, the receipt path, the hint under
  the commands.
- **Bold**: the headings. `Query:`, `Claims (1):`, the title of a panel, the
  number of a profile in the picker.
- **Yellow**: a warning from the picker, such as an unknown number.
- **Red**: a failure, such as the missing `LLM_API_KEY` or a model call that
  did not come back.

### Named rules

**The Meaning Rule.** A colour says what kind of thing this is. Green is the
answer, cyan is a value the tool read, dim is context, and red is a failure.
A colour that says nothing is not used.

**The One Panel Rule.** The answer, the help and the settings are in a panel.
Everything else is a line.

## Typography

The terminal's own face, at the terminal's own size. The tool sets no
typeface, no size and no weight beyond the `bold` and `dim` styles of Rich,
and it does not try to detect a terminal that cannot show them.

- **Body**: the terminal's face. The text of the answer and of the metadata.
- **Bold**: a heading, and a value the reader is meant to look at.
- **Dim**: everything that is context rather than content.

Two rules of the markup matter for a change:

- **The answer is printed as `Text`, not as markup.** Rich would read the
  `[1]` of a citation as a style tag, so the answer goes through
  `Text(report.answer)`. A change that prints the answer as a string breaks
  every citation on the screen.
- **The markdown of the research analysis is printed as `Markdown`.** The
  model writes markdown there, and Rich renders it.

## Layout

One column, and the output follows the order of the run.

```
  Query: When was Python 3.13 released?          <- a rule, the question in bold

  Profile: web - General search · active search: brave
  Searching (claim-level agent)...                <- a status line, cyan

  Claims (1):                                     <- bold cyan
  - unified-1
    verdict=supported | independent_sources=2     <- dim

  ┌─ Answer ─────────────────────────────────┐    <- a green panel
  │ Python 3.13 was released on ... [1]       │
  └──────────────────────────────────────────┘

  Receipt: /path/to/receipt.json                <- dim
```

The rule under the question separates what was asked from what was done. The
profile line sits under it, because the profile decides what the answer is
worth and the reader should see it before the answer. The claims come before
the panel, because a verdict is a fact and the answer is a claim.

The interactive prompt adds one line at the bottom:

```
[search] >
```

## Elevation & Depth

There is none. A terminal has no shadow, and the tool does not draw one or
simulate one. A panel is a box, and that is all the depth there is: the
answer is boxed because it is longer than the rest, not because it is more
important than the receipt under it.

## Shapes

The only shapes are the ones Rich draws: the box of a panel and the rule of a
`Rule`. Both are drawn with the terminal's own line characters, so they follow
the terminal's idea of what a line looks like.

There is no radius anywhere, because a terminal does not have one. A panel's
corners are the corners of the characters the terminal draws.

## Components

### The query rule

- One horizontal rule across the width of the terminal, with `Query:` in bold
  and the question after it.
- The interactive prompt draws its own rule with the question alone.

### The profile line

- `Profile:` in dim, the name in cyan, a dash, the description, then the
  active search backend in dim and cyan.
- One line. It answers the question of what the run really did.

### The status line

- A cyan spinner with a short phrase: `Searching (claim-level agent)...`.
- The arXiv mode uses magenta and its own phrase.

### The claims block

- A heading in bold cyan: `Claims (1):`. The number is always 1, because the
  runner keeps a single synthetic claim.
- Under it, one line per claim: the claim text, and under that a dim line
  with the verdict and the number of independent sources.
- The interactive prompt prints the id of the claim in cyan before the text.

### The answer panel

- A green border and a title in bold green.
- The body is `Text`, so the `[N]` of the citations survive.
- Nothing else goes inside it.

### The commands panel

- A border in the default style and the title `Commands` in bold.
- Six lines, each a command in bold cyan and its sentence in the default ink.
- The last two lines are dim: type any question, then the keys that work.

### The settings panel

- The title `Settings`, and one line per setting: the label in the default
  ink, the value in cyan, the two aligned by spaces.
- The lines are `Model`, `Provider`, `Search`, `Extractor`, `Extract limit`
  and `LLM task caps`.

### The receipt

- A dim line with the word `Receipt` and the path of the audit trail.
- It is the last thing on the screen and the quietest, because it is a path.

## Do's and Don'ts

### Do:

- **Do** print the answer through `Text(...)`, so that a citation is not read
  as a style tag.
- **Do** keep the order of the output: question, profile, claims, answer,
  receipt.
- **Do** use a colour for a kind of thing, and the same colour for that kind
  of thing everywhere.
- **Do** write the interface in English, including the title of a panel. The
  answer footer is the one exception, and it is fixed in the code.

### Don't:

- **Don't** put a box around anything but the answer, the help and the
  settings.
- **Don't** print a progress bar or a percentage. The work is announced and
  then it is over.
- **Don't** set a colour that says nothing, and don't set a colour outside
  the list above.
- **Don't** reorder the claims and the answer. The verdict comes first.
- **Don't** print the receipt as anything but a dim line.
