# Screenshots

There are no screenshots in this folder, and there is no image anywhere in the
repository.

The interface of this project is the text it writes to a terminal, and a
picture of a terminal is a picture of whatever was on the screen when it was
taken. A capture would carry the API keys of a session, the URLs of the pages
the agent read, and the questions of whoever used it.

## What stands in for a picture

The output is written down instead, in [../../DESIGN.md](../../DESIGN.md):

- The rule with the question on it.
- The profile line, with the active search backend.
- The status line while the agent works.
- The block of claims, with the verdict and the number of independent
  sources.
- The answer panel, and the receipt line under it.

The transcript of a run is also written to disk when `AGENT_RECEIPTS_DIR` is
set: the audit trail of the run, as JSON. That file records what the agent
did, and it is the one a reader should be given instead of a picture of a
screen.

## Adding one

A capture of the terminal belongs in this folder and is referenced by relative
path, the way the README does:

```markdown
![A run](docs/screenshots/run.png)
```

The checker resolves every image reference, so a file that is renamed without
its reference being updated fails the build. Redact the keys and the URLs
before a capture goes into the repository.
