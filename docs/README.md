# Documentation

The documents for self-hosted-search-agent.

| Document | What it covers |
| --- | --- |
| [architecture.md](architecture.md) | How the agent is put together: the layers, the ports, the loop and the decisions. |
| [../README.md](../README.md) | The front door: what the agent does and how to run it. |
| [../AGENTS.md](../AGENTS.md) | The commands, the traps and the rails for a session that opens the repository. |
| [../PRODUCT.md](../PRODUCT.md) | The product truth the code is measured against. |
| [../DESIGN.md](../DESIGN.md) | The terminal output: the rule, the panel, the claims and the prompt. |

## Reading order

New here: [../README.md](../README.md), then [architecture.md](architecture.md)
for the shape of the loop and the ports between the layers.

Changing the loop: [architecture.md](architecture.md), then
[../AGENTS.md](../AGENTS.md) for the traps. The scoring weights live in
`search_agent/application/policy_tuning.py` and are tuned, so a change to one
of them needs a dataset run to justify it.

## What is not here

The schema of a dataset row and the meaning of every metric are in
`eval_data/README.md`, next to the datasets themselves. The comparison
between two saved runs is in `search_agent/eval/`.
