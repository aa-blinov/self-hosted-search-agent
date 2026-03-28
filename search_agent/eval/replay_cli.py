from __future__ import annotations

import argparse
import json
from pathlib import Path

from search_agent.eval.replay_eval import evaluate_replay_dataset
from search_agent.eval.tracking import DEFAULT_EVAL_RUNS_DIR, save_eval_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic replay evaluation from frozen fixtures.")
    parser.add_argument("dataset", help="Replay dataset path (.json or .jsonl)")
    parser.add_argument(
        "--eval-out",
        default=DEFAULT_EVAL_RUNS_DIR,
        help="Directory or JSON artifact path for saved results",
    )
    parser.add_argument(
        "--eval-label",
        default=None,
        help="Optional label stored in run metadata",
    )
    parser.add_argument(
        "--eval-no-save",
        action="store_true",
        help="Do not write the JSON artifact",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full summary JSON",
    )
    args = parser.parse_args()

    summary = evaluate_replay_dataset(args.dataset, log=print)

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))

    if not args.eval_no_save:
        artifact = save_eval_run(summary, Path(args.eval_out), label=args.eval_label)
        print(f"Replay eval saved: {artifact}")


if __name__ == "__main__":
    main()
