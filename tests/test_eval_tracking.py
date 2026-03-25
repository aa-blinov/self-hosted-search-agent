import tempfile
import unittest
from pathlib import Path

from search_agent.eval.tracking import compare_metric_deltas, load_eval_run, merge_run_metadata, save_eval_run


class EvalTrackingTests(unittest.TestCase):
    def test_save_and_load_roundtrip(self) -> None:
        summary = {
            "dataset_path": "eval_data/sample_cases.jsonl",
            "case_count": 2,
            "metrics": {"claim_support_rate": 0.5},
            "by_split": {},
            "cases": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = save_eval_run(summary, Path(tmp) / "x.json", label="ci")
            self.assertTrue(path.exists())
            loaded = load_eval_run(path)
            self.assertEqual(loaded["case_count"], 2)
            self.assertEqual(loaded["metrics"]["claim_support_rate"], 0.5)
            self.assertEqual(loaded["run_metadata"]["label"], "ci")
            self.assertEqual(loaded["run_metadata"]["artifact_filename"], "x.json")
            self.assertIn("artifact_path", loaded["run_metadata"])
            self.assertEqual(loaded["run_metadata"]["eval_runs_dir"], "eval_runs")

    def test_merge_run_metadata_includes_eval_runs_dir(self) -> None:
        merged = merge_run_metadata({"dataset_path": "eval_data/x.jsonl", "case_count": 1})
        self.assertEqual(merged["run_metadata"]["eval_runs_dir"], "eval_runs")

    def test_compare_metric_deltas(self) -> None:
        a = {"metrics": {"claim_support_rate": 0.4, "median_answer_latency": 100.0}}
        b = {"metrics": {"claim_support_rate": 0.6, "median_answer_latency": 90.0}}
        d = compare_metric_deltas(a, b)
        self.assertAlmostEqual(d["claim_support_rate"]["delta"], 0.2)
        self.assertAlmostEqual(d["median_answer_latency"]["delta"], -10.0)


if __name__ == "__main__":
    unittest.main()
