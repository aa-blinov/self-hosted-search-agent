import unittest

from search_agent.eval.paired_control_cli import build_paired_control_report, find_case_differences


class PairedControlCliTests(unittest.TestCase):
    def test_find_case_differences_detects_claim_and_answer_variance(self):
        replay = {
            "cases": [
                {
                    "case_id": "case-1",
                    "split": "factual",
                    "answer_chars": 120,
                    "unique_sources_in_answer": 2,
                    "backend_issue": False,
                    "claims": [
                        {
                            "match": "foo",
                            "actual_verdict": "supported",
                            "actual_route": "fast",
                            "has_primary_source": True,
                            "independent_source_count": 2,
                        }
                    ],
                }
            ]
        }
        live = {
            "cases": [
                {
                    "case_id": "case-1",
                    "split": "factual",
                    "answer_chars": 160,
                    "unique_sources_in_answer": 3,
                    "backend_issue": True,
                    "claims": [
                        {
                            "match": "foo",
                            "actual_verdict": "supported",
                            "actual_route": "fast",
                            "has_primary_source": True,
                            "independent_source_count": 2,
                        }
                    ],
                }
            ]
        }

        diffs = find_case_differences(replay, live)

        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["case_id"], "case-1")
        self.assertEqual(diffs[0]["replay_answer_chars"], 120)
        self.assertEqual(diffs[0]["live_answer_chars"], 160)
        self.assertTrue(diffs[0]["live_backend_issue"])

    def test_build_paired_control_report_includes_selected_metric_deltas(self):
        replay = {
            "dataset_path": "eval_data/replay_control.jsonl",
            "metrics": {
                "claim_support_rate": 1.0,
                "route_match_rate": 1.0,
                "median_answer_latency": 5.0,
            },
            "cases": [],
        }
        live = {
            "dataset_path": "eval_data/control_dataset.jsonl",
            "metrics": {
                "claim_support_rate": 0.9,
                "route_match_rate": 0.8,
                "median_answer_latency": 20000.0,
            },
            "cases": [],
        }

        report = build_paired_control_report(replay, live)

        self.assertEqual(report["replay_dataset_path"], "eval_data/replay_control.jsonl")
        self.assertEqual(report["live_dataset_path"], "eval_data/control_dataset.jsonl")
        self.assertAlmostEqual(report["metric_deltas"]["claim_support_rate"]["delta"], -0.1)
        self.assertAlmostEqual(report["metric_deltas"]["route_match_rate"]["delta"], -0.2)
        self.assertAlmostEqual(report["metric_deltas"]["median_answer_latency"]["delta"], 19995.0)


if __name__ == "__main__":
    unittest.main()
