import unittest

from search_agent.eval.replay_eval import evaluate_replay_dataset, load_replay_cases


class ReplayEvalTests(unittest.TestCase):
    def test_load_replay_cases_reads_fixture_dataset(self):
        cases = load_replay_cases("eval_data/replay_smoke.json")

        self.assertEqual(len(cases), 4)
        self.assertEqual(cases[0].evaluation.case_id, "replay-ceo-microsoft")
        self.assertEqual(cases[-1].evaluation.case_id, "replay-asyncio-synthesis")

    def test_replay_eval_produces_stable_quality_metrics(self):
        summary = evaluate_replay_dataset("eval_data/replay_smoke.json")
        metrics = summary["metrics"]

        self.assertEqual(metrics["claim_support_rate"], 1.0)
        self.assertEqual(metrics["contradiction_detection_rate"], 1.0)
        self.assertEqual(metrics["insufficient_detection_rate"], 1.0)
        self.assertEqual(metrics["route_match_rate"], 1.0)
        self.assertEqual(metrics["primary_requirement_rate"], 1.0)
        self.assertEqual(metrics["source_requirement_rate"], 1.0)
        self.assertEqual(metrics["unsupported_statement_rate"], 0.0)
        self.assertEqual(metrics["answer_depth_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
