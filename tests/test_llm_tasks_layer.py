import unittest
from unittest.mock import MagicMock, patch

import search_agent.infrastructure.llm as llm
from search_agent.infrastructure import llm_tasks


class LlmTasksLayerTests(unittest.TestCase):
    def test_answer_with_sources_uses_task_runner(self):
        runner = type("Runner", (), {"answer_with_sources": lambda self, query, sources, today: "Grounded answer"})()
        with patch("search_agent.infrastructure.llm.build_task_runner", return_value=runner):
            answer = llm.answer_with_sources(
                "Who is the CEO of Microsoft?",
                [{"title": "Satya", "url": "https://example.com", "snippet": "Satya Nadella"}],
            )
        self.assertEqual(answer, "Grounded answer")

    def test_analyze_rag_papers_uses_task_runner(self):
        runner = type("Runner", (), {"analyze_rag_papers": lambda self, papers: "Research summary"})()
        with patch("search_agent.infrastructure.llm.build_task_runner", return_value=runner):
            answer = llm.analyze_rag_papers(
                [{"title": "Paper", "url": "https://example.com", "abstract": "Abstract", "authors": ["A"]}]
            )
        self.assertEqual(answer, "Research summary")

    def test_build_context_block_splits_char_budget_across_sources(self) -> None:
        with patch.object(llm_tasks, "get_settings") as get_s:
            s = MagicMock()
            s.resolved_extract_max_chars.return_value = 4000
            get_s.return_value = s
            block = llm_tasks._build_context_block(
                "q",
                [
                    {"title": "a", "url": "http://a", "snippet": "A" * 10000},
                    {"title": "b", "url": "http://b", "snippet": "B" * 10000},
                ],
            )
        self.assertLessEqual(block.count("A"), 2100)
        self.assertLessEqual(block.count("B"), 2100)


if __name__ == "__main__":
    unittest.main()
