import unittest
from unittest.mock import patch

import search_agent.infrastructure.llm as llm


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


if __name__ == "__main__":
    unittest.main()
