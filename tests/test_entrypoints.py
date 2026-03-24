import importlib
import unittest
from unittest.mock import patch

import search_agent.application.agent_steps as agent_steps


class EntryPointTests(unittest.TestCase):
    def test_main_module_imports_cleanly(self):
        module = importlib.import_module("main")
        self.assertTrue(callable(module.main))

    def test_run_search_agent_delegates_to_use_case(self):
        mock_report = object()
        mock_use_case = type("UseCase", (), {"run": lambda self, *args, **kwargs: mock_report})()

        with patch("search_agent.build_search_agent_use_case", return_value=mock_use_case) as builder:
            report = agent_steps.run_search_agent("Who is the CEO of Microsoft?", profile=object(), log=print)

        self.assertIs(report, mock_report)
        builder.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
