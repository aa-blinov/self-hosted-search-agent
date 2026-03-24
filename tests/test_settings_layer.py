import unittest

from search_agent.settings import AppSettings


class SettingsLayerTests(unittest.TestCase):
    def test_resolved_send_to_logfire_parses_common_values(self):
        self.assertEqual(AppSettings(logfire_send_to_logfire="if-token-present").resolved_send_to_logfire(), "if-token-present")
        self.assertTrue(AppSettings(logfire_send_to_logfire="true").resolved_send_to_logfire())
        self.assertFalse(AppSettings(logfire_send_to_logfire="false").resolved_send_to_logfire())


if __name__ == "__main__":
    unittest.main()
