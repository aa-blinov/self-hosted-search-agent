import unittest

from search_agent.evaluation import _answer_source_urls


class EvaluationLayerTests(unittest.TestCase):
    def test_answer_source_urls_preserves_http_scheme(self) -> None:
        answer = (
            "Sources\n"
            "[1] HyperPhysics - http://hyperphysics.phy-astr.gsu.edu/hbase/Kinetic/vappre.html\n"
            "[2] Python - https://www.python.org/downloads/release/python-3130/\n"
        )

        urls = _answer_source_urls(answer)

        self.assertEqual(
            urls,
            [
                "http://hyperphysics.phy-astr.gsu.edu/hbase/Kinetic/vappre.html",
                "https://www.python.org/downloads/release/python-3130/",
            ],
        )


if __name__ == "__main__":
    unittest.main()
