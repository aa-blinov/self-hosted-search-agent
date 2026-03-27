import unittest

from search_agent.infrastructure.scholarly_sources import (
    parse_arxiv_id_from_url,
    parse_doi_from_url,
    parse_github_repo_url,
    parse_semanticscholar_paper_id,
)


class ScholarlySourcesUrlTests(unittest.TestCase):
    def test_github_repo(self) -> None:
        self.assertEqual(
            parse_github_repo_url("https://github.com/microsoft/vscode"),
            ("microsoft", "vscode"),
        )
        self.assertEqual(
            parse_github_repo_url("https://github.com/microsoft/vscode/tree/main/src"),
            ("microsoft", "vscode"),
        )
        self.assertIsNone(parse_github_repo_url("https://github.com/topics/python"))

    def test_arxiv_id(self) -> None:
        self.assertEqual(
            parse_arxiv_id_from_url("https://arxiv.org/abs/2312.12345"),
            "2312.12345",
        )
        self.assertEqual(
            parse_arxiv_id_from_url("https://arxiv.org/pdf/2312.12345.pdf"),
            "2312.12345",
        )

    def test_doi(self) -> None:
        self.assertEqual(
            parse_doi_from_url("https://doi.org/10.1000/182"),
            "10.1000/182",
        )

    def test_semantic_scholar_hex(self) -> None:
        u = (
            "https://www.semanticscholar.org/paper/"
            "Some-Title/649deb3e889f0d29cd0d7d698d7d7b7d7d7d7d7d"
        )
        self.assertEqual(
            parse_semanticscholar_paper_id(u),
            "649deb3e889f0d29cd0d7d698d7d7b7d7d7d7d7d",
        )

    def test_semantic_scholar_arxiv_and_doi_ids(self) -> None:
        self.assertEqual(
            parse_semanticscholar_paper_id("https://www.semanticscholar.org/paper/arxiv/2312.12345v2"),
            "ARXIV:2312.12345",
        )
        self.assertEqual(
            parse_semanticscholar_paper_id("https://www.semanticscholar.org/paper/doi/10.1000/182"),
            "DOI:10.1000/182",
        )


if __name__ == "__main__":
    unittest.main()
