"""Offline contract and evidence tests; no model or network calls."""
import unittest
from unittest.mock import patch
from pathlib import Path

import hw4
import ablation


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = Path(__file__).parent / "public_test"
        cls.corpus = hw4.load_corpus(folder / "corpus")
        cls.questions = hw4.load_json(folder / "questions.json")
        cls.gold = hw4.load_json(folder / "ground_truth.json")
        cls.metadata = hw4.load_json(folder / "docs.json")
        cls.tables = hw4.load_json(folder / "tables.json")
        cls.by_file = {name: hw4.chunk_document(name, html) for name, html in cls.corpus.items()}
        cls.chunks = [chunk for chunks in cls.by_file.values() for chunk in chunks]
        cls.index = hw4.build_index(cls.chunks)

    def test_tables_and_document_contracts(self):
        for table in self.tables:
            with self.subTest(table=table["caption"]):
                self.assertEqual(hw4.table_score(hw4.table_to_text(table["html"]), table), 1)
        for name, chunks in self.by_file.items():
            with self.subTest(document=name):
                self.assertEqual(hw4.document_problems(chunks, None, self.metadata[name], name), [])

    def test_public_queries_and_evidence(self):
        identities = {id(chunk) for chunk in self.chunks}
        for qid, question in self.questions.items():
            with self.subTest(question=qid):
                self.assertEqual(hw4.parse_score(hw4.parse_query(question), self.gold[qid]), 1)
                retrieved = hw4.retrieve(self.index, question)
                self.assertLessEqual(len(retrieved), 5)
                self.assertTrue(all(id(chunk) in identities for chunk in retrieved))
                if self.gold[qid]["evidence"]:
                    self.assertTrue(hw4.retrieval_hit(retrieved, self.gold[qid]))
                else:
                    self.assertEqual(retrieved, [])

    def test_synthetic_company_prior_year_and_unknown_suffix(self):
        html = '''<html><body><h1>Cedar Circuit Ltd.</h1>
        <p class="subtitle">Annual Report 2031</p><h2>Financial statements</h2>
        <table><caption>Income statement (USD millions)</caption>
        <tr><th>Item</th><th>2031</th><th>2030</th></tr>
        <tr><td>Revenue</td><td>7,321.5</td><td>6,100.2</td></tr></table>
        </body></html>'''
        chunks = hw4.chunk_document("unseen.html", html)
        index = hw4.build_index(chunks)
        result = hw4.retrieve(index, "What was Cedar Circuit's revenue in FY2030, in USD millions?")
        self.assertTrue(any("6,100.2" in chunk.page_content for chunk in result))
        self.assertEqual(hw4.retrieve(index, "What was Cedar Circuit Systems' revenue in FY2030?"), [])
        self.assertEqual(hw4.retrieve(index, "What was Cedar Circuit's revenue in FY2029?"), [])
        self.assertEqual(hw4.retrieve(index, "What were Cedar Circuit's advertising expenses in FY2031?"), [])

    def test_table_punctuation_and_long_paragraph(self):
        table = '''<table><caption>USD millions</caption><tr><th>Item</th><th>2032</th></tr>
        <tr><td>Expense</td><td>(1,234.5)</td></tr></table>'''
        self.assertIn("Expense | 2032: (1,234.5)", hw4.table_to_text(table))
        html = '<h1>Elm Ltd.</h1><p class="subtitle">Annual Report 2032</p><p>' + 'word ' * 1000 + '</p>'
        chunks = hw4.chunk_document("long.html", html)
        self.assertTrue(all(len(chunk.page_content) <= 1500 for chunk in chunks))
        self.assertEqual(sum(chunk.page_content.count("word") for chunk in chunks), 1000)

    def test_empty_index(self):
        self.assertEqual(hw4.retrieve(hw4.build_index([]), "What is revenue?"), [])

    def test_ablation_isolation(self):
        previous = ablation.configuration(0)
        for variant in range(1, len(ablation.VARIANTS)):
            current = ablation.configuration(variant)
            self.assertEqual(sum(previous[key] != current[key] for key in current), 1)
            previous = current
        without_headers = ablation.make_chunks(self.corpus, 3)
        with_headers = ablation.make_chunks(self.corpus, 4)
        self.assertEqual(len(without_headers), len(with_headers))
        for plain, contextual in zip(without_headers, with_headers):
            self.assertEqual(plain.metadata, contextual.metadata)
            self.assertEqual(plain.page_content, contextual.page_content.split("\n", 1)[1])
        with patch.object(hw4, "parse_query", side_effect=AssertionError("Baseline parsed a query")):
            baseline = ablation.make_chunks(self.corpus, 0)
            tokenizer = ablation.tokenizer_for(0)
            tokenized = [tokenizer(chunk.page_content) for chunk in baseline]
            index = {"chunks": baseline, "bm25": ablation.BM25Okapi(tokenized),
                     "tokens": [set(words) for words in tokenized]}
            result = hw4.retrieve_config(index, "What is revenue?", filtering=False,
                balanced=False, abstention=False, tokenizer=tokenizer, preprocess=False,
                coverage_weight=0, table_weight=0, year_weight=0)
            self.assertEqual(len(result), 5)


if __name__ == "__main__":
    unittest.main()
