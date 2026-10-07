from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from financial_analyst.retrieval import LocalKeywordAdapter, MAX_PASSAGE_CHARACTERS, MAX_RESULTS, RetrievalError
from tests.support import _write_pdf


@unittest.skipUnless(importlib.util.find_spec("pdfplumber"), "PDF dependency unavailable")
class KeywordRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / "manifest.json"
        self.payload = {"documents": []}
        self.add_document("a", "alpha.pdf", [
            [(40, 760, "Example Company revenue revenue grew through portfolio execution.")],
            [(40, 760, "Vaccines supported adoption in 1QFY27.")],
        ])
        self.add_document("b", "beta.pdf", [
            [(40, 760, "Renewable Company builds wind turbines for energy customers.")],
        ])

    def add_document(self, document_id, filename, pages):
        path = self.root / filename
        _write_pdf(path, pages)
        self.payload["documents"].append({
            "document_id": document_id, "document_name": filename,
            "local_path": filename,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "url": f"https://example.com/reports/{document_id}",
        })

    def load(self):
        self.manifest.write_text(json.dumps(self.payload), encoding="utf-8")
        return LocalKeywordAdapter(self.manifest)

    def test_keyword_results_carry_exact_source_and_citation(self):
        adapter = self.load()
        hits = adapter.search("REVENUE, portfolio!")
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertGreater(hit.score, 0)
        passage = hit.passage
        self.assertEqual(passage.document_id, "a")
        self.assertEqual(passage.document_name, "alpha.pdf")
        self.assertEqual(passage.page, 1)
        self.assertEqual(passage.excerpt, "Example Company revenue revenue grew through portfolio execution.")
        self.assertEqual(passage.source_sha256, self.payload["documents"][0]["sha256"])
        self.assertEqual(passage.citation.ref, passage.ref)
        self.assertEqual(passage.citation.link, "https://example.com/reports/a")
        self.assertIn("characters 0:", passage.locator)
        self.assertEqual(adapter.mode, "local_keyword")

    def test_period_tokens_and_no_semantic_expansion(self):
        self.assertEqual(self.load().search("1qfy27")[0].passage.page, 2)
        self.assertEqual(self.load().search("sales"), ())

    def test_empty_outcome_and_blank_page_coverage(self):
        self.add_document("c", "blank.pdf", [[]])
        adapter = self.load()
        self.assertEqual(adapter.search("unsupportedterm"), ())
        coverage = adapter.coverage[-1]
        self.assertEqual(coverage.page_count, 1)
        self.assertEqual(coverage.pages_with_text, 0)
        self.assertEqual(coverage.passage_count, 0)
        self.assertEqual(sum(item.page_count for item in adapter.coverage), 4)

    def test_bounded_exact_passages_stay_within_pages(self):
        lines = [(40, 760 - index * 12, f"Line {index} contains recurring keyword and source text " * 3)
                 for index in range(30)]
        self.add_document("c", "long.pdf", [lines, [(40, 760, "Separate page keyword.")]])
        adapter = self.load()
        import pdfplumber
        with pdfplumber.open(self.root / "long.pdf") as pdf:
            page_text = tuple(page.extract_text() or "" for page in pdf.pages)
        passages = [item for item in adapter.passages if item.document_id == "c"]
        self.assertGreater(len(passages), 2)
        for passage in passages:
            self.assertLessEqual(len(passage.excerpt), MAX_PASSAGE_CHARACTERS)
            self.assertEqual(passage.excerpt, page_text[passage.page - 1][passage.start_offset:passage.end_offset])
            self.assertIn(f":page:{passage.page}:offset:{passage.start_offset}", passage.ref)
        first_page = [item for item in passages if item.page == 1]
        self.assertEqual("".join(item.excerpt for item in first_page), page_text[0])

    def test_stable_references_ranking_and_query_term_deduplication(self):
        first = self.load()
        second = self.load()
        self.assertEqual(first.search("company"), second.search("company"))
        self.assertEqual(first.search("company"), first.search("company company"))
        self.assertEqual(len(first.search("company", limit=1)), 1)
        self.assertEqual(len(first.search("company", limit=MAX_RESULTS)), 2)

    def test_bm25_prefers_more_matching_terms(self):
        adapter = self.load()
        hits = adapter.search("company revenue")
        self.assertEqual([hit.passage.document_id for hit in hits], ["a", "b"])
        self.assertGreater(hits[0].score, hits[1].score)

    def test_immutable_returned_values(self):
        adapter = self.load()
        with self.assertRaises(FrozenInstanceError):
            adapter.search("revenue")[0].score = 123
        with self.assertRaises(FrozenInstanceError):
            adapter.passages[0].excerpt = "Invented text"
        with self.assertRaises(FrozenInstanceError):
            adapter.coverage[0].page_count = 20

    def test_hash_mismatch_and_unavailable_sources_fail(self):
        self.payload["documents"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(RetrievalError, "SHA-256"):
            self.load()
        self.payload["documents"][0]["sha256"] = hashlib.sha256((self.root / "alpha.pdf").read_bytes()).hexdigest()
        (self.root / "alpha.pdf").unlink()
        with self.assertRaisesRegex(RetrievalError, "unavailable"):
            self.load()

    def test_post_load_file_changes_do_not_replace_authenticated_passages(self):
        adapter = self.load()
        _write_pdf(self.root / "alpha.pdf", [[(40, 760, "Invented replacement text.")]])
        self.assertEqual(adapter.search("replacement"), ())
        self.assertIn("portfolio execution", adapter.search("revenue")[0].passage.excerpt)

    def test_invalid_or_renamed_pdf_fail(self):
        (self.root / "alpha.pdf").write_bytes(b"not a PDF")
        self.payload["documents"][0]["sha256"] = hashlib.sha256(b"not a PDF").hexdigest()
        with self.assertRaisesRegex(RetrievalError, "extracted"):
            self.load()
        self.payload["documents"][0]["document_name"] = "renamed.pdf"
        with self.assertRaisesRegex(RetrievalError, "filename"):
            self.load()

    def test_malformed_manifests_fail(self):
        for payload in [None, [], {}, {"documents": []}, {"documents": "wrong"}, {"documents": [None]}]:
            with self.subTest(payload=payload):
                self.payload = payload
                with self.assertRaises(RetrievalError):
                    self.load()

    def test_manifest_field_types_hashes_and_urls_are_checked(self):
        original = json.loads(json.dumps(self.payload))
        for field, value in [("document_id", ""), ("document_name", 1), ("local_path", None),
                             ("sha256", "oops"), ("url", "file:///secret"),
                             ("url", "https://user:password@example.com"),
                             ("url", "https://@example.com"),
                             ("url", "https://example.com:badport"),
                             ("url", "https://example.com/has space"),
                             ("local_path", "alpha.pdf\u0000"),
                             ("url", "https://[broken"), ("url", " https://example.com")]:
            with self.subTest(field=field, value=value):
                self.payload = json.loads(json.dumps(original))
                self.payload["documents"][0][field] = value
                with self.assertRaises(RetrievalError):
                    self.load()

    def test_duplicate_ids_names_files_and_hashes_fail(self):
        original = json.loads(json.dumps(self.payload))
        for field in ["document_id", "document_name", "sha256"]:
            with self.subTest(field=field):
                self.payload = json.loads(json.dumps(original))
                self.payload["documents"][1][field] = self.payload["documents"][0][field]
                if field == "document_name":
                    self.payload["documents"][1]["local_path"] = "alpha.pdf"
                with self.assertRaises(RetrievalError):
                    self.load()
        self.payload = json.loads(json.dumps(original))
        self.payload["documents"].append(dict(self.payload["documents"][0]))
        with self.assertRaisesRegex(RetrievalError, "Duplicate"):
            self.load()

    def test_absolute_paths_and_uppercase_hashes_work(self):
        for item in self.payload["documents"]:
            item["local_path"] = str(self.root / item["local_path"])
            item["sha256"] = item["sha256"].upper()
        self.assertEqual(self.load().search("revenue")[0].passage.document_id, "a")

    def test_nonblank_queries_and_bounded_integer_limits(self):
        adapter = self.load()
        for query in ["", "  ", None, [], "!!!"]:
            with self.subTest(query=query), self.assertRaises(RetrievalError):
                adapter.search(query)
        for limit in [0, -1, MAX_RESULTS + 1, True, 1.0, "5", None]:
            with self.subTest(limit=limit), self.assertRaises(RetrievalError):
                adapter.search("revenue", limit)

    def test_manifest_read_errors_are_clear(self):
        with self.assertRaisesRegex(RetrievalError, "manifest"):
            LocalKeywordAdapter(self.root / "missing.json")
        self.manifest.write_text("not JSON", encoding="utf-8")
        with self.assertRaisesRegex(RetrievalError, "manifest"):
            LocalKeywordAdapter(self.manifest)


if __name__ == "__main__":
    unittest.main()
