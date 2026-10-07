from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from financial_analyst.adapters import EvidenceError, FileFixtureAdapter
from financial_analyst.knowledge import FinancialContext, KnowledgeError, ReviewedKnowledgeIndex
from financial_analyst.retrieval import MAX_RESULTS
from tests.support import _write_pdf, create_fixture


@unittest.skipUnless(importlib.util.find_spec("pdfplumber"), "PDF dependency unavailable")
class ReviewedKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = create_fixture(Path(self.directory.name))
        self.payload = json.loads(self.path.read_text())
        self.adapter = FileFixtureAdapter(self.path)
        self.index = ReviewedKnowledgeIndex.from_fixture(self.adapter)

    def load_changed(self):
        self.path.write_text(json.dumps(self.payload), encoding="utf-8")
        return ReviewedKnowledgeIndex.from_fixture(FileFixtureAdapter(self.path))

    def search(self, query, **filters):
        return self.index.search(query, "Example Pharma", "1QFY27", **filters)

    def test_company_period_filters_are_exact_and_applied_before_ranking(self):
        self.assertTrue(self.search("revenue"))
        self.assertEqual(self.index.search("revenue", "Other Company", "1QFY27"), ())
        self.assertEqual(self.index.search("revenue", "example pharma", "1QFY27"), ())
        self.assertEqual(self.index.search("revenue", "Example Pharma", "2QFY27"), ())
        self.assertEqual(self.index.search("revenue", "Example Pharma", "1qfy27"), ())

    def test_all_optional_filters_apply_exactly(self):
        self.assertTrue(self.search("120", scope="consolidated", metric="net_sales",
                                    kind="reported_actual", currency="INR", unit="INRm"))
        for filter in [{"scope": "standalone"}, {"metric": "profit"}, {"kind": "broker_forecast"},
                       {"currency": "USD"}, {"unit": "million"}, {"unit": "INRb"}]:
            with self.subTest(filter=filter):
                self.assertEqual(self.search("revenue", **filter), ())

    def test_mixed_context_fields_cannot_form_false_conjunction(self):
        context_record = next(record for record in self.index.records if record.ref == "context")
        self.assertIn(("1QFY26", "reported_actual"), {(item.period, item.kind) for item in context_record.contexts})
        self.assertIn(("1QFY27", "broker_estimate"), {(item.period, item.kind) for item in context_record.contexts})
        self.assertEqual(self.index.search("Qtr", "Example Pharma", "1QFY26", kind="broker_estimate"), ())
        prior = self.index.search("80", "Example Pharma", "1QFY26", kind="reported_actual")
        self.assertEqual([hit.record.ref for hit in prior], ["prior"])

    def test_numeric_actual_estimate_and_commentary_are_distinct(self):
        actuals = self.search("120", kind="reported_actual")
        self.assertIn("actual", [hit.record.ref for hit in actuals])
        self.assertNotIn("estimate", [hit.record.ref for hit in actuals])
        estimates = self.search("100", kind="broker_estimate")
        self.assertIn("estimate", [hit.record.ref for hit in estimates])
        commentary = self.search("portfolio", kind="broker_commentary")
        self.assertEqual({hit.record.ref for hit in commentary}, {"growth", "portfolio"})
        self.assertTrue(all({item.kind for item in hit.matching_contexts} == {"broker_commentary"} for hit in commentary))
        self.assertEqual(self.index.search("portfolio", "Example Pharma", "1QFY26", kind="broker_commentary"), ())

    def test_a_reviewed_forecast_column_cannot_match_an_actual_filter(self):
        source = Path(self.payload["source"]["local_path"])
        evidence = {item["ref"]: item for item in self.payload["evidence"]}
        _write_pdf(source, [
            [(40, 760, "Example Pharma"), (40, 710, evidence["status"]["excerpt"]),
             (40, 690, evidence["summary"]["excerpt"])],
            [(40, 760, "Example Pharma"), (40, 700, "Qtr Perf. (Consol.)"), (450, 700, "(INRm)"),
             (40, 680, "Y/E March"), (200, 680, "FY27E"), (300, 680, "FY27E"), (450, 680, "FY27E"),
             (200, 660, "2Q"), (300, 660, "1Q"), (450, 660, "1QE"),
             (40, 640, "Net Sales"), (202.22, 640, "80"), (296.66, 640, "120"), (453.33, 640, "100")],
            [(40, 760, "Example Pharma"), (40, 710, evidence["growth"]["excerpt"]),
             (40, 690, evidence["portfolio"]["excerpt"]), (40, 650, "Y/E December")],
        ])
        import pdfplumber
        with pdfplumber.open(source) as pdf:
            words = pdf.pages[1].extract_words()
        group = evidence["context"]["bindings"]["column_groups"][0]
        for binding, text in [(group, "FY27E"), (group["columns"][0], "2Q")]:
            word = next(item for item in words if item["text"] == text and abs(item["x0"] - 200) < .1)
            binding.update(text=text, bbox_pt=[word["x0"], word["top"], word["x1"], word["bottom"]])
        forecast = self.payload["observations"][-1]
        forecast.update(period="2QFY27", kind="broker_forecast", source_header_raw="FY27E / 2Q")
        self.payload["source"]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        index = self.load_changed()
        self.assertEqual(index.search("80", "Example Pharma", "2QFY27", kind="reported_actual"), ())
        hits = index.search("80", "Example Pharma", "2QFY27", kind="broker_forecast")
        self.assertEqual([hit.record.ref for hit in hits], ["prior"])
        self.assertEqual(hits[0].matching_contexts[0].period, "2QFY27")

    def test_prefix_is_deterministic_and_separate_from_unchanged_quote(self):
        record = next(record for record in self.index.records if record.ref == "actual")
        expected = "company=Example Pharma | period=1QFY27 | scope=consolidated | metric=net_sales | kind=reported_actual | currency=INR | unit=INRm | page=2"
        self.assertEqual(record.context_prefix, expected)
        self.assertEqual(record.quote, "120")
        self.assertEqual(record.index_text, expected + "\n120")
        self.assertEqual(record.quote, self.adapter.resolve(record.ref).excerpt)
        self.payload["observations"].reverse()
        self.assertEqual(self.index.records, self.load_changed().records)

    def test_matching_prefix_excludes_other_contexts(self):
        hits = self.index.search("Qtr", "Example Pharma", "1QFY26", kind="reported_actual")
        context_hit = next(hit for hit in hits if hit.record.ref == "context")
        self.assertNotIn("period=1QFY27", context_hit.context_prefix)
        self.assertNotIn("broker_estimate", context_hit.context_prefix)
        # The excluded fiscal period cannot manufacture a lexical match from
        # the table heading, whose exact quote contains no period at all.
        self.assertEqual(self.index.search("1QFY27", "Example Pharma", "1QFY26", kind="reported_actual"), ())

    def test_evidence_identity_and_support_closure_are_retained(self):
        for record in self.index.records:
            evidence = self.adapter.resolve(record.ref)
            self.assertEqual(record.quote, evidence.excerpt)
            self.assertEqual(record.citation, evidence.citation)
            self.assertEqual(record.document_name, self.adapter.document_name)
            self.assertEqual(record.source_sha256, self.adapter.source_sha256)
            self.assertEqual(record.supporting_evidence_refs, tuple(sorted(set(record.supporting_evidence_refs))))
            self.assertIn(record.ref, record.supporting_evidence_refs)
            self.assertEqual({item.ref for item in record.supporting_citations}, set(record.supporting_evidence_refs))
        growth = next(record for record in self.index.records if record.ref == "portfolio")
        self.assertIn("actual", growth.supporting_evidence_refs)
        self.assertIn("growth", growth.supporting_evidence_refs)

    def test_unassociated_reviewed_evidence_and_raw_pages_are_excluded(self):
        all_refs = {item.ref for item in self.adapter.reviewed_evidence()}
        indexed_refs = {item.ref for item in self.index.records}
        self.assertLess(indexed_refs, all_refs)  # fiscal conflict refs are reviewed but uncontextualized
        self.assertFalse(any(item.ref.startswith("source-year-end") for item in self.index.records))
        self.assertEqual(self.search("December"), ())

    def test_source_hash_and_metadata_tampering_fail_before_preparation(self):
        self.payload["source"]["sha256"] = "0" * 64
        with self.assertRaises(EvidenceError):
            self.load_changed()
        self.payload["source"]["sha256"] = hashlib.sha256(Path(self.payload["source"]["local_path"]).read_bytes()).hexdigest()
        self.payload["observations"][0]["period"] = "2QFY27"
        with self.assertRaises(EvidenceError):
            self.load_changed()

    def test_invalid_growth_metadata_is_not_silently_ignored(self):
        self.payload["growth_explanation_refs"] = ["status"]
        with self.assertRaises(EvidenceError):
            self.load_changed()

    def test_queries_and_limits_are_validated_and_no_hits_are_bounded_outcomes(self):
        self.assertEqual(self.search("unmatchedtoken"), ())
        self.assertEqual(len(self.search("revenue", limit=1)), 1)
        for query in ["", "  ", None, [], "!!!"]:
            with self.subTest(query=query), self.assertRaises(KnowledgeError):
                self.search(query)
        for limit in [0, -1, MAX_RESULTS + 1, True, 1.0, "5"]:
            with self.subTest(limit=limit), self.assertRaises(KnowledgeError):
                self.search("revenue", limit=limit)
        for field in ["scope", "metric", "kind", "currency", "unit"]:
            with self.subTest(field=field), self.assertRaises(KnowledgeError):
                self.search("revenue", **{field: ""})
        for company, period in [("", "1QFY27"), ("Example Pharma", ""), (None, "1QFY27")]:
            with self.subTest(company=company, period=period), self.assertRaises(KnowledgeError):
                self.index.search("revenue", company, period)

    def test_immutable_contexts_records_and_hits(self):
        hit = self.search("revenue")[0]
        with self.assertRaises(FrozenInstanceError):
            hit.record.quote = "Invented quote"
        with self.assertRaises(FrozenInstanceError):
            hit.matching_contexts[0].period = "2QFY27"
        with self.assertRaises(FrozenInstanceError):
            hit.score = 100
        with self.assertRaises(KnowledgeError):
            FinancialContext("", "1QFY27", "consolidated", "net_sales", "reported_actual", "INR", "INRm")
        with self.assertRaises(KnowledgeError):
            ReviewedKnowledgeIndex.from_fixture(object())


if __name__ == "__main__":
    unittest.main()
