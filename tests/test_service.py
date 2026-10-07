from dataclasses import replace
from decimal import Decimal, localcontext
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from financial_analyst.adapters import EvidenceError, FileFixtureAdapter
from financial_analyst.service import ApplicationService
from tests.support import create_fixture


@unittest.skipUnless(importlib.util.find_spec("pdfplumber"), "optional PDF fixture dependency unavailable")
class FixtureServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = create_fixture(Path(self.directory.name))
        self.payload = json.loads(self.path.read_text())

    def load(self):
        self.path.write_text(json.dumps(self.payload))
        return FileFixtureAdapter(self.path)

    def assert_rejected(self, fragment=None):
        with self.assertRaises(EvidenceError) as result:
            self.load()
        if fragment:
            self.assertIn(fragment, str(result.exception))

    def evidence(self, ref):
        return next(item for item in self.payload["evidence"] if item["ref"] == ref)

    def test_supported_comparison_and_claim_evidence_closure(self):
        adapter = self.load()
        answer = ApplicationService(adapter, adapter, adapter).comparison_answer("Example Pharma", "1QFY27", True, True)
        self.assertEqual(answer.status, "answered")
        self.assertEqual(answer.comparison.delta_millions, Decimal("20"))
        self.assertEqual(answer.comparison.beat_percent, Decimal("20"))
        self.assertEqual(answer.comparison.yoy_percent, Decimal("50"))
        comparison, yoy, *growth = answer.claims
        self.assertEqual(set(comparison.evidence_refs), {"context", "actual", "estimate", "status", "summary", "growth"})
        self.assertIn("prior", yoy.evidence_refs)
        self.assertEqual({claim.kind for claim in growth}, {"growth"})
        self.assertTrue(all(claim.values["attribution"] == "broker" for claim in growth))
        citation_refs = {citation.ref for citation in answer.citations}
        self.assertTrue(all(set(claim.evidence_refs) <= citation_refs for claim in answer.claims))
        self.assertEqual(answer.execution["retrieval"], "curated_fixture")
        self.assertTrue(answer.execution["no_llm"])
        self.assertTrue(answer.execution["no_database"])
        self.assertTrue(all(citation.link is None for citation in answer.citations))
        self.assertNotIn(self.directory.name, str(answer))

    def test_expected_answer_fields_are_not_authority(self):
        self.payload["expected_calculation"] = {"delta": "-9000", "beat_percent": "-9000"}
        self.payload["unsupported_question"]["expected_response"] = "Invented driver allocation."
        adapter = self.load()
        service = ApplicationService(adapter, adapter, adapter)
        self.assertEqual(service.comparison_answer("Example Pharma", "1QFY27").comparison.delta_millions, Decimal("20"))
        refusal = service.refuse_beat_attribution("Example Pharma", "1QFY27", "oncology")
        self.assertEqual(refusal.status, "refused")
        self.assertNotIn("Invented", refusal.reason)
        self.assertTrue(refusal.citations)

    def test_hash_mismatch(self):
        self.payload["source"]["sha256"] = "0" * 64
        self.assert_rejected("identity")

    def test_excerpt_tampering(self):
        self.evidence("growth")["excerpt"] = "The document never said this."
        self.assert_rejected("absent")

    def test_unresolved_ref(self):
        self.payload["observations"][0]["evidence_refs"].append("unresolved")
        self.assert_rejected("resolved")

    def test_wrong_document_and_page(self):
        self.evidence("actual")["document_name"] = "other.pdf"
        self.assert_rejected("identity")
        self.evidence("actual")["document_name"] = self.payload["source"]["document_name"]
        self.evidence("actual")["page"] = 7
        self.assert_rejected("page")

    def test_registry_cannot_rename_the_original_source(self):
        self.payload["source"]["document_name"] = "invented-report.pdf"
        for item in self.payload["evidence"]:
            item["document_name"] = "invented-report.pdf"
        self.assert_rejected("document name")

    def test_renamed_resolvable_ref_still_cannot_forge_cell(self):
        self.evidence("actual")["ref"] = "forged"
        self.payload["observations"][0]["evidence_refs"][1] = "forged"
        self.assert_rejected("column identity")

    def test_wrong_cell_box(self):
        self.evidence("actual")["bbox_pt"] = self.evidence("estimate")["bbox_pt"]
        self.assert_rejected("PDF location")

    def test_same_number_observation_but_wrong_fiscal_header(self):
        self.payload["observations"][0]["source_header_raw"] = "FY26 / 1Q"
        self.assert_rejected("fiscal column")

    def test_header_binding_tampering(self):
        self.evidence("context")["bindings"]["unit"]["text"] = "(INRb)"
        self.assert_rejected("source label")

    def test_citation_locator_comes_from_verified_source_identity(self):
        self.evidence("actual")["locator"] = "Operating Profit row / FY26 / 1Q"
        self.evidence("growth")["locator"] = "Unrelated quotation location"
        adapter = self.load()
        answer = ApplicationService(adapter, adapter, adapter).comparison_answer("Example Pharma", "1QFY27", True)
        citations = {citation.ref: citation for citation in answer.citations}
        self.assertEqual(citations["actual"].locator, "Net Sales row / FY27E / 1Q")
        self.assertNotIn("Unrelated", citations["growth"].locator)

    def test_source_amount_and_scope_mismatches(self):
        self.payload["observations"][0]["value"] = "121"
        self.assert_rejected("value")
        self.payload["observations"][0]["value"] = "120"
        self.payload["observations"][0]["scope"] = "standalone"
        self.assert_rejected("scope")

    def test_actual_status_needs_both_reported_result_pages(self):
        self.payload["observations"][0]["evidence_refs"].remove("status")
        self.assert_rejected("pages 1 and 3")
        self.payload["observations"][0]["evidence_refs"].append("status")
        self.payload["observations"][0]["evidence_refs"].remove("growth")
        self.assert_rejected("pages 1 and 3")

    def test_valid_but_unrelated_growth_reference_is_refused(self):
        self.payload["growth_explanation_refs"].append("estimate")
        adapter = self.load()
        answer = ApplicationService(adapter, adapter, adapter).comparison_answer("Example Pharma", "1QFY27", True)
        self.assertEqual(answer.status, "refused")
        self.assertIn("revenue-growth", answer.reason)

    def test_calendar_refusal_and_conflicts_have_authenticated_citations(self):
        self.payload["source_conflicts"][0]["handling"] = "Invented interpretation."
        adapter = self.load()
        answer = ApplicationService(adapter, adapter, adapter).refuse_calendar_dates()
        self.assertEqual(answer.status, "refused")
        self.assertIn("conflicting", answer.reason)
        self.assertNotIn("Invented", str(answer))
        self.assertEqual(len(answer.source_conflicts), 1)
        self.assertTrue(set(answer.source_conflicts[0]["evidence_refs"]) <= {citation.ref for citation in answer.citations})

    def test_growth_only_keeps_calculation_out_of_claims(self):
        adapter = self.load()
        answer = ApplicationService(adapter, adapter, adapter).growth_answer("Example Pharma", "1QFY27")
        self.assertEqual(answer.status, "answered")
        self.assertIsNone(answer.comparison)
        self.assertTrue(all(claim.kind == "growth" for claim in answer.claims))

    def test_read_port_cannot_relabel_valid_source_observations(self):
        adapter = self.load()

        class WrongReadPort:
            def read_observations(self, company, period):
                return adapter.read_observations("Example Pharma", "1QFY27")

        answer = ApplicationService(WrongReadPort(), adapter).comparison_answer("Other Company", "1QFY42")
        self.assertEqual(answer.status, "refused")
        self.assertIn("different company or quarter", answer.reason)

    def test_malformed_fixture_shapes_fail_with_bounded_error(self):
        for payload in [[], {"source": []}, {**self.payload, "evidence": ["invalid"]},
                        {**self.payload, "observations": [{"value": []}]}]:
            with self.subTest(payload_type=type(payload).__name__):
                self.path.write_text(json.dumps(payload))
                with self.assertRaises(EvidenceError):
                    FileFixtureAdapter(self.path)

    def test_prior_amount_normalization_ignores_ambient_decimal_precision(self):
        adapter = self.load()
        originals = [*adapter.read_observations("Example Pharma", "1QFY27"),
                     *adapter.read_observations("Example Pharma", "1QFY26")]
        prior = replace(originals[-1], value=Decimal("123456789.123456789"), source_unit="billion",
                        source_unit_raw="INRb", source_value_raw="123456789.123456789")

        class SyntheticRead:
            def read_observations(self, company, period):
                return [item for item in [*originals[:-1], prior] if item.company == company and item.period == period]

        class SyntheticEvidence:
            def validate_observation(self, observation):
                pass

            def resolve(self, ref):
                return adapter.resolve(ref)

        with localcontext() as context:
            context.prec = 6
            answer = ApplicationService(SyntheticRead(), SyntheticEvidence()).comparison_answer("Example Pharma", "1QFY27", include_yoy=True)
        self.assertEqual(answer.status, "answered")
        self.assertEqual(answer.claims[1].values["prior_actual"], Decimal("123456789123.456789"))


if __name__ == "__main__":
    unittest.main()
