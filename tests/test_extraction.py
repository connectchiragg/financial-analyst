from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from financial_analyst.extraction import ExtractionError, extract_document, prepare_document, validate_extracted_document
from financial_analyst.inference import InferenceResponse, ProviderError
from tests.support import _write_pdf


class FakeInference:
    provider, model, mode = "synthetic", "extract-test", "test_double"
    capabilities = frozenset({"json_schema"})
    execution = {}

    def __init__(self, change=None, response=None, on_call=None):
        self.change, self.response, self.on_call = change, response, on_call
        self.calls = []

    def infer(self, request):
        self.calls.append(request)
        payload = json.loads(request.messages[1].content)
        first = payload["passages"][0]
        if self.on_call:
            self.on_call()
        if self.response is not None:
            if isinstance(self.response, Exception):
                raise self.response
            return self.response
        fact = {"company": payload["company"], "period": "1QFY27", "scope": "consolidated", "metric": "net_sales",
                "kind": None, "currency": "INR", "unit": "million", "value_text": "1200",
                "source_value_raw": "1,200", "source_metric_raw": "Net Sales", "source_period_raw": "1QFY27",
                "source_scope_raw": "consolidated", "source_unit_raw": "INRm", "source_kind_raw": "reported_actual",
                "evidence_refs": [first["ref"]]}
        if self.change:
            self.change(fact)
        facts = [fact] if first["page"] == 1 else []
        return InferenceResponse(json.dumps({"facts": facts}), "stop")


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "synthetic-extraction.pdf"
        _write_pdf(self.source, [
            [(40, 760, "Example Pharma"), (40, 740, "1QFY27 consolidated reported_actual"), (40, 720, "Net Sales INRm 1,200")],
            [(40, 760, "Historical source text")], [(40, 760, "Other source text")], [(40, 760, "Beyond primary page coverage")]])
        self.args = {"document_id": "synthetic", "document_name": self.source.name, "url": "https://example.test/source",
                     "company": "Example Pharma", "agency": "Synthetic Research"}

    def extract(self, inference=None, **options):
        inference = inference or FakeInference()
        return extract_document(self.source, **self.args, inference=inference, **options), inference

    def test_all_pages_stored_with_exact_canonical_refs_and_explicit_primary_coverage(self):
        bundle, inference = self.extract()
        self.assertEqual(bundle.source.sha256, hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(bundle.coverage.stored_pages, (1, 2, 3, 4))
        self.assertEqual(bundle.coverage.extracted_pages, (1, 2, 3))
        self.assertEqual(bundle.coverage.uncovered_pages, (4,))
        self.assertEqual(len(inference.calls), 3)
        self.assertEqual(bundle.facts[0].value_text, "1200")
        self.assertEqual(bundle.facts[0].source_value_raw, "1,200")
        self.assertEqual(bundle.facts[0].source_unit_raw, "INRm")
        self.assertEqual(bundle.facts[0].review_state, "quote_validated_pending_review")
        self.assertTrue(all(p.review_state == "unreviewed" for p in bundle.passages))
        self.assertFalse(bundle.execution["answerable"])
        self.assertTrue(bundle.execution["no_llm"])
        self.assertEqual(bundle.execution["evidence_resolution"], "full_canonical_passage")
        self.assertIsNone(bundle.facts[0].kind)
        self.assertEqual(bundle.facts[0].source_kind_raw, "reported_actual")
        self.assertEqual(bundle.facts[0].evidence[0].quote, bundle.passages[0].excerpt)
        validate_extracted_document(bundle, self.source)
        second, _ = self.extract()
        self.assertEqual(bundle.facts, second.facts)

    def test_window_and_selected_page_limits_expose_uncovered_pages(self):
        bundle, inference = self.extract(max_windows=1)
        self.assertEqual(len(inference.calls), 1)
        self.assertEqual(bundle.coverage.extracted_pages, (1,))
        self.assertEqual(bundle.coverage.uncovered_pages, (2, 3, 4))
        self.assertEqual(bundle.execution["available_windows"], 3)
        last, inference = self.extract(pages=(4,), max_windows=1)
        self.assertEqual(last.coverage.extracted_pages, (4,))
        self.assertEqual(last.coverage.uncovered_pages, (1, 2, 3))
        self.assertEqual(len(inference.calls), 1)

    def test_invalid_windows_and_pages_never_call_model(self):
        for options in [{"max_windows": True}, {"max_windows": 0}, {"max_windows": 9},
                        {"pages": ()}, {"pages": [1]}, {"pages": (True,)}, {"pages": (1, 1)}, {"pages": (5,)}]:
            inference = FakeInference()
            with self.subTest(options=options), self.assertRaises(ExtractionError):
                self.extract(inference, **options)
            self.assertEqual(inference.calls, [])

    def test_wrong_quotes_refs_labels_values_company_and_units_reject_every_fact(self):
        changes = [lambda f: f.update(company="Other Pharma"), lambda f: f.update(value_text="999"),
                   lambda f: f.update(value_text="NaN"), lambda f: f.update(currency="USD"),
                   lambda f: f.update(unit="billion"), lambda f: f.update(source_metric_raw="Invented Revenue"),
                   lambda f: f.update(kind="actual"), lambda f: f.update(extra="unsafe"),
                   lambda f: f.update(kind="reported_actual"),
                   lambda f: f.update(source_scope_raw=None), lambda f: f.update(evidence_refs=["invented"]),
                   lambda f: f.update(evidence=[{"ref": "invented", "quote": "Model authored quote"}]),
                   lambda f: f.update(financial_claim="Model authored financial answer"),
                   lambda f: f.update(evidence_refs=f["evidence_refs"] * 2),
                   lambda f: f.update(evidence_refs=[])]
        for change in changes:
            inference = FakeInference(change=change)
            with self.subTest(change=change), self.assertRaises(ExtractionError) as failure:
                self.extract(inference)
            self.assertEqual(failure.exception.source_document.facts, ())
            self.assertEqual(failure.exception.source_document.coverage.inference_calls, 1)
            self.assertEqual(failure.exception.source_document.coverage.attempted_pages, (1,))
            self.assertEqual(failure.exception.source_document.coverage.extracted_pages, ())
            self.assertEqual(failure.exception.source_document.execution["extraction"], "failed")
            self.assertEqual(len(inference.calls), 1)

    def test_ref_from_unselected_source_page_cannot_anchor_proposal(self):
        prepared = prepare_document(self.source, **self.args)
        later_ref = next(p.ref for p in prepared.passages if p.page == 4)
        inference = FakeInference(change=lambda f: f.update(evidence_refs=[later_ref]))
        with self.assertRaises(ExtractionError) as failure:
            self.extract(inference, pages=(1,), max_windows=1)
        self.assertEqual(failure.exception.source_document.facts, ())
        self.assertEqual(len(inference.calls), 1)

    def test_proposal_budget_is_small_and_does_not_claim_complete_fact_coverage(self):
        bundle, inference = self.extract(max_windows=1)
        self.assertEqual(json.loads(inference.calls[0].messages[1].content)["max_facts"], 3)
        self.assertEqual(bundle.execution["fact_coverage"], "selected_proposals_only")
        self.assertLessEqual(inference.calls[0].max_output_tokens, 4096)

    def test_exact_sibling_header_is_resolved_but_value_must_be_selected(self):
        lines = [(40, 760, "1QFY27 consolidated reported_actual INRm")]
        lines += [(40, 740 - index * 15, "Context filler words " * 4) for index in range(20)]
        lines += [(40, 410, "Net Sales 1,200")]
        _write_pdf(self.source, [lines])

        class ValueInference(FakeInference):
            def infer(self, request):
                response = super().infer(request)
                data = json.loads(response.content)
                passages = json.loads(request.messages[1].content)["passages"]
                data["facts"][0]["evidence_refs"] = [next(p["ref"] for p in passages if "Net Sales" in p["text"])]
                return InferenceResponse(json.dumps(data), "stop")

        bundle, _ = self.extract(ValueInference(), pages=(1,), max_windows=1)
        fact = bundle.facts[0]
        self.assertEqual(len(fact.evidence), 2)
        self.assertIn("Net Sales", fact.evidence[0].quote)
        self.assertIn("1QFY27", fact.evidence[1].quote)
        self.assertEqual(bundle.execution["context_anchor_resolution"], "literal_selected_window")
        validate_extracted_document(bundle, self.source)
        with self.assertRaises(ExtractionError):
            self.extract(FakeInference(), pages=(1,), max_windows=1)

    def test_header_from_unselected_page_is_not_borrowed(self):
        _write_pdf(self.source, [[(40, 760, "Net Sales 1,200")],
                                 [(40, 760, "1QFY27 consolidated reported_actual INRm")]])
        with self.assertRaises(ExtractionError) as failure:
            self.extract(FakeInference(), pages=(1,), max_windows=1)
        self.assertEqual(failure.exception.source_document.facts, ())

    def test_source_literal_schema_anchors_are_not_invented(self):
        bundle, inference = self.extract(max_windows=1)
        properties = inference.calls[0].schema["properties"]["facts"]["items"]["properties"]
        self.assertEqual(properties["kind"]["enum"], [None])
        self.assertEqual(properties["evidence_refs"]["items"]["enum"], ["p1"])
        for name in ("source_period_raw", "source_scope_raw", "source_unit_raw", "source_kind_raw"):
            self.assertTrue(all(value is None or value in bundle.passages[0].excerpt
                                for value in properties[name]["enum"]))

    def test_ambiguous_context_remains_null_and_unreviewed(self):
        def change(f):
            for field in ("period", "scope", "kind", "currency", "unit", "source_period_raw", "source_scope_raw", "source_kind_raw", "source_unit_raw"):
                f[field] = None
        bundle, _ = self.extract(FakeInference(change=change), max_windows=1)
        self.assertIsNone(bundle.facts[0].kind)
        self.assertIsNone(bundle.facts[0].scope)
        self.assertFalse(bundle.execution["answerable"])
        validate_extracted_document(bundle, self.source)

    def test_malformed_unfinished_extra_json_and_provider_failure_have_no_fallback(self):
        responses = [InferenceResponse("not JSON", "stop"), InferenceResponse('{"facts": [], "answer": "invented"}', "stop"),
                     InferenceResponse('{"facts": [], "facts": []}', "stop"), InferenceResponse('{"facts": []}', "length"),
                     ProviderError("private-payload-and-key")]
        for response in responses:
            with self.subTest(response=response), self.assertRaises(ExtractionError) as failure:
                self.extract(FakeInference(response=response))
            bundle = failure.exception.source_document
            self.assertEqual(bundle.facts, ())
            self.assertEqual(bundle.coverage.inference_calls, 1)
            self.assertNotIn("private-payload", str(failure.exception))
            self.assertEqual(len(bundle.passages), 4)
            validate_extracted_document(bundle, self.source)

    def test_source_mutation_during_model_execution_is_rejected_before_store(self):
        inference = FakeInference(on_call=lambda: self.source.write_bytes(b"replaced after authenticated extraction"))
        bundle, _ = self.extract(inference, max_windows=1)
        with self.assertRaises(ExtractionError):
            validate_extracted_document(bundle, self.source)

    def test_prepared_source_requires_no_llm_and_cannot_be_promoted(self):
        prepared = prepare_document(self.source, **self.args)
        self.assertEqual(prepared.facts, ())
        self.assertEqual(prepared.coverage.inference_calls, 0)
        self.assertEqual(prepared.coverage.uncovered_pages, (1, 2, 3, 4))
        validate_extracted_document(prepared, self.source)
        bundle, _ = self.extract(max_windows=1)
        with self.assertRaises(ExtractionError):
            validate_extracted_document(replace(bundle, facts=(replace(bundle.facts[0], review_state="reviewed"),)), self.source)
        with self.assertRaises(ExtractionError):
            validate_extracted_document(replace(prepared, coverage=replace(prepared.coverage, inference_calls=True)), self.source)

    def test_no_text_request_cannot_reuse_prior_live_execution(self):
        _write_pdf(self.source, [[], [], []])
        inference = FakeInference()
        inference.mode = "live"
        inference.execution = {"mode": "live", "outcome": "completed"}
        bundle, _ = self.extract(inference)
        self.assertEqual(inference.calls, [])
        self.assertEqual(bundle.execution["llm"]["mode"], "not_called")
        self.assertTrue(bundle.execution["no_llm"])
        self.assertEqual(bundle.coverage.extracted_pages, ())
        self.assertEqual(bundle.coverage.uncovered_pages, (1, 2, 3))
        validate_extracted_document(bundle, self.source)


if __name__ == "__main__":
    unittest.main()
