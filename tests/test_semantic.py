from dataclasses import FrozenInstanceError, replace
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from financial_analyst.knowledge import KnowledgeError, ReviewedKnowledgeIndex
from financial_analyst.retrieval import MAX_RESULTS
from financial_analyst.selection import GroqPassageSelector, ProviderError, SelectionAbstention, SelectionError
from financial_analyst.semantic import MAX_SEMANTIC_CANDIDATES, SemanticGrowthAdapter, SemanticKnowledgeRetriever
from financial_analyst.adapters import FileFixtureAdapter
from tests.support import create_fixture
from tests.test_selection import FakeClient


class FakeSelector:
    def __init__(self, refs=("portfolio",), error=None):
        self.refs = refs
        self.error = error
        self.calls = []
        self.query = None
        self.contexts = None
        self.execution = {"provider": "synthetic", "model": "test", "mode": "not_called"}

    def select(self, company, period, passages):
        self.calls.append((company, period, passages, self.query, dict(self.contexts)))
        self.execution = {"provider": "synthetic", "model": "test", "mode": "test_double"}
        if self.error:
            raise self.error
        return self.refs


@unittest.skipUnless(importlib.util.find_spec("pdfplumber"), "PDF dependency unavailable")
class SemanticRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = FileFixtureAdapter(create_fixture(Path(self.directory.name)))
        self.index = ReviewedKnowledgeIndex.from_fixture(self.fixture)

    def retriever(self, refs=("portfolio",), error=None, index=None):
        selector = FakeSelector(refs, error)
        return SemanticKnowledgeRetriever(index or self.index, selector), selector

    def test_semantic_query_with_no_keyword_overlap_still_reaches_candidates(self):
        query = "What propelled expansion?"
        self.assertEqual(self.index.search(query, "Example Pharma", "1QFY27", kind="broker_commentary"), ())
        retriever, selector = self.retriever()
        hits = retriever.search(query, "Example Pharma", "1QFY27", kind="broker_commentary")
        self.assertEqual([hit.record.ref for hit in hits], ["portfolio"])
        self.assertEqual(selector.query, query)
        self.assertEqual({passage.ref for passage in selector.calls[0][2]}, {"growth", "portfolio"})
        self.assertEqual(retriever.mode, "llm_semantic_selection")
        self.assertEqual(retriever.execution["mode"], "test_double")

    def test_exact_filters_reject_wrong_company_period_and_kind_before_a_call(self):
        retriever, selector = self.retriever()
        for company, period, filters in [
            ("Other Company", "1QFY27", {}), ("Example Pharma", "2QFY27", {}),
            ("Example Pharma", "1QFY27", {"kind": "broker_forecast"}),
            ("Example Pharma", "1QFY27", {"scope": "standalone"}),
            ("Example Pharma", "1QFY27", {"currency": "USD"}),
        ]:
            with self.subTest(company=company, period=period, filters=filters):
                self.assertEqual(retriever.search("expansion", company, period, **filters), ())
                self.assertEqual(retriever.execution["mode"], "not_called")
                self.assertEqual(selector.contexts, {})
        self.assertEqual(selector.calls, [])

    def test_only_matching_complete_contexts_enter_the_prompt(self):
        retriever, selector = self.retriever(("prior",))
        hits = retriever.search("historic turnover", "Example Pharma", "1QFY26", kind="reported_actual")
        self.assertEqual({item.ref for item in selector.calls[0][2]}, {"context", "prior"})
        for prefix in selector.contexts.values():
            self.assertIn("period=1QFY26", prefix)
            self.assertNotIn("period=1QFY27", prefix)
            self.assertNotIn("broker_estimate", prefix)
        self.assertEqual(hits[0].record.quote, "80")
        self.assertEqual(retriever.search("turnover", "Example Pharma", "1QFY26", kind="broker_estimate"), ())
        self.assertEqual(len(selector.calls), 1)

    def test_selection_order_is_ordinal_and_canonical_quotes_stay_unchanged(self):
        retriever, selector = self.retriever(("portfolio", "growth"))
        hits = retriever.search("reasons", "Example Pharma", "1QFY27", kind="broker_commentary")
        self.assertEqual([hit.rank for hit in hits], [1, 2])
        self.assertEqual([hit.record.ref for hit in hits], ["portfolio", "growth"])
        for hit in hits:
            evidence = self.fixture.resolve(hit.record.ref)
            self.assertEqual(hit.record.quote, evidence.excerpt)
            self.assertEqual(hit.record.citation, evidence.citation)
            self.assertFalse(hasattr(hit, "score"))
        self.assertEqual(selector.calls[0][2][0].excerpt, self.fixture.resolve(selector.calls[0][2][0].ref).excerpt)
        limited = retriever.search("reasons", "Example Pharma", "1QFY27", kind="broker_commentary", limit=1)
        self.assertEqual([hit.record.ref for hit in limited], ["portfolio"])
        self.assertEqual(len(selector.calls[-1][2]), 2)
        with self.assertRaises(FrozenInstanceError):
            hits[0].rank = 100

    def test_any_injected_selector_is_checked_for_unknown_duplicate_and_bad_refs(self):
        for refs in [("invented",), ("portfolio", "portfolio"), (1,), "portfolio", {"selected_refs": ["portfolio"]}]:
            with self.subTest(refs=refs):
                retriever, selector = self.retriever(refs)
                with self.assertRaises(SelectionError):
                    retriever.search("reasons", "Example Pharma", "1QFY27", kind="broker_commentary")
                self.assertEqual(len(selector.calls), 1)

    def test_abstention_is_an_empty_search_outcome_without_a_fallback(self):
        for refs, error in [((), None), (("portfolio",), SelectionAbstention("No relevant records."))]:
            with self.subTest(explicit_error=error is not None):
                retriever, selector = self.retriever(refs, error)
                self.assertEqual(retriever.search("unsupported meaning", "Example Pharma", "1QFY27", kind="broker_commentary"), ())
                self.assertEqual(len(selector.calls), 1)
                self.assertEqual(retriever.execution["mode"], "test_double")

    def test_malformed_selection_and_provider_errors_propagate_without_fallback(self):
        for error in [SelectionError("Malformed response."), ProviderError("Provider unavailable.")]:
            with self.subTest(error=type(error).__name__):
                retriever, selector = self.retriever(error=error)
                with self.assertRaises(type(error)):
                    retriever.search("reasons", "Example Pharma", "1QFY27", kind="broker_commentary")
                self.assertEqual(len(selector.calls), 1)

    def test_nohit_request_does_not_inherit_a_previous_model_invocation(self):
        retriever, selector = self.retriever()
        retriever.search("reasons", "Example Pharma", "1QFY27", kind="broker_commentary")
        self.assertEqual(retriever.execution["mode"], "test_double")
        self.assertEqual(retriever.search("reasons", "Unknown", "1QFY27"), ())
        self.assertEqual(retriever.execution, {"provider": "synthetic", "model": "test", "mode": "not_called"})
        self.assertEqual(len(selector.calls), 1)

    def test_candidate_budget_fails_without_truncating_or_calling_selector(self):
        record = next(item for item in self.index.records if item.ref == "portfolio")
        records = tuple(replace(record, ref=f"candidate-{index}", citation=replace(record.citation, ref=f"candidate-{index}"))
                        for index in range(MAX_SEMANTIC_CANDIDATES + 1))
        retriever, selector = self.retriever(index=ReviewedKnowledgeIndex(records))
        with self.assertRaisesRegex(SelectionError, "32-candidate budget"):
            retriever.search("reasons", "Example Pharma", "1QFY27", kind="broker_commentary")
        self.assertEqual(selector.calls, [])
        self.assertEqual(len(selector.contexts), MAX_SEMANTIC_CANDIDATES + 1)
        self.assertEqual(retriever.execution["mode"], "not_called")

    def test_selector_request_size_budget_blocks_transport_without_fallback(self):
        record = next(item for item in self.index.records if item.ref == "portfolio")
        index = ReviewedKnowledgeIndex((replace(record, quote="large source quote " * 1000),))
        client = FakeClient({"abstain": False, "selected_refs": ["portfolio"]})
        selector = GroqPassageSelector(client=client)
        retriever = SemanticKnowledgeRetriever(index, selector)
        with self.assertRaisesRegex(SelectionError, "bounded inference request size"):
            retriever.search("reasons", "Example Pharma", "1QFY27", kind="broker_commentary")
        self.assertEqual(client.calls, [])
        self.assertEqual(retriever.execution["mode"], "not_called")

    def test_groq_test_client_receives_semantic_query_and_canonical_prefixes(self):
        client = FakeClient({"abstain": False, "selected_refs": ["portfolio"]})
        selector = GroqPassageSelector(client=client)
        retriever = SemanticKnowledgeRetriever(self.index, selector)
        hits = retriever.search("What propelled expansion?", "Example Pharma", "1QFY27", kind="broker_commentary")
        self.assertEqual(hits[0].record.ref, "portfolio")
        content = json.loads(client.calls[0]["messages"][1]["content"])
        self.assertEqual(content["query"], "What propelled expansion?")
        for passage in content["passages"]:
            self.assertEqual(passage["excerpt"], self.fixture.resolve(passage["ref"]).excerpt)
            self.assertIn("kind=broker_commentary", passage["derived_context"])
        self.assertEqual(retriever.execution["mode"], "test_double")

    def test_growth_adapter_resolves_canonical_evidence_after_semantic_selection(self):
        selector = FakeSelector(("portfolio", "growth"))
        adapter = SemanticGrowthAdapter(self.fixture, selector)
        passages = adapter.growth_passages("Example Pharma", "1QFY27")
        self.assertEqual(passages, (self.fixture.resolve("portfolio"), self.fixture.resolve("growth")))
        self.assertEqual({item.ref for item in selector.calls[0][2]}, {"portfolio", "growth"})
        self.assertTrue(all("kind=broker_commentary" in prefix for prefix in selector.contexts.values()))
        self.assertEqual(adapter.source_sha256, self.fixture.source_sha256)
        self.assertEqual(adapter.mode, "llm_semantic_selection")
        self.assertEqual(adapter.execution["mode"], "test_double")

    def test_semantic_query_and_result_limits_fail_before_a_call(self):
        retriever, selector = self.retriever()
        for query in [None, "", "  ", []]:
            with self.subTest(query=query), self.assertRaises(KnowledgeError):
                retriever.search(query, "Example Pharma", "1QFY27")
        for limit in [0, -1, MAX_RESULTS + 1, True, 1.0, "5"]:
            with self.subTest(limit=limit), self.assertRaises(KnowledgeError):
                retriever.search("reasons", "Example Pharma", "1QFY27", limit=limit)
        self.assertEqual(selector.calls, [])
        self.assertEqual(retriever.execution["mode"], "not_called")


if __name__ == "__main__":
    unittest.main()
