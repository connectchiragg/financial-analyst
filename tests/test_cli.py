from decimal import Decimal, Inexact, localcontext
import json
from pathlib import Path
import subprocess
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest

from financial_analyst.cli import _render_text
from support import create_fixture


PROJECT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = create_fixture(Path(self.directory.name))

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "financial_analyst", *args],
                              cwd=PROJECT, capture_output=True, text=True, check=False)

    def command(self, *args, output="json"):
        return self.run_cli("--mode", "fixture", "--fixture", str(self.fixture),
                            "--format", output, *args)

    def test_combined_json_preserves_originals_citations_and_modes(self):
        run = self.command("combined", "--company", "Example Pharma", "--period", "1QFY27", "--yoy")
        self.assertEqual(run.returncode, 0, run.stderr)
        answer = json.loads(run.stdout)
        self.assertEqual(answer["status"], "answered")
        comparison, yoy = answer["claims"][:2]
        self.assertEqual(Decimal(comparison["values"]["delta"]), Decimal("20"))
        self.assertEqual(Decimal(comparison["values"]["beat_percent"]), Decimal("20"))
        self.assertEqual(Decimal(yoy["values"]["yoy_percent"]), Decimal("50"))
        self.assertEqual(answer["comparison"]["actual"]["source_header_raw"], "FY27E / 1Q")
        citation_refs = {citation["ref"] for citation in answer["citations"]}
        for claim in answer["claims"]:
            self.assertTrue(set(claim["evidence_refs"]) <= citation_refs)
        self.assertTrue({"status", "growth"} <= set(comparison["evidence_refs"]))
        self.assertEqual(answer["execution"]["retrieval"], "curated_fixture")
        self.assertTrue(answer["execution"]["no_llm"])
        self.assertTrue(answer["execution"]["no_database"])

    def test_text_labels_source_context_and_requested_retrieval_truthfully(self):
        run = self.command("compare", "--company", "Example Pharma", "--period", "1QFY27", output="text")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("retrieval: not requested", run.stdout)
        self.assertIn("FY27E / 1Q", run.stdout)
        self.assertIn("consolidated", run.stdout)
        self.assertIn("INRm", run.stdout)
        self.assertIn("synthetic-report.pdf, pp.", run.stdout)
        self.assertIn("Y/E March", run.stdout)
        self.assertIn("Y/E December", run.stdout)
        self.assertIn("Calendar dates remain unresolved", run.stdout)

    def test_attribution_and_unknown_questions_refuse_without_claims(self):
        attribution = self.command("beat-attribution", "--company", "Example Pharma", "--period", "1QFY27")
        self.assertEqual(attribution.returncode, 1, attribution.stderr)
        answer = json.loads(attribution.stdout)
        self.assertEqual(answer["status"], "refused")
        self.assertEqual(answer["claims"], [])
        self.assertIn("allocation", answer["reason"])
        unknown = self.command("ask", "--company", "Example Pharma", "--period", "1QFY27",
                               "What was Example Pharma revenue and PAT?")
        self.assertEqual(unknown.returncode, 1)
        self.assertEqual(json.loads(unknown.stdout)["claims"], [])

    def test_question_route_and_unavailable_capabilities_do_not_fallback(self):
        run = self.command("ask", "--company", "Example Pharma", "--period", "1QFY27",
                           "By how much did Example Pharma's 1QFY27 revenue exceed the broker estimate?")
        self.assertEqual(run.returncode, 0, run.stderr)
        for flags in [("--mode", "live"), ("--mode", "fixture", "--retrieval", "semantic"),
                      ("--mode", "fixture", "--retrieval", "keyword")]:
            unavailable = self.run_cli(*flags, "compare", "--company", "Example Pharma", "--period", "1QFY27")
            self.assertEqual(unavailable.returncode, 2)
            self.assertIn("no fallback", unavailable.stderr)
            self.assertEqual(unavailable.stdout, "")

    def test_explicit_fixture_and_supported_company_period_are_required(self):
        missing = self.run_cli("--mode", "fixture", "compare", "--company", "Example Pharma", "--period", "1QFY27")
        self.assertEqual(missing.returncode, 2)
        for company, period in [("Other Pharma", "1QFY27"), ("Example Pharma", "2QFY27")]:
            run = self.command("compare", "--company", company, "--period", period)
            self.assertEqual(run.returncode, 1, run.stderr)
            self.assertEqual(json.loads(run.stdout)["claims"], [])

    def test_exact_miss_display_is_independent_of_decimal_context(self):
        amount = Decimal("123456789012345678901234567890123456789012345")
        citation = SimpleNamespace(ref="cell", document_name="synthetic-report.pdf", page=2, link=None)
        observation = SimpleNamespace(scope="consolidated", source_metric_raw="Net Sales", source_unit_raw="INRm",
                                      year_end_raw="Y/E March", source_header_raw="FY27 / 1Q", kind="reported_actual")
        claim = SimpleNamespace(kind="comparison", evidence_refs=("cell",), values={
            "company": "Example Pharma", "period": "1QFY27", "currency": "INR", "actual": Decimal("0"),
            "estimate": amount, "delta": amount.copy_negate(), "beat_percent": Decimal("-100")})
        answer = SimpleNamespace(status="answered", claims=(claim,), citations=(citation,), source_conflicts=(),
                                 comparison=SimpleNamespace(actual=observation, broker_estimate=observation),
                                 execution={"retrieval": "not_requested"})
        with localcontext() as context:
            context.prec = 6
            context.traps[Inexact] = True
            text = _render_text(answer)
        self.assertIn(f"missed by {format(amount, ',f')} INR million", text)

    def test_local_keyword_search_is_exploratory_and_not_fixture_execution(self):
        source = json.loads(self.fixture.read_text())["source"]
        manifest = Path(self.directory.name) / "manifest.json"
        manifest.write_text(json.dumps({"documents": [{"document_id": "synthetic", "document_name": source["document_name"],
            "local_path": source["local_path"], "sha256": source["sha256"], "url": "https://example.test/report"}]}))
        run = self.run_cli("--mode", "local", "--retrieval", "keyword", "--manifest", str(manifest),
                           "--format", "json", "search", "revenue growth")
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(run.stdout)
        self.assertEqual(result["status"], "retrieved")
        self.assertEqual(result["claims"], [])
        self.assertEqual(result["execution"]["mode"], "local")
        self.assertEqual(result["execution"]["context"], "unreviewed_page_text")
        self.assertTrue(result["execution"]["no_llm"])
        self.assertTrue(result["execution"]["no_database"])
        self.assertTrue(all(hit["passage"]["page"] == hit["citation"]["page"] for hit in result["hits"]))
        empty = self.run_cli("--mode", "local", "--retrieval", "keyword", "--manifest", str(manifest),
                             "--format", "json", "search", "unfindabletoken")
        self.assertEqual(empty.returncode, 0, empty.stderr)
        self.assertEqual(json.loads(empty.stdout)["status"], "no_matches")

    def test_search_configuration_never_falls_back(self):
        for flags in [("--mode", "fixture", "--retrieval", "keyword"),
                      ("--mode", "local", "--retrieval", "semantic"),
                      ("--mode", "local", "--retrieval", "keyword", "--llm", "groq")]:
            run = self.run_cli(*flags, "search", "revenue")
            self.assertEqual(run.returncode, 2)
            self.assertEqual(run.stdout, "")

    def test_ingestion_is_idempotent_and_live_comparison_reads_sqlite(self):
        database = Path(self.directory.name) / "analyst.sqlite"
        flags = ("--mode", "live", "--database", str(database), "--fixture", str(self.fixture), "--format", "json")
        ingest = self.run_cli(*flags, "ingest")
        self.assertEqual(ingest.returncode, 0, ingest.stderr)
        result = json.loads(ingest.stdout)
        self.assertEqual(result["status"], "ingested")
        self.assertEqual(result["counts"]["observations_added"], 3)
        repeated = self.run_cli(*flags, "ingest")
        self.assertEqual(repeated.returncode, 0, repeated.stderr)
        self.assertEqual(json.loads(repeated.stdout)["status"], "unchanged")
        comparison = self.run_cli(*flags, "combined", "--company", "Example Pharma", "--period", "1QFY27", "--yoy")
        self.assertEqual(comparison.returncode, 0, comparison.stderr)
        answer = json.loads(comparison.stdout)
        self.assertEqual(answer["claims"][0]["values"]["delta"], "20")
        self.assertEqual(answer["execution"]["mode"], "live")
        self.assertEqual(answer["execution"]["analytics"], "sqlite")
        self.assertEqual(answer["execution"]["database"], "sqlite")
        self.assertFalse(answer["execution"]["no_database"])
        self.assertEqual(answer["execution"]["seed_provenance"], ["reviewed_fixture"])
        self.assertTrue(answer["execution"]["no_llm"])
        connection = sqlite3.connect(database)
        try:
            connection.execute("UPDATE observations SET value_text = '999' WHERE period='1QFY27' AND kind='reported_actual'")
            connection.commit()
        finally:
            connection.close()
        tampered = self.run_cli(*flags, "compare", "--company", "Example Pharma", "--period", "1QFY27")
        self.assertEqual(tampered.returncode, 1, tampered.stderr)
        self.assertEqual(json.loads(tampered.stdout)["claims"], [])

    def test_ingestion_rejects_source_tampering_before_database_creation(self):
        database = Path(self.directory.name) / "not-created.sqlite"
        payload = json.loads(self.fixture.read_text())
        payload["source"]["sha256"] = "0" * 64
        self.fixture.write_text(json.dumps(payload))
        run = self.run_cli("--mode", "live", "--database", str(database), "--fixture", str(self.fixture), "ingest")
        self.assertEqual(run.returncode, 2)
        self.assertFalse(database.exists())


if __name__ == "__main__":
    unittest.main()
