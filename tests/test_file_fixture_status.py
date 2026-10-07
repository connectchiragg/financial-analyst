"""Regression checks for source-derived actual/forecast status."""

import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from financial_analyst.adapters import EvidenceError, FileFixtureAdapter
from tests.support import create_fixture


@unittest.skipUnless(importlib.util.find_spec("pdfplumber"), "optional PDF fixture dependency unavailable")
class SourceStatusTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = create_fixture(Path(self.directory.name))
        self.payload = json.loads(self.path.read_text())

    def load(self):
        self.path.write_text(json.dumps(self.payload))
        return FileFixtureAdapter(self.path)

    def test_reported_quarter_cannot_be_reclassified_as_forecast(self):
        self.payload["observations"][0]["kind"] = "broker_forecast"
        with self.assertRaisesRegex(EvidenceError, "cannot be relabeled as a forecast"):
            self.load()

    def test_removing_supplied_actual_citations_cannot_hide_reported_status(self):
        self.payload["observations"][0]["kind"] = "broker_forecast"
        self.payload["observations"][0]["evidence_refs"] = ["context", "actual"]
        self.payload["evidence"] = [item for item in self.payload["evidence"] if item["ref"] not in {"status", "growth"}]
        self.payload["growth_explanation_refs"] = []
        with self.assertRaisesRegex(EvidenceError, "cannot be relabeled as a forecast"):
            self.load()

    def test_future_quarter_with_no_reported_status_remains_a_forecast(self):
        # Re-label the synthetic source's current-column text as a future
        # quarter while keeping reported-result prose about the first quarter.
        pdf = Path(self.payload["source"]["local_path"])
        original = pdf.read_bytes()
        marker = b"1 0 0 1 300 660 Tm (1Q)"
        self.assertIn(marker, original)
        pdf.write_bytes(original.replace(marker, b"1 0 0 1 300 660 Tm (2Q)"))
        self.payload["source"]["sha256"] = hashlib.sha256(pdf.read_bytes()).hexdigest()
        context = next(item for item in self.payload["evidence"] if item["ref"] == "context")
        context["bindings"]["column_groups"][1]["columns"][0]["text"] = "2Q"
        future = self.payload["observations"][0]
        future.update(kind="broker_forecast", period="2QFY27", source_header_raw="FY27E / 2Q", evidence_refs=["context", "actual"])
        adapter = self.load()
        loaded = adapter.read_observations("Example Pharma", "2QFY27")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].kind, "broker_forecast")


if __name__ == "__main__":
    unittest.main()
