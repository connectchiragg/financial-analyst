"""Synthetic, non-private PDF fixture used by integration and CLI tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _write_pdf(path: Path, pages: list[list[tuple[float, float, str]]]) -> None:
    """Write a tiny deterministic PDF using only the standard library."""
    objects: list[bytes] = []
    page_ids = [4 + index * 2 for index in range(len(pages))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{index} 0 R" for index in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for index, lines in enumerate(pages):
        page_id = page_ids[index]
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 600 800] /Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>".encode())
        commands = []
        for x, y, text in lines:
            escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            commands.append(f"BT /F1 10 Tf 1 0 0 1 {x} {y} Tm ({escaped}) Tj ET")
        stream = "\n".join(commands).encode("ascii")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(content))
        content.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode())
    content.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(content)


def create_fixture(directory: Path) -> Path:
    import pdfplumber

    directory.mkdir(parents=True, exist_ok=True)
    pdf_path = directory / "synthetic-report.pdf"
    status = "Example Pharma delivered revenue in 1QFY27."
    summary = "Revenue increased to INR120m (est: INR100m)."
    growth = "Example Pharma reported 1QFY27 revenue of INR120m, driven by portfolio execution."
    portfolio = "The oncology portfolio supported revenue growth."
    _write_pdf(pdf_path, [
        [(40, 760, "Example Pharma"), (40, 710, status), (40, 690, summary)],
        [(40, 760, "Example Pharma"), (40, 700, "Qtr Perf. (Consol.)"), (450, 700, "(INRm)"),
         (40, 680, "Y/E March"), (200, 680, "FY26"), (300, 680, "FY27E"), (450, 680, "FY27E"),
         (200, 660, "1Q"), (300, 660, "1Q"), (450, 660, "1QE"),
         (40, 640, "Net Sales"), (202.22, 640, "80"), (296.66, 640, "120"), (453.33, 640, "100")],
        [(40, 760, "Example Pharma"), (40, 710, growth), (40, 690, portfolio), (40, 650, "Y/E December")],
    ])
    with pdfplumber.open(pdf_path) as pdf:
        words = pdf.pages[1].extract_words()

    def box(text: str, x: float, y: float) -> list[float]:
        selected = [word for word in words if word["x0"] >= x - .1
                    and abs(word["top"] - (800 - y - 7.93)) < .1]
        parts = text.split()
        selected = selected[:len(parts)]
        assert " ".join(word["text"] for word in selected) == text
        return [selected[0]["x0"], selected[0]["top"], selected[-1]["x1"], selected[-1]["bottom"]]

    def binding(text: str, x: float, y: float) -> dict:
        return {"text": text, "bbox_pt": box(text, x, y)}

    groups = []
    for group, x, quarter, ref in [("FY26", 200, "1Q", "prior"), ("FY27E", 300, "1Q", "actual"), ("FY27E", 450, "1QE", "estimate")]:
        bound = binding(group, x, 680)
        column = binding(quarter, x, 660)
        column["cell_refs"] = [ref]
        bound["columns"] = [column]
        groups.append(bound)
    evidence = [{"ref": "context", "document_name": pdf_path.name, "page": 2,
                 "locator": "Quarterly table", "excerpt": "Qtr Perf. (Consol.)",
                 "bindings": {
                     "company": binding("Example Pharma", 40, 760),
                     "scope": binding("(Consol.)", 80.57, 700),
                     "unit": binding("(INRm)", 450, 700),
                     "year_end": binding("Y/E March", 40, 680),
                     "metric": binding("Net Sales", 40, 640),
                     "column_groups": groups,
                 }}]
    for ref, raw, x in [("actual", "120", 296.66), ("estimate", "100", 453.33), ("prior", "80", 202.22)]:
        evidence.append({"ref": ref, "document_name": pdf_path.name, "page": 2,
                         "locator": f"Net Sales / {ref} column", "excerpt": raw,
                         "bbox_pt": box(raw, x, 640), "context_refs": ["context"]})
    for ref, page, excerpt in [("status", 1, status), ("summary", 1, summary), ("growth", 3, growth), ("portfolio", 3, portfolio)]:
        evidence.append({"ref": ref, "document_name": pdf_path.name, "page": page,
                         "locator": f"{ref} passage", "excerpt": excerpt})
    observations = []
    for ref, value, period, kind, header, extra in [
        ("actual", "120", "1QFY27", "reported_actual", "FY27E / 1Q", ["status", "summary", "growth"]),
        ("estimate", "100", "1QFY27", "broker_estimate", "FY27E / 1QE", ["summary"]),
        ("prior", "80", "1QFY26", "reported_actual", "FY26 / 1Q", []),
    ]:
        observations.append({"company": "Example Pharma", "period": period, "scope": "consolidated",
                             "metric": "net_sales", "currency": "INR", "source_unit": "million",
                             "value": value, "kind": kind, "evidence_refs": ["context", ref] + extra,
                             "year_end_raw": "Y/E March", "source_header_raw": header,
                             "source_value_raw": value, "source_unit_raw": "INRm", "source_metric_raw": "Net Sales"})
    fixture = {"source": {"document_name": pdf_path.name, "local_path": str(pdf_path),
                           "sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(), "page_count": 3},
               "observations": observations, "evidence": evidence,
               "growth_explanation_refs": ["growth", "portfolio"],
               "expected_calculation": {"delta": "999"},
               "unsupported_question": {"expected_response": "Invented answer must be ignored."},
               "source_conflicts": [{"subject": "Fiscal year-end labels", "source_labels": {"quarter": "March", "annual": "December"}}]}
    fixture_path = directory / "synthetic-fixture.json"
    fixture_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    return fixture_path
