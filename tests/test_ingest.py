import csv
import json
import tempfile
import unittest
from pathlib import Path

from pnc.audit import audit_sources
from pnc.ingest import CorpusSource, ingest_manifest


class IngestTests(unittest.TestCase):
    def test_ingest_jsonl_and_csv_sources(self) -> None:
        scratch_root = Path(__file__).parent / ".tmp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as temporary_directory:
            tmp_path = Path(temporary_directory)
            self._run_ingestion(tmp_path)

    def _run_ingestion(self, tmp_path: Path) -> None:
        (tmp_path / "data/raw/yor").mkdir(parents=True)
        (tmp_path / "data/raw/ibo").mkdir(parents=True)
        manifest = tmp_path / "data/manifests/corpus_inventory.csv"
        manifest.parent.mkdir(parents=True)

        (tmp_path / "data/raw/yor/source.jsonl").write_text(
            json.dumps({"text": "Ẹ káàrọ̀.", "source": "Yoruba source"}) + "\n",
            encoding="utf-8",
        )
        with (tmp_path / "data/raw/ibo/source.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["Language", "Text", "Source", "License"])
            writer.writeheader()
            writer.writerow({"Language": "ibo", "Text": "Ndewo ụwa.", "Source": "Igbo source", "License": "CC-BY"})

        manifest.write_text(
            "dataset_id,language,path,format,text_field,domain_field,source_field,split_field,license_field,role,notes\n"
            "yor_source,yor,data/raw/yor/source.jsonl,jsonl,text,,source,,,primary_pnc,test\n"
            "ibo_source,ibo,data/raw/ibo/source.csv,csv,Text,,Source,,License,primary_pnc,test\n",
            encoding="utf-8",
        )

        output = tmp_path / "data/interim/unified.jsonl"
        counts = ingest_manifest(manifest, output)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(counts, {"yor_source": 1, "ibo_source": 1})
        self.assertEqual(
            [(row["language"], row["raw_text"]) for row in rows],
            [("yor", "Ẹ káàrọ̀."), ("ibo", "Ndewo ụwa.")],
        )

    def test_audit_selects_deterministic_review_rows(self) -> None:
        scratch_root = Path(__file__).parent / ".tmp"
        scratch_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch_root) as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "data/raw/yor/source.jsonl"
            source_path.parent.mkdir(parents=True)
            source_path.write_text(
                "\n".join(
                    json.dumps({"text": text, "source": "fixture"})
                    for text in ["Àkọ́kọ́.", "ẹlẹ́ẹ̀kejì", "Ẹlẹ́ẹ̀kẹta!"]
                ) + "\n",
                encoding="utf-8",
            )
            source = CorpusSource(
                dataset_id="yor_source",
                language="yor",
                path="data/raw/yor/source.jsonl",
                format="jsonl",
                text_field="text",
                domain_field="",
                source_field="source",
                split_field="",
                license_field="",
                role="primary_pnc",
                notes="fixture",
            )
            summary, rows = audit_sources([source], root, sample_size=2, seed=7)

        self.assertEqual(summary["yor_source"]["records"], 3)
        self.assertEqual(summary["yor_source"]["terminal_punctuation_pct"], 66.67)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["language"] == "yor" for row in rows))
