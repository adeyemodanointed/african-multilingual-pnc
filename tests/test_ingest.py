import csv
import json
import tempfile
import unittest
from pathlib import Path

from pnc.ingest import ingest_manifest


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
