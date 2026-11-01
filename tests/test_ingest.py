import csv
import json
import tempfile
import unittest
from pathlib import Path

from pnc.audit import audit_sources
from pnc.ingest import CorpusSource, ingest_manifest
from pnc.quality import select_high_confidence_records
from pnc.dataset import normalize_reference_text, prepare_record, prepare_records
from pnc.training import chunk_record, f1_from_counts, token_classification_counts


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

    def test_select_high_confidence_records_is_balanced_and_strict(self) -> None:
        records = [
            {"id": "1", "dataset_id": "yor", "language": "yor", "raw_text": "Ẹ káàrọ̀ àwọn ọ̀rẹ́ mi."},
            {"id": "2", "dataset_id": "yor", "language": "yor", "raw_text": "ẹ káàrọ̀ àwọn ọ̀rẹ́ mi."},
            {"id": "3", "dataset_id": "pcm", "language": "pcm", "raw_text": "Dis one get correct punctuation."},
            {"id": "4", "dataset_id": "pcm", "language": "pcm", "raw_text": "Too short."},
        ]
        selected, report = select_high_confidence_records(records, max_per_language=1, seed=7)

        self.assertEqual({language: len(values) for language, values in selected.items()}, {"pcm": 1, "yor": 1})
        self.assertEqual(report["yor"]["missing_initial_capital"], 1)
        self.assertEqual(report["pcm"]["too_short"], 1)

    def test_prepare_record_round_trips_punctuation_and_case(self) -> None:
        record = {
            "id": "yor-1",
            "dataset_id": "yor",
            "language": "yor",
            "raw_text": "Ẹ káàrọ̀, àwọn ọ̀rẹ́ mi.",
        }
        prepared, reason = prepare_record(record)

        self.assertIsNone(reason)
        self.assertEqual(prepared["input_text"], "ẹ káàrọ̀ àwọn ọ̀rẹ́ mi")
        self.assertEqual(prepared["input_words"], ["ẹ", "káàrọ̀", "àwọn", "ọ̀rẹ́", "mi"])
        self.assertEqual(prepared["punctuation_labels"], ["O", "COMMA", "O", "O", "PERIOD"])
        self.assertEqual(prepared["case_labels"], ["TITLE", "LOWER", "LOWER", "LOWER", "LOWER"])

    def test_prepare_records_excludes_embedded_punctuation(self) -> None:
        records = [
            {"id": "1", "dataset_id": "pcm", "language": "pcm", "raw_text": "Dis valid example ends well."},
            {"id": "2", "dataset_id": "pcm", "language": "pcm", "raw_text": "Dis quote has \"punctuation.\""},
        ]
        prepared, report = prepare_records(records)

        self.assertEqual(len(prepared), 1)
        self.assertEqual(report["pcm"]["embedded_or_unsupported_punctuation"], 1)

    def test_prepare_record_preserves_colons_and_semicolons(self) -> None:
        record = {
            "id": "yor-2",
            "dataset_id": "yor",
            "language": "yor",
            "raw_text": "Ìròyìn náà: àwọn ènìyàn dé; wọ́n sì jókòó.",
        }
        prepared, reason = prepare_record(record)

        self.assertIsNone(reason)
        self.assertEqual(prepared["punctuation_labels"], ["O", "COLON", "O", "O", "SEMICOLON", "O", "O", "PERIOD"])

    def test_normalization_separates_inline_punctuation_without_breaking_decimals(self) -> None:
        text = "Ìròyìn:Akọsilẹ.Ọjọ́ 3.14 wà."

        self.assertEqual(normalize_reference_text(text), "Ìròyìn: Akọsilẹ. Ọjọ́ 3.14 wà.")

    def test_chunking_preserves_word_label_alignment(self) -> None:
        record = {
            "id": "kin-1",
            "language": "kin",
            "words": ["A", "B", "C", "D", "E"],
            "punctuation_labels": ["O", "COMMA", "O", "O", "PERIOD"],
            "case_labels": ["TITLE", "UPPER", "UPPER", "UPPER", "UPPER"],
        }
        chunks = list(chunk_record(record, word_chunk_size=2))

        self.assertEqual([chunk["words"] for chunk in chunks], [["A", "B"], ["C", "D"], ["E"]])
        self.assertEqual(chunks[-1]["punctuation_labels"], ["PERIOD"])

    def test_metrics_ignore_subword_targets(self) -> None:
        tp, fp, fn = token_classification_counts([0, 1, 1, 2], [0, -100, 2, 1], label_count=3)
        metrics = f1_from_counts(tp, fp, fn)

        self.assertEqual(metrics["support"], [1, 1, 1])
        self.assertEqual(metrics["per_class_f1"], [1.0, 0.0, 0.0])
