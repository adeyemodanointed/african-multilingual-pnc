"""Extract the five canonical all-domain files from Africa_PnC_Corpus archives.

The downloaded archives also contain per-domain exports, text-only exports,
and shards.  Those representations overlap, so this utility extracts only one
all-domain CSV.gz file per language.  The raw archives and extracted files stay
outside the Git repository in ``../new_data``.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


CANONICAL_FILES = {
    "African_PnC_Corpus-20261120T183635Z-1-001.zip": {
        "African_PnC_Corpus/Igbo/igbo_ALL_DOMAINS.csv.gz": "ibo_all_domains.csv.gz",
        "African_PnC_Corpus/Kinyarwanda/kinyarwanda_ALL_DOMAINS.csv.gz": "kin_all_domains.csv.gz",
        "African_PnC_Corpus/Pidgin/pidgin_ALL_DOMAINS.csv.gz": "pcm_all_domains.csv.gz",
        "African_PnC_Corpus/Yoruba/yoruba_ALL_DOMAINS.csv.gz": "yor_all_domains.csv.gz",
    },
    "African_PnC_Corpus-20261120T094353Z-1-002.zip": {
        "African_PnC_Corpus/Swahili/swahili_ALL_DOMAINS.csv.gz": "swa_all_domains.csv.gz",
    },
}

BIBLE_FILES = {
    "ibo_ibo.jsonl": "bible_ibo.jsonl",
    "pcm_pcm.jsonl": "bible_pcm.jsonl",
    "swh_swhonen.jsonl": "bible_swa.jsonl",
    "yor_yor.jsonl": "bible_yor.jsonl",
}


def extract_file(archive_path: Path, member_name: str, destination: Path) -> str:
    with zipfile.ZipFile(archive_path) as archive:
        try:
            member = archive.getinfo(member_name)
        except KeyError as error:
            raise FileNotFoundError(f"Missing {member_name} in {archive_path.name}") from error

        if destination.exists():
            if destination.stat().st_size == member.file_size:
                return "already present"
            raise FileExistsError(
                f"{destination} exists but its size does not match {archive_path.name}:{member_name}. "
                "Remove or rename it before retrying."
            )

        with archive.open(member) as source, destination.open("xb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    return "extracted"


def extract_members(archive_path: Path, members: dict[str, str], output_dir: Path) -> None:
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    for member_name, output_name in members.items():
        destination = output_dir / output_name
        outcome = extract_file(archive_path, member_name, destination)
        print(f"{outcome:15} {destination.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=Path("../new_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("../new_data/africa_pnc_canonical"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for archive_name, members in CANONICAL_FILES.items():
        archive_path = args.archive_dir / archive_name
        extract_members(archive_path, members, args.output_dir)
    extract_members(args.archive_dir / "bible_data_export.zip", BIBLE_FILES, args.output_dir)


if __name__ == "__main__":
    main()
