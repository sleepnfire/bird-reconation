#!/usr/bin/env python3
"""
Complète les espèces sous-représentées du dataset europe/ avec des photos iNaturalist.

Stratégie en deux paliers :
  1. Europe d'abord (mêmes bornes que download_europe.py)
  2. Mondial en complément si la cible n'est pas atteinte

Télécharge via boto3 (SDK Python AWS) avec accès public sans signature.
Les photos vont dans dataset/europe_to_trait/{slug}/ (structure plate).

Génère un manifest JSON pour la reprise après interruption.

Usage:
    python supplement_europe.py --dry-run
    python supplement_europe.py --species "Bubo scandiacus"
    python supplement_europe.py
    python supplement_europe.py --manifest dataset/europe_to_trait/manifest.json
"""

import argparse
import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

DB_PATH = Path(__file__).parent / "iNaturalist.db"
DATASET_DIR = Path(__file__).parent.parent / "dataset"
EUROPE_DIR = DATASET_DIR / "europe"
EUROPE_REJECTED_DIR = DATASET_DIR / "europe_rejected"
OUTPUT_DIR = DATASET_DIR / "europe_to_trait"

S3_BUCKET = "inaturalist-open-data"
S3_PREFIX = "photos"

EUROPE_BOUNDS = {"lat_min": 35, "lat_max": 72, "lon_min": -25, "lon_max": 45}

MIN_PHOTO_WIDTH = 400
MIN_PHOTO_HEIGHT = 400

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

AUTO_RELAX_THRESHOLD = 50


def make_slug(scientific_name: str) -> str:
    return scientific_name.lower().replace(" ", "_")


def load_metadata(metadata_path: Path) -> dict:
    with open(metadata_path, encoding="utf-8") as f:
        return json.load(f)


def count_existing_images(europe_dir: Path) -> dict[str, int]:
    counts = {}
    for split in ("train", "validation", "test"):
        split_dir = europe_dir / split
        if not split_dir.exists():
            continue
        for sp_dir in split_dir.iterdir():
            if sp_dir.is_dir():
                slug = sp_dir.name
                n = sum(1 for f in sp_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
                counts[slug] = counts.get(slug, 0) + n
    return counts


def collect_excluded_photo_ids(slug: str) -> set[int]:
    excluded = set()
    for base_dir in (EUROPE_DIR, EUROPE_REJECTED_DIR):
        for split in ("train", "validation", "test"):
            sp_dir = base_dir / split / slug
            if not sp_dir.exists():
                continue
            for f in sp_dir.iterdir():
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    try:
                        excluded.add(int(f.stem))
                    except ValueError:
                        pass
    sp_dir = OUTPUT_DIR / slug
    if sp_dir.exists():
        for f in sp_dir.iterdir():
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    excluded.add(int(f.stem))
                except ValueError:
                    pass
    return excluded


def get_taxon_id(db: sqlite3.Connection, scientific_name: str) -> int | None:
    row = db.execute(
        "SELECT taxon_id FROM taxa WHERE name = ? AND rank = 'species' "
        "ORDER BY active DESC LIMIT 1",
        (scientific_name,),
    ).fetchone()
    return row[0] if row else None


def get_photos(db: sqlite3.Connection, taxon_id: int, limit: int,
               excluded_ids: set[int], max_position: int = 0,
               bounds: dict | None = None) -> list[dict]:
    params = {
        "taxon_id": taxon_id,
        "limit": limit,
        "min_w": MIN_PHOTO_WIDTH,
        "min_h": MIN_PHOTO_HEIGHT,
        "max_pos": max_position,
    }
    geo_clause = ""
    if bounds:
        geo_clause = ("AND o.latitude BETWEEN :lat_min AND :lat_max "
                       "AND o.longitude BETWEEN :lon_min AND :lon_max")
        params.update(bounds)

    excluded_clause = ""
    if excluded_ids:
        placeholders = ",".join(str(pid) for pid in excluded_ids)
        excluded_clause = f"AND p.photo_id NOT IN ({placeholders})"

    query = f"""
        SELECT p.photo_id, p.extension
        FROM photos p
        JOIN observations o ON p.observation_uuid = o.observation_uuid
        WHERE o.taxon_id = :taxon_id
          AND o.quality_grade = 'research'
          AND p.position <= :max_pos
          AND p.width >= :min_w
          AND p.height >= :min_h
          {geo_clause}
          {excluded_clause}
        GROUP BY o.observer_id, o.observed_on,
                 ROUND(o.latitude, 2), ROUND(o.longitude, 2)
        ORDER BY RANDOM()
        LIMIT :limit
    """
    cursor = db.execute(query, params)
    return [{"photo_id": r[0], "extension": r[1]} for r in cursor]


def count_available(db: sqlite3.Connection, taxon_id: int,
                    excluded_ids: set[int], max_position: int = 0,
                    bounds: dict | None = None) -> int:
    params = {
        "taxon_id": taxon_id,
        "min_w": MIN_PHOTO_WIDTH,
        "min_h": MIN_PHOTO_HEIGHT,
        "max_pos": max_position,
    }
    geo_clause = ""
    if bounds:
        geo_clause = ("AND o.latitude BETWEEN :lat_min AND :lat_max "
                       "AND o.longitude BETWEEN :lon_min AND :lon_max")
        params.update(bounds)

    excluded_clause = ""
    if excluded_ids:
        placeholders = ",".join(str(pid) for pid in excluded_ids)
        excluded_clause = f"AND p.photo_id NOT IN ({placeholders})"

    query = f"""
        SELECT COUNT(*) FROM (
            SELECT p.photo_id
            FROM photos p
            JOIN observations o ON p.observation_uuid = o.observation_uuid
            WHERE o.taxon_id = :taxon_id
              AND o.quality_grade = 'research'
              AND p.position <= :max_pos
              AND p.width >= :min_w
              AND p.height >= :min_h
              {geo_clause}
              {excluded_clause}
            GROUP BY o.observer_id, o.observed_on,
                     ROUND(o.latitude, 2), ROUND(o.longitude, 2)
        )
    """
    return db.execute(query, params).fetchone()[0]


def build_manifest(db: sqlite3.Connection, metadata: dict,
                   existing_counts: dict, target: int,
                   allow_position: int, min_download: int = 0) -> dict:
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "target": target,
        "species": {},
    }

    species_list = sorted(metadata.keys())
    total = len(species_list)

    for i, slug in enumerate(species_list, 1):
        meta = metadata[slug]
        sci_name = meta["scientific_name"]
        existing = existing_counts.get(slug, 0)
        needed = max(0, target - existing)

        if needed == 0:
            continue

        if min_download > 0 and 0 < needed < min_download:
            needed = min_download

        taxon_id = get_taxon_id(db, sci_name)
        if taxon_id is None:
            print(f"  [{i}/{total}] {sci_name} : taxon_id introuvable, ignore",
                  flush=True)
            continue

        excluded = collect_excluded_photo_ids(slug)

        photos_eu = get_photos(db, taxon_id, needed, excluded,
                               max_position=allow_position,
                               bounds=EUROPE_BOUNDS)
        eu_ids = {p["photo_id"] for p in photos_eu}
        eu_count = len(photos_eu)

        photos_world = []
        remaining = needed - eu_count
        if remaining > 0:
            excluded_with_eu = excluded | eu_ids
            max_pos = allow_position

            if max_pos == 0:
                avail_pos0 = count_available(db, taxon_id, excluded_with_eu,
                                             max_position=0)
                if avail_pos0 < AUTO_RELAX_THRESHOLD:
                    max_pos = 2

            photos_world = get_photos(db, taxon_id, remaining,
                                      excluded_with_eu,
                                      max_position=max_pos)

        all_photos = (
            [{"photo_id": p["photo_id"], "extension": p["extension"],
              "source": "europe"} for p in photos_eu]
            + [{"photo_id": p["photo_id"], "extension": p["extension"],
                "source": "worldwide"} for p in photos_world]
        )

        world_count = len(photos_world)
        total_found = eu_count + world_count
        status = "ok" if total_found >= needed else "insuffisant"

        manifest["species"][slug] = {
            "scientific_name": sci_name,
            "existing": existing,
            "needed": needed,
            "found_europe": eu_count,
            "found_worldwide": world_count,
            "photos": all_photos,
        }

        label = f"EU:{eu_count}"
        if world_count > 0:
            label += f" + MONDE:{world_count}"
        flag = "  ** INSUFFISANT **" if status == "insuffisant" else ""
        print(f"  [{i}/{total}] {sci_name:<40} "
              f"existant:{existing:>4}  besoin:{needed:>4}  "
              f"trouve:{total_found:>4} ({label}){flag}",
              flush=True)

    return manifest


def download_photo(s3_client, photo_id: int, extension: str,
                   dest_dir: Path) -> bool:
    dest = dest_dir / f"{photo_id}.{extension.lower()}"
    if dest.exists():
        return True
    key = f"{S3_PREFIX}/{photo_id}/medium.{extension}"
    try:
        s3_client.download_file(S3_BUCKET, key, str(dest))
        return True
    except Exception:
        dest.unlink(missing_ok=True)
        return False


def download_from_manifest(manifest: dict, workers: int):
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    all_tasks = []
    for slug, sp_data in manifest["species"].items():
        sp_dir = OUTPUT_DIR / slug
        sp_dir.mkdir(parents=True, exist_ok=True)
        for photo in sp_data["photos"]:
            dest = sp_dir / f"{photo['photo_id']}.{photo['extension'].lower()}"
            if not dest.exists():
                all_tasks.append((slug, photo, sp_dir))

    if not all_tasks:
        print("Rien a telecharger (tout est deja sur disque).")
        return

    print(f"\nTelechargement de {len(all_tasks)} photos avec {workers} workers...",
          flush=True)

    downloaded = 0
    failed = 0
    failed_details = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_photo, s3_client,
                        task[1]["photo_id"], task[1]["extension"],
                        task[2]): task
            for task in all_tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            if future.result():
                downloaded += 1
            else:
                failed += 1
                failed_details.append(
                    f"{task[0]}/{task[1]['photo_id']}.{task[1]['extension']}")

            done = downloaded + failed
            if done % 500 == 0 or done == len(all_tasks):
                print(f"  Progression : {done}/{len(all_tasks)} "
                      f"({downloaded} ok, {failed} echecs)",
                      flush=True)

    print(f"\nTermine : {downloaded} telechargees, {failed} echouees",
          flush=True)
    if failed_details:
        print(f"Echecs ({len(failed_details)}) :")
        for d in failed_details[:20]:
            print(f"  - {d}")
        if len(failed_details) > 20:
            print(f"  ... et {len(failed_details) - 20} de plus")


def print_summary(manifest: dict):
    total_species = len(manifest["species"])
    total_photos = sum(len(sp["photos"]) for sp in manifest["species"].values())
    total_eu = sum(sp["found_europe"] for sp in manifest["species"].values())
    total_world = sum(sp["found_worldwide"] for sp in manifest["species"].values())

    insufficient = [
        (slug, sp) for slug, sp in manifest["species"].items()
        if len(sp["photos"]) < sp["needed"]
    ]

    print(f"\n{'=' * 70}")
    print(f"RESUME")
    print(f"{'=' * 70}")
    print(f"Especes a completer : {total_species}")
    print(f"Photos a telecharger : {total_photos} (EU: {total_eu}, Monde: {total_world})")
    print(f"Especes completement satisfaites : {total_species - len(insufficient)}/{total_species}")

    if insufficient:
        print(f"\nEspeces insuffisantes ({len(insufficient)}) :")
        for slug, sp in sorted(insufficient, key=lambda x: len(x[1]["photos"])):
            found = len(sp["photos"])
            print(f"  {sp['scientific_name']:<40} "
                  f"existant:{sp['existing']:>4} + trouve:{found:>4} "
                  f"= {sp['existing'] + found:>4} / {manifest['target']}")


def main():
    parser = argparse.ArgumentParser(
        description="Complete les especes sous-representees avec des photos iNaturalist mondiales"
    )
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true",
                        help="Genere le manifest sans telecharger")
    parser.add_argument("--species", nargs="*",
                        help="Noms scientifiques des especes a completer")
    parser.add_argument("--manifest", type=Path,
                        help="Reprendre depuis un manifest existant")
    parser.add_argument("--allow-position", type=int, default=0,
                        help="Position max des photos (0=primaire, auto-relax pour especes rares)")
    parser.add_argument("--min-download", type=int, default=0,
                        help="Minimum d'images a telecharger par espece (0=desactive)")
    args = parser.parse_args()

    if args.manifest and args.manifest.exists():
        print(f"Reprise depuis le manifest : {args.manifest}")
        with open(args.manifest, encoding="utf-8") as f:
            manifest = json.load(f)
        print_summary(manifest)
        download_from_manifest(manifest, args.workers)
        return

    if not DB_PATH.exists():
        print(f"Base de donnees non trouvee : {DB_PATH}")
        sys.exit(1)

    metadata_path = EUROPE_DIR / "metadata.json"
    if not metadata_path.exists():
        print(f"metadata.json non trouve : {metadata_path}")
        sys.exit(1)

    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    metadata = load_metadata(metadata_path)

    if args.species:
        filter_slugs = {make_slug(s) for s in args.species}
        metadata = {k: v for k, v in metadata.items() if k in filter_slugs}
        print(f"Filtre sur {len(metadata)} espece(s)")

    print(f"Comptage des images existantes dans {EUROPE_DIR}...")
    existing_counts = count_existing_images(EUROPE_DIR)
    filtered_counts = {slug: existing_counts.get(slug, 0) for slug in metadata}
    total_existing = sum(filtered_counts.values())
    print(f"  {len(filtered_counts)} especes selectionnees, {total_existing} images",
          flush=True)

    need_supplement = sum(1 for c in filtered_counts.values() if c < args.target)
    print(f"  {need_supplement} especes sous la cible de {args.target}",
          flush=True)

    print(f"\nConstruction du manifest (requetes SQLite)...", flush=True)
    manifest = build_manifest(db, metadata, existing_counts,
                              args.target, args.allow_position,
                              args.min_download)
    db.close()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest sauvegarde : {manifest_path}", flush=True)

    print_summary(manifest)

    if args.dry_run:
        print("\n(dry-run : pas de telechargement)")
        return

    download_from_manifest(manifest, args.workers)


if __name__ == "__main__":
    main()
