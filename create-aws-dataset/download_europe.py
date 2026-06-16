#!/usr/bin/env python3
"""
Télécharge des photos d'oiseaux européens depuis iNaturalist Open Data (AWS S3).

Utilise la base SQLite locale pour identifier les observations « research grade »
géolocalisées en Europe, puis télécharge les photos via AWS CLI (--no-sign-request).

Pour les 20 espèces de jardin, un filtre géographique France est appliqué en priorité.

Génère automatiquement label_map.json et metadata.json dans dataset/europe/.
Les noms communs (FR/EN) sont récupérés via Wikidata SPARQL.

Usage:
    python download_europe.py [--max-per-species 500] [--min-observations 50] [--workers 8] [--dry-run]
"""

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

DB_PATH = Path(__file__).parent / "iNaturalist.db"
DATASET_DIR = Path(__file__).parent.parent / "dataset" / "europe"
S3_BUCKET = "s3://inaturalist-open-data/photos"

EUROPE_BOUNDS = {"lat_min": 35, "lat_max": 72, "lon_min": -25, "lon_max": 45}
FRANCE_BOUNDS = {"lat_min": 41.3, "lat_max": 51.1, "lon_min": -5.1, "lon_max": 9.6}

GARDEN_SPECIES = {
    "Aegithalos caudatus", "Carduelis carduelis", "Chloris chloris",
    "Coloeus monedula", "Columba livia", "Columba palumbus",
    "Corvus corone", "Cyanistes caeruleus", "Erithacus rubecula",
    "Fringilla coelebs", "Parus major", "Passer domesticus",
    "Periparus ater", "Pica pica", "Prunella modularis",
    "Streptopelia decaocto", "Sturnus vulgaris", "Troglodytes troglodytes",
    "Turdus merula", "Turdus philomelos",
}

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_BATCH_SIZE = 50

MIN_PHOTO_WIDTH = 400
MIN_PHOTO_HEIGHT = 400


def get_european_species(db: sqlite3.Connection, min_obs: int) -> list[dict]:
    query = """
        SELECT t.taxon_id, t.name, COUNT(*) as obs_count
        FROM observations o
        JOIN taxa t ON o.taxon_id = t.taxon_id
        WHERE t.rank = 'species'
          AND (t.ancestry LIKE '%/3/%' OR t.ancestry LIKE '%/3')
          AND o.latitude BETWEEN :lat_min AND :lat_max
          AND o.longitude BETWEEN :lon_min AND :lon_max
          AND o.quality_grade = 'research'
        GROUP BY t.taxon_id
        HAVING obs_count >= :min_obs
        ORDER BY obs_count DESC
    """
    cursor = db.execute(query, {**EUROPE_BOUNDS, "min_obs": min_obs})
    return [{"taxon_id": r[0], "name": r[1], "count": r[2]} for r in cursor]


def get_photo_ids(db: sqlite3.Connection, taxon_id: int, limit: int,
                  bounds: dict) -> list[dict]:
    query = """
        SELECT p.photo_id, p.extension
        FROM photos p
        JOIN observations o ON p.observation_uuid = o.observation_uuid
        WHERE o.taxon_id = :taxon_id
          AND o.latitude BETWEEN :lat_min AND :lat_max
          AND o.longitude BETWEEN :lon_min AND :lon_max
          AND o.quality_grade = 'research'
          AND p.position = 0
          AND p.width >= :min_w
          AND p.height >= :min_h
        GROUP BY o.observer_id, o.observed_on, ROUND(o.latitude, 2), ROUND(o.longitude, 2)
        ORDER BY RANDOM()
        LIMIT :limit
    """
    cursor = db.execute(query, {
        **bounds, "taxon_id": taxon_id, "limit": limit,
        "min_w": MIN_PHOTO_WIDTH, "min_h": MIN_PHOTO_HEIGHT,
    })
    return [{"photo_id": r[0], "extension": r[1]} for r in cursor]


def get_family_for_species(db: sqlite3.Connection, taxon_id: int) -> str:
    row = db.execute("SELECT ancestry FROM taxa WHERE taxon_id = ?", (taxon_id,)).fetchone()
    if not row or not row[0]:
        return "Unknown"
    ancestor_ids = row[0].split("/")
    placeholders = ",".join("?" * len(ancestor_ids))
    cursor = db.execute(
        f"SELECT name FROM taxa WHERE taxon_id IN ({placeholders}) AND rank = 'family'",
        ancestor_ids,
    )
    result = cursor.fetchone()
    return result[0] if result else "Unknown"


def fetch_common_names_batch(scientific_names: list[str]) -> dict[str, dict]:
    """Récupère les noms FR/EN depuis Wikidata SPARQL par batch (via curl)."""
    results = {}
    total_batches = (len(scientific_names) + WIKIDATA_BATCH_SIZE - 1) // WIKIDATA_BATCH_SIZE

    for i in range(0, len(scientific_names), WIKIDATA_BATCH_SIZE):
        batch = scientific_names[i:i + WIKIDATA_BATCH_SIZE]
        values = " ".join(f'"{name}"' for name in batch)
        sparql = f"""
        SELECT ?scientificName ?frName ?enName WHERE {{
          VALUES ?scientificName {{ {values} }}
          ?species wdt:P225 ?scientificName .
          OPTIONAL {{ ?species rdfs:label ?frName . FILTER(LANG(?frName) = "fr") }}
          OPTIONAL {{ ?species rdfs:label ?enName . FILTER(LANG(?enName) = "en") }}
        }}
        """
        result = subprocess.run(
            ["curl", "-s", "-G", WIKIDATA_ENDPOINT,
             "--data-urlencode", f"query={sparql}",
             "--data-urlencode", "format=json",
             "-H", "User-Agent: BirdDetectionBot/1.0"],
            capture_output=True, text=True,
        )
        try:
            data = json.loads(result.stdout)
            for binding in data["results"]["bindings"]:
                name = binding["scientificName"]["value"]
                results[name] = {
                    "french_name": binding.get("frName", {}).get("value"),
                    "english_name": binding.get("enName", {}).get("value"),
                }
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Wikidata batch {i//WIKIDATA_BATCH_SIZE + 1} echoue : {e}")

        batch_num = i // WIKIDATA_BATCH_SIZE + 1
        found = sum(1 for b in batch if b in results)
        print(f"  Wikidata batch {batch_num}/{total_batches} : {found}/{len(batch)} noms trouves")

    return results


def download_photo(photo_id: int, extension: str, dest_dir: Path) -> bool:
    dest = dest_dir / f"{photo_id}.{extension.lower()}"
    if dest.exists():
        return True
    s3_path = f"{S3_BUCKET}/{photo_id}/medium.{extension}"
    result = subprocess.run(
        ["aws", "s3", "cp", s3_path, str(dest), "--no-sign-request"],
        capture_output=True,
    )
    if result.returncode != 0:
        dest.unlink(missing_ok=True)
        return False
    return True


def make_slug(scientific_name: str) -> str:
    return scientific_name.lower().replace(" ", "_")


def generate_label_map(species_list: list[dict], dest: Path):
    label_map = {}
    for i, sp in enumerate(sorted(species_list, key=lambda s: make_slug(s["name"]))):
        label_map[make_slug(sp["name"])] = i
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)
    print(f"label_map.json : {len(label_map)} especes")


def generate_metadata(species_list: list[dict], db: sqlite3.Connection, dest: Path):
    print("Recuperation des noms communs via Wikidata...")
    all_names = [sp["name"] for sp in species_list]
    common_names = fetch_common_names_batch(all_names)

    found = sum(1 for n in all_names if n in common_names)
    missing = [n for n in all_names if n not in common_names]
    print(f"  Noms trouves : {found}/{len(all_names)}")
    if missing:
        print(f"  Manquants ({len(missing)}) : {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}")

    metadata = {}
    for sp in sorted(species_list, key=lambda s: s["name"]):
        slug = make_slug(sp["name"])
        family = get_family_for_species(db, sp["taxon_id"])
        names = common_names.get(sp["name"], {})
        metadata[slug] = {
            "slug": slug,
            "scientific_name": sp["name"],
            "family": family,
            "english_name": names.get("english_name"),
            "french_name": names.get("french_name"),
        }

    with open(dest, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    filled = sum(1 for v in metadata.values() if v["french_name"] and v["english_name"])
    print(f"metadata.json : {len(metadata)} especes, {filled} avec noms FR+EN complets")


def main():
    parser = argparse.ArgumentParser(
        description="Telecharge des photos d'oiseaux europeens depuis iNaturalist (AWS S3)"
    )
    parser.add_argument("--max-per-species", type=int, default=500)
    parser.add_argument("--min-observations", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--species", nargs="*")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Genere uniquement label_map.json et metadata.json")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"Base de donnees non trouvee : {DB_PATH}")
        sys.exit(1)

    db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    print(f"Recherche des especes europeennes (seuil >= {args.min_observations} obs)...")
    species_list = get_european_species(db, args.min_observations)
    print(f"  {len(species_list)} especes trouvees")

    garden_in_list = [s for s in species_list if s["name"] in GARDEN_SPECIES]
    print(f"  dont {len(garden_in_list)} especes de jardin (filtre France active)")

    if args.species:
        filter_set = {s.lower() for s in args.species}
        species_list = [s for s in species_list if s["name"].lower() in filter_set]
        print(f"  Filtre a {len(species_list)} especes demandees")

    if args.dry_run:
        print(f"\n{'Espece':<40} {'Obs':>8} {'Jardin':>8}")
        print("-" * 60)
        for sp in species_list:
            tag = "  FR *" if sp["name"] in GARDEN_SPECIES else ""
            print(f"  {sp['name']:<38} {sp['count']:>8}{tag}")
        print(f"\nTotal : {len(species_list)} especes")
        db.close()
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    generate_label_map(species_list, DATASET_DIR / "label_map.json")
    generate_metadata(species_list, db, DATASET_DIR / "metadata.json")

    if args.metadata_only:
        db.close()
        return

    train_dir = DATASET_DIR / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    total_downloaded = 0
    total_failed = 0
    total_skipped = 0

    for i, sp in enumerate(species_list, 1):
        slug = make_slug(sp["name"])
        sp_dir = train_dir / slug
        sp_dir.mkdir(exist_ok=True)

        existing = len(list(sp_dir.glob("*.*")))
        if existing >= args.max_per_species:
            total_skipped += 1
            continue

        remaining = args.max_per_species - existing
        is_garden = sp["name"] in GARDEN_SPECIES

        photos = []
        if is_garden:
            photos = get_photo_ids(db, sp["taxon_id"], remaining, FRANCE_BOUNDS)
            fr_count = len(photos)
            if len(photos) < remaining:
                extra = get_photo_ids(db, sp["taxon_id"], remaining, EUROPE_BOUNDS)
                existing_ids = {p["photo_id"] for p in photos}
                for p in extra:
                    if p["photo_id"] not in existing_ids and len(photos) < remaining:
                        photos.append(p)
            label = f"FR:{fr_count}+EU:{len(photos)-fr_count}"
        else:
            photos = get_photo_ids(db, sp["taxon_id"], remaining, EUROPE_BOUNDS)
            label = f"EU:{len(photos)}"

        print(f"[{i}/{len(species_list)}] {sp['name']}"
              f" ({sp['count']} obs) - {label}"
              f"{'  * jardin' if is_garden else ''}")

        downloaded = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(download_photo, p["photo_id"], p["extension"], sp_dir): p
                for p in photos
            }
            for future in as_completed(futures):
                if future.result():
                    downloaded += 1
                else:
                    failed += 1

        total_downloaded += downloaded
        total_failed += failed
        if downloaded > 0 or failed > 0:
            print(f"  -> {downloaded} telechargees, {failed} echouees")

    db.close()
    print(f"\nTermine : {total_downloaded} photos telechargees, "
          f"{total_failed} echouees, {total_skipped} especes deja completes")


if __name__ == "__main__":
    main()
