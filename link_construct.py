import os
from collections import defaultdict
from dc_utils import mysql_execute_select_dict


# ---------- CONFIG ----------

IMAGE_TYPES = {
    "CAROUSEL_ALBUM", "IMAGE", "photo"
}

VIDEO_TYPES = {
    "animated_gif", "VIDEO", "video"
}

TWITTER_PLATFORM_ID = 3

START_DATE = "2025-08-01 00:00:00"
END_DATE = "2025-10-31 23:59:59" # l'ensemble des publications allant du 01/08 au 31/10


# ---------- SQL QUERY ----------

SQL_QUERY = """
SELECT
    spa.id_study,
    spa.id_plateform,
    spa.id_publication,
    pmedia.media_url,
    pmedia.type AS media_type,
    pms.status,
    p.name AS platform_name
FROM study_publication_affectation spa
LEFT JOIN publications_meta pm
    ON pm.plateform_id = spa.id_plateform
    AND pm.publication_id = spa.id_publication
LEFT JOIN plateforms p
    ON p.id_plateform = spa.id_plateform
LEFT JOIN publication_medias pmedia
    ON pmedia.id_plateform = spa.id_plateform
    AND pmedia.publication_id = spa.id_publication
LEFT JOIN publication_media_status pms
    ON pms.publication_id COLLATE utf8mb4_0900_as_cs = spa.id_publication
WHERE spa.id_study = %s
  AND spa.flag_validated > 0
  AND pm.publication_datetime BETWEEN %s AND %s;
"""


# ---------- UTILS ----------

def generate_aws_link(pub_id: str, platform_id: int, media_type: str) -> str:
    """Generate AWS S3 media link depending on media type."""
    if media_type in IMAGE_TYPES:
        ext = "jpg"
    elif media_type in VIDEO_TYPES:
        ext = "mp4"
    else:
        ext = "bin"  # Safe fallback
    return f"https://dataclutcher-reco.s3.eu-west-3.amazonaws.com/{pub_id}_{platform_id}.{ext}"


def get_media_folder(media_type: str, study_id: str) -> str:
    """Determine appropriate folder name for media type."""
    media_type = media_type.lower() if media_type else ""
    if media_type in (t.lower() for t in IMAGE_TYPES):
        return f"images_{study_id}_10"
    elif media_type in (t.lower() for t in VIDEO_TYPES):
        return f"videos_{study_id}_10"
    else:
        return "unknown"


def ensure_folders_exist(study_id):
    """Ensure image and video folders exist for a given study."""
    for folder in [f"images_{study_id}", f"videos_{study_id}"]:
        try:
            os.makedirs(folder, exist_ok=True)
        except OSError as e:
            print(f"Warning: Could not create folder '{folder}': {e}")


def write_links_to_file(folder: str, filename: str, links: list[str]):
    """Write a list of URLs to a text file inside the specified folder."""
    path = os.path.join(folder, filename)
    try:
        with open(path, "w") as f:
            for link in links:
                f.write(link + "\n")
        print(f"{len(links)} links written to {path}")
    except IOError as e:
        print(f"Error writing file {path}: {e}")


# ---------- MAIN LOGIC ----------
def process_media(rows: list[dict], study_id: str):
    twitter_links = defaultdict(set)  
    aws_links = defaultdict(set)     
    unknown_media = 0

    for row in rows:
        platform_id = row.get("id_plateform")
        media_type = row.get("media_type")
        folder = get_media_folder(media_type, study_id)

        if folder == "unknown":
            unknown_media += 1
            print(f"Unknown media type '{media_type}' for publication {row.get('id_publication')}")
            continue

        if platform_id == TWITTER_PLATFORM_ID:
            media_url = row.get("media_url")
            if media_url:
                twitter_links[folder].add(media_url)  # use add()
            else:
                print(f" Missing media_url for Twitter publication {row.get('id_publication')} (study {study_id})")
                continue
        else:
            aws_links[folder].add(generate_aws_link(row["id_publication"], platform_id, media_type))

    # Convert sets back to lists for writing files
    for folder, links in twitter_links.items():
        if links:
            write_links_to_file(folder, "twitter_links.txt", list(links))
    for folder, links in aws_links.items():
        if links:
            write_links_to_file(folder, "aws_links.txt", list(links))

    if unknown_media > 0:
        print(f"Skipped {unknown_media} rows due to unknown media types.")


def main():
    """Main script entry point."""
    study_id = "53"
    ensure_folders_exist(study_id)

    try:
        rows = mysql_execute_select_dict(SQL_QUERY, (study_id, START_DATE, END_DATE))
    except Exception as e:
        print(f"Error querying database: {e}")
        return

    print(f"Retrieved {len(rows)} records from DB")
    process_media(rows, study_id)


if __name__ == "__main__":
    main()
