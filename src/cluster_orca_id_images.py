import argparse
import bisect
import re
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

try:
    import requests
except ImportError:
    requests = None

# Detection Settings
TIMESPAN_SECONDS = 1.5
MAX_BOX_MOVEMENT_PER_STEP = 300.0
MAX_BOX_SIZE_CHANGE_RATIO = 2

# User settings
ID_DIR = "/Users/paul/Desktop/NOS/CreateFinDataset/test_outputs/quality_75/Orca_ID/2022-09-07_Andøya_RichardKaroliussen_IDs"
ORIGINAL_DIR = "/Users/paul/Desktop/NOS/CreateFinDataset/test_outputs/quality_75/Orca_ID/2022-09-07_Andøya_RichardKaroliussen_All_pictures"
OUTPUT_DIR = "/Users/paul/Desktop/NOS/CreateFinDataset/output/orca_id_clusters"
BASE_URL = "http://127.0.0.1:8000/api/inference"
DETECT_PATH = "/fin-detect"
VERIFY_SSL = True
BOX_COLOR = "lime"
DISCARDED_BOX_COLOR = "gray"
BOX_WIDTH = 24
REQUEST_TIMEOUT_SECONDS = 60
DRAW_BOXES = True
YOLO_CLASS_ID = 0
SUMMARY_THUMBNAIL_WIDTH = 420
SUMMARY_THUMBNAIL_HEIGHT = 280
SUMMARY_COLUMNS = 4


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".png",
}

DATE_IN_NAME_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
CAMERA_NUMBER_RE = re.compile(r"(?:IMG|6B5A|DSC|DSCF|DSCN|_)(?:_)?(\d{3,8})", re.IGNORECASE)
EXIF_DATETIME_TAGS = (36867, 36868, 306)  # DateTimeOriginal, DateTimeDigitized, DateTime
EXIF_SUBSECOND_TAGS = (37521, 37522, 37520)


def cluster_id_images(
    id_dir,
    original_dir,
    output_dir,
    timespan_seconds,
    use_mtime_fallback,
    draw_boxes=DRAW_BOXES,
    base_url=BASE_URL,
    detect_path=DETECT_PATH,
    verify_ssl=VERIFY_SSL,
    box_color=BOX_COLOR,
    discarded_box_color=DISCARDED_BOX_COLOR,
    box_width=BOX_WIDTH,
    max_box_movement_per_step=MAX_BOX_MOVEMENT_PER_STEP,
    max_box_size_change_ratio=MAX_BOX_SIZE_CHANGE_RATIO,
    request_timeout=REQUEST_TIMEOUT_SECONDS,
):
    id_path = Path(id_dir).expanduser().resolve()
    original_path = Path(original_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    timespan = timedelta(seconds=validate_timespan(timespan_seconds))

    if not id_path.is_dir():
        raise NotADirectoryError(f"ID directory does not exist: {id_path}")
    if not original_path.is_dir():
        raise NotADirectoryError(f"Original directory does not exist: {original_path}")

    started_at = datetime.now()
    start_time = time.monotonic()
    id_images = get_image_files(id_path)
    original_images = get_image_files(original_path)

    print(f"Total ID images: {len(id_images)}")
    print(f"Started: {format_datetime(started_at)}")
    print(f"Original images queued for clustering analysis: {len(original_images)}")

    original_records = []
    missing_original_times = []
    for original_image in original_images:
        captured_at = get_image_time(original_image, use_mtime_fallback)
        if captured_at is None:
            missing_original_times.append(original_image)
            continue
        original_records.append((captured_at, original_image))

    original_records.sort(key=lambda item: (item[0], str(item[1])))
    original_times = [record[0] for record in original_records]
    original_order = {path.resolve(): index for index, (_, path) in enumerate(original_records)}
    original_by_key = build_original_key_index(original_images)

    clustered_counts = []
    copied_cluster_paths = set()
    id_images_with_time = 0
    id_images_without_time = []
    id_images_without_original = []
    summary_groups = {}
    summary_slides_created = 0
    summary_slides_failed = 0

    if not id_images:
        print("No ID images found.")
        return

    print_progress_bar(0, len(id_images), start_time)
    for index, id_image in enumerate(id_images, start=1):
        id_time = get_image_time(id_image, use_mtime_fallback)
        if id_time is None:
            id_images_without_time.append(id_image)
            clustered_counts.append(0)
            print_progress_bar(index, len(id_images), start_time)
            continue

        id_images_with_time += 1
        cluster_images = find_cluster(original_records, original_times, id_time, timespan)
        id_name = extract_id_name(id_image)
        id_output_dir = output_path / safe_path_name(id_name)
        id_output_dir.mkdir(parents=True, exist_ok=True)
        summary_group = get_summary_group(summary_groups, id_name, id_output_dir)

        original_id_image = find_matching_original(id_image, original_by_key)
        if original_id_image is not None:
            cluster_images = sorted(set(cluster_images + [original_id_image]))
        else:
            id_images_without_original.append(id_image)

        try:
            manual_output_path = copy_unique(id_image, id_output_dir / f"manual__{id_image.name}")
        except OSError as exc:
            print(f"\nCould not copy manual ID image {id_image}: {exc}")
            print_progress_bar(index, len(id_images), start_time)
            continue
        summary_group["manual"].append(
            {
                "path": manual_output_path,
                "source_path": original_id_image or id_image,
                "source_index": get_source_index(original_id_image or id_image, original_order),
            }
        )

        for cluster_image in cluster_images:
            try:
                target_path = copy_unique(cluster_image, id_output_dir / cluster_image.name)
            except OSError as exc:
                print(f"\nCould not copy clustered image {cluster_image}: {exc}")
                continue
            copied_cluster_paths.add(target_path.resolve())
            source_key = cluster_image.resolve()
            is_manual_original = (
                original_id_image is not None
                and source_key == original_id_image.resolve()
            )
            if source_key not in summary_group["additional_sources"]:
                summary_group["additional"].append(
                    {
                        "path": target_path,
                        "source_path": cluster_image,
                        "source_index": get_source_index(cluster_image, original_order),
                        "is_manual_original": is_manual_original,
                    }
                )
                summary_group["additional_sources"].add(source_key)
            elif is_manual_original:
                mark_manual_original(summary_group["additional"], source_key)

        clustered_counts.append(len(cluster_images))
        print_progress_bar(index, len(id_images), start_time)

    if draw_boxes and summary_groups:
        print()
        print("Creating summary slides")
        summary_start_time = time.monotonic()
        summary_items = list(summary_groups.values())
        print_progress_bar(0, len(summary_items), summary_start_time)
        box_cache = {}
        for index, summary_group in enumerate(summary_items, start=1):
            try:
                summary_group["summary_stats"] = create_summary_slide(
                    summary_group,
                    base_url,
                    detect_path,
                    verify_ssl,
                    box_color,
                    discarded_box_color,
                    box_width,
                    max_box_movement_per_step,
                    max_box_size_change_ratio,
                    request_timeout,
                    box_cache,
                )
                summary_slides_created += 1
            except Exception as exc:
                print(f"\nCould not create summary slide for {summary_group['id_name']}: {exc}")
                summary_slides_failed += 1
            print_progress_bar(index, len(summary_items), summary_start_time)

    print()
    print(f"Finished: {format_datetime(datetime.now())}")
    print("Final report")
    print(f"Total number of images analysed for clustering: {len(original_images)}")
    print(f"Original images with usable timestamps: {len(original_records)}")
    print(f"ID images with usable timestamps: {id_images_with_time}")
    print(f"Images used for clustering: {len(copied_cluster_paths)}")
    print(f"Average number of clustered images: {average(clustered_counts):.2f}")
    if draw_boxes:
        print(f"Summary slides created: {summary_slides_created}")
        print(f"Summary slides failed: {summary_slides_failed}")

    if missing_original_times:
        print(f"Original images skipped because no timestamp was available: {len(missing_original_times)}")
    if id_images_without_time:
        print(f"ID images skipped because no timestamp was available: {len(id_images_without_time)}")
    if id_images_without_original:
        print(f"ID images where the exact original image was not found by filename: {len(id_images_without_original)}")

    summary_path = write_overall_summary(
        output_path,
        {
            "started_at": started_at,
            "finished_at": datetime.now(),
            "id_dir": id_path,
            "original_dir": original_path,
            "output_dir": output_path,
            "timespan_seconds": timespan_seconds,
            "use_mtime_fallback": use_mtime_fallback,
            "draw_boxes": draw_boxes,
            "base_url": base_url,
            "detect_path": detect_path,
            "max_box_movement_per_step": max_box_movement_per_step,
            "max_box_size_change_ratio": max_box_size_change_ratio,
            "total_id_images": len(id_images),
            "total_original_images": len(original_images),
            "original_images_with_timestamps": len(original_records),
            "id_images_with_timestamps": id_images_with_time,
            "images_used_for_clustering": len(copied_cluster_paths),
            "average_clustered_images": average(clustered_counts),
            "summary_slides_created": summary_slides_created,
            "summary_slides_failed": summary_slides_failed,
            "original_images_without_timestamps": len(missing_original_times),
            "id_images_without_timestamps": len(id_images_without_time),
            "id_images_without_original": len(id_images_without_original),
            "summary_groups": summary_groups,
        },
    )
    print(f"Overall summary written to: {summary_path}")


def get_image_files(directory):
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_image_time(path, use_mtime_fallback):
    exif_time = get_exif_time(path)
    if exif_time is not None:
        return exif_time
    if use_mtime_fallback:
        return datetime.fromtimestamp(path.stat().st_mtime)
    return None


def get_exif_time(path):
    if Image is None:
        return None

    try:
        with Image.open(path) as image:
            exif = image.getexif()
    except Exception:
        return None

    if not exif:
        return None

    for datetime_tag, subsecond_tag in zip(EXIF_DATETIME_TAGS, EXIF_SUBSECOND_TAGS):
        value = exif.get(datetime_tag)
        if not value:
            continue
        parsed = parse_exif_datetime(value)
        if parsed is None:
            continue
        subsecond = parse_exif_subsecond(exif.get(subsecond_tag))
        if subsecond:
            parsed = parsed.replace(microsecond=subsecond)
        return parsed

    return None


def parse_exif_datetime(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    value = str(value).strip().replace("\x00", "")

    for date_format in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, date_format)
        except ValueError:
            pass
    return None


def parse_exif_subsecond(value):
    if value is None:
        return 0
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    digits = "".join(char for char in str(value) if char.isdigit())
    if not digits:
        return 0
    return int((digits[:6]).ljust(6, "0"))


def find_cluster(original_records, original_times, id_time, timespan):
    start = id_time - timespan
    end = id_time + timespan
    start_index = bisect.bisect_left(original_times, start)
    end_index = bisect.bisect_right(original_times, end)
    return [path for _, path in original_records[start_index:end_index]]


def build_original_key_index(original_images):
    index = {}
    for image in original_images:
        key = original_key(image)
        index.setdefault(key, image)
    return index


def find_matching_original(id_image, original_by_key):
    key = original_key(id_image)
    if key in original_by_key:
        return original_by_key[key]

    id_stem = id_image.stem
    for original_key_value, original_path in original_by_key.items():
        if id_stem.endswith(original_key_value):
            return original_path

    camera_number = extract_camera_number(id_stem)
    if camera_number is None:
        return None

    for original_key_value, original_path in original_by_key.items():
        if extract_camera_number(original_key_value) == camera_number:
            return original_path

    return None


def original_key(path):
    stem = path.stem
    match = DATE_IN_NAME_RE.search(stem)
    if match:
        return stem[match.start():]
    return stem


def extract_id_name(path):
    stem = path.stem
    match = DATE_IN_NAME_RE.search(stem)
    if match and match.start() > 0:
        return stem[: match.start()].rstrip("_")

    camera_number = extract_camera_number(stem)
    if camera_number is not None:
        return stem[: stem.rfind(camera_number)].rstrip("_")

    return stem


def extract_camera_number(value):
    matches = CAMERA_NUMBER_RE.findall(value)
    if not matches:
        return None
    return matches[-1]


def get_summary_group(summary_groups, id_name, output_dir):
    if id_name not in summary_groups:
        summary_groups[id_name] = {
            "id_name": id_name,
            "output_dir": output_dir,
            "manual": [],
            "additional": [],
            "additional_sources": set(),
        }
    return summary_groups[id_name]


def get_source_index(source_path, original_order):
    if source_path is None:
        return None
    return original_order.get(source_path.resolve())


def mark_manual_original(additional_records, source_key):
    for record in additional_records:
        if record["source_path"].resolve() == source_key:
            record["is_manual_original"] = True
            return


def copy_unique(source_path, target_path):
    target_path = get_unique_target_path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    return target_path


def save_approved_crop(image_path, box, crop_dir, source_label):
    crop_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_size = image.size
        left, top, right, bottom = clamp_box(box, image_size)
        if right <= left or bottom <= top:
            return None
        crop = image.crop((left, top, right, bottom))

    target_path = crop_dir / f"{source_label}__{crop_file_stem(image_path)}_fin.jpg"
    target_path = get_unique_target_path(target_path)
    crop.save(target_path, quality=95)
    save_yolo_label_for_crop(target_path, (left, top, right, bottom), image_size)
    return target_path


def save_yolo_label_for_crop(crop_path, source_box, source_image_size):
    label_path = yolo_label_path_for_crop(crop_path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    x_center, y_center, width, height = box_to_yolo(source_box, source_image_size)
    label_path.write_text(
        f"{YOLO_CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n",
        encoding="utf-8",
    )
    return label_path


def yolo_label_path_for_crop(crop_path):
    return crop_path.parent / "yolo_labels" / f"{crop_path.stem}.txt"


def box_to_yolo(box, image_size):
    image_width, image_height = image_size
    left, top, right, bottom = box
    box_width = right - left
    box_height = bottom - top
    x_center = left + box_width / 2
    y_center = top + box_height / 2
    return (
        x_center / image_width,
        y_center / image_height,
        box_width / image_width,
        box_height / image_height,
    )


def write_overall_summary(output_path, report):
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "overall_summary.txt"
    lines = [
        "Orca ID clustering summary",
        f"Started: {format_datetime(report['started_at'])}",
        f"Finished: {format_datetime(report['finished_at'])}",
        "",
        "Settings",
        f"ID directory: {report['id_dir']}",
        f"Original directory: {report['original_dir']}",
        f"Output directory: {report['output_dir']}",
        f"Timespan seconds: {report['timespan_seconds']}",
        f"Use modification-time fallback: {report['use_mtime_fallback']}",
        f"Draw summary boxes: {report['draw_boxes']}",
        f"Detection API: {report['base_url'].rstrip('/')}/{report['detect_path'].lstrip('/')}",
        f"Max box movement per step: {report['max_box_movement_per_step']}",
        f"Max box size change ratio: {report['max_box_size_change_ratio']}",
        "",
        "Totals",
        f"Total ID images: {report['total_id_images']}",
        f"Total original images analysed for clustering: {report['total_original_images']}",
        f"Original images with usable timestamps: {report['original_images_with_timestamps']}",
        f"ID images with usable timestamps: {report['id_images_with_timestamps']}",
        f"Images used for clustering: {report['images_used_for_clustering']}",
        f"Average number of clustered images: {report['average_clustered_images']:.2f}",
        f"Summary slides created: {report['summary_slides_created']}",
        f"Summary slides failed: {report['summary_slides_failed']}",
        f"Original images without timestamps: {report['original_images_without_timestamps']}",
        f"ID images without timestamps: {report['id_images_without_timestamps']}",
        f"ID images without exact original filename match: {report['id_images_without_original']}",
        "",
        "Per ID",
    ]

    for id_name in sorted(report["summary_groups"]):
        group = report["summary_groups"][id_name]
        stats = group.get("summary_stats", {})
        lines.extend(
            [
                f"{id_name}",
                f"  Output folder: {group['output_dir']}",
                f"  Manual images: {len(group['manual'])}",
                f"  Additional clustered images: {len(group['additional'])}",
                f"  Manual kept: {stats.get('manual_kept', 0)}",
                f"  Manual discarded: {stats.get('manual_discarded', 0)}",
                f"  Additional kept: {stats.get('additional_kept', 0)}",
                f"  Additional discarded: {stats.get('additional_discarded', 0)}",
                f"  Discarded by movement: {stats.get('discarded_movement', 0)}",
                f"  Discarded by size: {stats.get('discarded_size', 0)}",
                f"  Discarded by movement+size: {stats.get('discarded_movement_size', 0)}",
                f"  Discarded no box/ref: {stats.get('discarded_no_box_or_ref', 0)}",
                f"  Original time-series images highlighted: {stats.get('original_highlighted', 0)}",
                f"  Approved crops: {stats.get('approved_crops', 0)}",
                f"  YOLO label files: {stats.get('yolo_labels', 0)}",
                f"  Summary slide: {stats.get('summary_slide', '')}",
            ]
        )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def crop_file_stem(image_path):
    stem = image_path.stem
    for prefix in ("manual__", "additional__"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def create_summary_slide(
    summary_group,
    base_url,
    detect_path,
    verify_ssl,
    box_color,
    discarded_box_color,
    box_width,
    max_box_movement_per_step,
    max_box_size_change_ratio,
    request_timeout,
    box_cache,
):
    if requests is None:
        raise RuntimeError("requests is not installed")
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is not installed")

    crop_dir = summary_group["output_dir"] / "cropped"
    manual_references = build_manual_references(
        summary_group["manual"],
        base_url,
        detect_path,
        verify_ssl,
        request_timeout,
        box_cache,
    )

    manual_records = sorted_records_by_time(summary_group["manual"])
    additional_records = sorted_records_by_time(summary_group["additional"])

    manual_tiles = [
        create_manual_summary_tile(
            record,
            "IDed",
            box_color,
            discarded_box_color,
            box_width,
            manual_references,
            crop_dir,
        )
        for record in manual_records
    ]
    additional_tiles = [
        create_candidate_summary_tile(
            record,
            "Added",
            base_url,
            detect_path,
            verify_ssl,
            box_color,
            discarded_box_color,
            box_width,
            max_box_movement_per_step,
            max_box_size_change_ratio,
            request_timeout,
            box_cache,
            manual_references,
            crop_dir,
        )
        for record in additional_records
    ]

    tiles = manual_tiles + additional_tiles
    if not tiles:
        raise RuntimeError("no images available for summary")

    slide = compose_summary_slide(summary_group["id_name"], manual_tiles, additional_tiles)
    target_path = summary_group["output_dir"] / f"summary__{safe_path_name(summary_group['id_name'])}.jpg"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    slide.save(target_path, quality=95)
    return summarize_tiles(manual_tiles, additional_tiles, target_path)


def build_manual_references(
    manual_records,
    base_url,
    detect_path,
    verify_ssl,
    request_timeout,
    box_cache,
):
    references = []
    for record in manual_records:
        boxes = get_cached_detection_boxes(
            record["path"],
            base_url,
            detect_path,
            verify_ssl,
            request_timeout,
            box_cache,
        )
        reference_box = largest_box(boxes)
        if reference_box is None:
            continue
        references.append(
            {
                "path": record["path"],
                "source_index": record["source_index"],
                "box": reference_box,
            }
        )
    return references


def sorted_records_by_time(records):
    return sorted(
        records,
        key=lambda record: (
            record["source_index"] is None,
            record["source_index"] if record["source_index"] is not None else 0,
            str(record["source_path"]),
        ),
    )


def create_manual_summary_tile(
    record,
    label,
    box_color,
    discarded_box_color,
    box_width,
    manual_references,
    crop_dir,
):
    image_path = record["path"]
    reference = next((item for item in manual_references if item["path"] == image_path), None)
    crop_path = None
    if reference:
        crop_path = save_approved_crop(image_path, reference["box"], crop_dir, "manual")
    image = render_classified_thumbnail(
        image_path,
        [(reference["box"], True)] if reference else [],
        [],
        box_color,
        discarded_box_color,
        box_width,
    )
    status = "KEEP" if reference else "NOT"
    return {
        "image": image,
        "label": label,
        "status": status,
        "filename": image_path.name,
        "crop_path": crop_path,
        "is_manual_original": False,
    }


def create_candidate_summary_tile(
    record,
    label,
    base_url,
    detect_path,
    verify_ssl,
    box_color,
    discarded_box_color,
    box_width,
    max_box_movement_per_step,
    max_box_size_change_ratio,
    request_timeout,
    box_cache,
    manual_references,
    crop_dir,
):
    image_path = record["path"]
    boxes = get_cached_detection_boxes(
        image_path,
        base_url,
        detect_path,
        verify_ssl,
        request_timeout,
        box_cache,
    )
    kept_box, discard_reason = classify_candidate_box(
        boxes,
        record["source_index"],
        manual_references,
        max_box_movement_per_step,
        max_box_size_change_ratio,
    )
    discarded_boxes = [box for box in boxes if box != kept_box]
    crop_path = None
    if kept_box:
        crop_path = save_approved_crop(image_path, kept_box, crop_dir, "additional")
    image = render_classified_thumbnail(
        image_path,
        [(kept_box, True)] if kept_box else [],
        discarded_boxes,
        box_color,
        discarded_box_color,
        box_width,
    )
    status = "KEEP" if kept_box else "DISCARD"
    return {
        "image": image,
        "label": label,
        "status": status,
        "filename": image_path.name,
        "crop_path": crop_path,
        "is_manual_original": record.get("is_manual_original", False),
        "discard_reason": discard_reason,
    }


def summarize_tiles(manual_tiles, additional_tiles, summary_slide_path):
    crop_paths = [tile["crop_path"] for tile in manual_tiles + additional_tiles if tile.get("crop_path")]
    return {
        "manual_kept": count_status(manual_tiles, "KEEP"),
        "manual_discarded": count_status(manual_tiles, "DISCARD"),
        "additional_kept": count_status(additional_tiles, "KEEP"),
        "additional_discarded": count_status(additional_tiles, "DISCARD"),
        "discarded_movement": count_discard_reason(additional_tiles, "movement"),
        "discarded_size": count_discard_reason(additional_tiles, "size"),
        "discarded_movement_size": count_discard_reason(additional_tiles, "movement+size"),
        "discarded_no_box_or_ref": count_discard_reason(additional_tiles, "no box")
        + count_discard_reason(additional_tiles, "no ref"),
        "original_highlighted": sum(1 for tile in additional_tiles if tile.get("is_manual_original")),
        "approved_crops": len(crop_paths),
        "yolo_labels": sum(1 for crop_path in crop_paths if yolo_label_path_for_crop(crop_path).is_file()),
        "summary_slide": summary_slide_path,
    }


def count_status(tiles, status):
    return sum(1 for tile in tiles if tile.get("status") == status)


def count_discard_reason(tiles, reason):
    return sum(1 for tile in tiles if tile.get("discard_reason") == reason)


def render_classified_thumbnail(
    image_path,
    kept_boxes,
    discarded_boxes,
    kept_box_color,
    discarded_box_color,
    box_width,
):
    with Image.open(image_path) as image:
        full_image = image.convert("RGB")

    original_size = full_image.size
    thumbnail = full_image.copy()
    thumbnail.thumbnail((SUMMARY_THUMBNAIL_WIDTH, SUMMARY_THUMBNAIL_HEIGHT), Image.Resampling.LANCZOS)

    kept_scaled_boxes = scale_boxes([box for box, _ in kept_boxes], original_size, thumbnail.size)
    discarded_scaled_boxes = scale_boxes(discarded_boxes, original_size, thumbnail.size)
    draw_boxes_on_loaded_image(thumbnail, discarded_scaled_boxes, discarded_box_color, box_width)
    draw_boxes_on_loaded_image(thumbnail, kept_scaled_boxes, kept_box_color, box_width)
    return thumbnail


def get_cached_detection_boxes(
    image_path,
    base_url,
    detect_path,
    verify_ssl,
    request_timeout,
    box_cache,
):
    cache_key = image_path.resolve()
    if cache_key not in box_cache:
        response_content = request_fin_detection(
            image_path,
            base_url,
            detect_path,
            verify_ssl,
            request_timeout,
        )
        box_cache[cache_key] = get_detection_boxes(response_content, image_path)
    return box_cache[cache_key]


def largest_box(boxes):
    if not boxes:
        return None
    return max(boxes, key=box_area)


def box_area(box):
    left, top, right, bottom = box
    return max(0, right - left) * max(0, bottom - top)


def classify_candidate_box(
    boxes,
    candidate_source_index,
    manual_references,
    max_box_movement_per_step,
    max_box_size_change_ratio,
):
    if not boxes:
        return None, "no box"
    if not manual_references:
        return None, "no ref"

    best_match = None
    best_distance = None
    saw_movement_pass = False
    saw_size_pass = False

    for box in boxes:
        for reference in manual_references:
            if reference["source_index"] is None or candidate_source_index is None:
                step_distance = 1
            else:
                step_distance = abs(candidate_source_index - reference["source_index"])
            allowed_distance = float(max_box_movement_per_step) * max(1, step_distance)
            distance = box_center_distance(box, reference["box"])
            movement_ok = distance <= allowed_distance
            size_ok = box_size_is_allowed(box, reference["box"], max_box_size_change_ratio)
            saw_movement_pass = saw_movement_pass or movement_ok
            saw_size_pass = saw_size_pass or size_ok
            if (
                movement_ok
                and size_ok
                and (best_distance is None or distance < best_distance)
            ):
                best_match = box
                best_distance = distance

    if best_match:
        return best_match, ""
    if not saw_movement_pass and not saw_size_pass:
        return None, "movement+size"
    if not saw_movement_pass:
        return None, "movement"
    return None, "size"


def box_size_is_allowed(candidate_box, reference_box, max_box_size_change_ratio):
    reference_area = box_area(reference_box)
    candidate_area = box_area(candidate_box)
    if reference_area <= 0 or candidate_area <= 0:
        return False

    max_ratio = max(1, float(max_box_size_change_ratio))
    lower_ratio = 1 / max_ratio
    upper_ratio = max_ratio
    size_ratio = candidate_area / reference_area
    return lower_ratio <= size_ratio <= upper_ratio


def box_center_distance(box_a, box_b):
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def box_center(box):
    left, top, right, bottom = box
    return (left + right) / 2, (top + bottom) / 2


def scale_boxes(boxes, original_size, thumbnail_size):
    original_width, original_height = original_size
    thumbnail_width, thumbnail_height = thumbnail_size
    x_scale = thumbnail_width / original_width
    y_scale = thumbnail_height / original_height
    return [
        (
            left * x_scale,
            top * y_scale,
            right * x_scale,
            bottom * y_scale,
        )
        for left, top, right, bottom in boxes
    ]


def draw_boxes_on_loaded_image(image, boxes, box_color, box_width):
    draw = ImageDraw.Draw(image)
    effective_box_width = max(2, int(round(min(image.size) * 0.015)), int(box_width / 4))

    for box in boxes:
        left, top, right, bottom = clamp_box(box, image.size)
        for offset in range(effective_box_width):
            draw.rectangle(
                (left - offset, top - offset, right + offset, bottom + offset),
                outline=box_color,
            )


def compose_summary_slide(id_name, manual_tiles, additional_tiles):
    margin = 40
    gap = 24
    header_height = 80
    section_header_height = 42
    tile_label_height = 48
    tile_width = SUMMARY_THUMBNAIL_WIDTH
    tile_height = SUMMARY_THUMBNAIL_HEIGHT + tile_label_height
    columns = SUMMARY_COLUMNS

    manual_rows = rows_needed(len(manual_tiles), columns)
    additional_rows = rows_needed(len(additional_tiles), columns)
    width = margin * 2 + columns * tile_width + (columns - 1) * gap
    height = (
        margin
        + header_height
        + section_header_height
        + manual_rows * tile_height
        + max(0, manual_rows - 1) * gap
        + gap
        + section_header_height
        + additional_rows * tile_height
        + max(0, additional_rows - 1) * gap
        + margin
    )

    slide = Image.new("RGB", (width, max(height, 600)), "white")
    draw = ImageDraw.Draw(slide)
    y = margin
    draw.text((margin, y), f"ID summary: {id_name}", fill="black")
    y += header_height

    y = paste_section(slide, draw, "Manually identified image(s)", manual_tiles, margin, y, columns, gap, tile_width, tile_height)
    y += gap
    paste_section(slide, draw, "Additional clustered images", additional_tiles, margin, y, columns, gap, tile_width, tile_height)
    return slide


def paste_section(slide, draw, title, tiles, margin, y, columns, gap, tile_width, tile_height):
    draw.text((margin, y), title, fill="black")
    y += 42

    if not tiles:
        draw.text((margin, y), "None", fill="black")
        return y + tile_height

    for index, tile in enumerate(tiles):
        row = index // columns
        column = index % columns
        x = margin + column * (tile_width + gap)
        tile_y = y + row * (tile_height + gap)
        image = tile["image"]
        image_x = x + (tile_width - image.width) // 2
        slide.paste(image, (image_x, tile_y))
        if tile.get("is_manual_original"):
            draw_tile_frame(draw, image_x, tile_y, image.width, image.height, "blue", 8)
        label = f"{tile['label']} {tile['status']}: {tile['filename']}"
        if tile.get("status") == "DISCARD" and tile.get("discard_reason"):
            label = f"{tile['label']} DISCARD {tile['discard_reason']}: {tile['filename']}"
        if tile.get("is_manual_original"):
            label = f"ORIGINAL {label}"
        draw.text((x, tile_y + SUMMARY_THUMBNAIL_HEIGHT + 8), truncate_text(label, 58), fill="black")

    return y + rows_needed(len(tiles), columns) * tile_height + max(0, rows_needed(len(tiles), columns) - 1) * gap


def draw_tile_frame(draw, x, y, width, height, color, line_width):
    for offset in range(line_width):
        draw.rectangle(
            (x - offset, y - offset, x + width + offset, y + height + offset),
            outline=color,
        )


def rows_needed(item_count, columns):
    if item_count == 0:
        return 1
    return (item_count + columns - 1) // columns


def truncate_text(value, max_length):
    if len(value) <= max_length:
        return value
    return value[: max_length - 3] + "..."


def request_fin_detection(image_path, base_url, detect_path, verify_ssl, request_timeout):
    url = base_url.rstrip("/") + "/" + detect_path.lstrip("/")
    with image_path.open("rb") as image_file:
        response = requests.post(
            url,
            files={"file": image_file},
            verify=verify_ssl,
            timeout=request_timeout,
        )
    response.raise_for_status()
    return response.json()["response"]


def get_detection_boxes(response_content, image_path):
    absolute_boxes = response_content.get("absoluteBoxes", [])
    if absolute_boxes:
        return [box_from_absolute(item) for item in absolute_boxes if item is not None]

    proportion_boxes = response_content.get("proportionBoxes", [])
    if not proportion_boxes:
        return []

    with Image.open(image_path) as image:
        width, height = image.size
    return [box_from_proportion(item, width, height) for item in proportion_boxes if item is not None]


def box_from_absolute(item):
    left = float(item["x"])
    top = float(item["y"])
    right = left + float(item["w"])
    bottom = top + float(item["h"])
    return left, top, right, bottom


def box_from_proportion(item, width, height):
    box_width = float(item["w"]) * width
    box_height = float(item["h"]) * height
    center_x = float(item["x"]) * width
    center_y = float(item["y"]) * height
    left = center_x - box_width / 2
    top = center_y - box_height / 2
    right = center_x + box_width / 2
    bottom = center_y + box_height / 2
    return left, top, right, bottom


def draw_boxes_on_image(image_path, boxed_path, boxes, box_color, box_width):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    effective_box_width = max(int(box_width), int(round(min(image.size) * 0.006)))

    for box in boxes:
        left, top, right, bottom = clamp_box(box, image.size)
        for offset in range(max(1, effective_box_width)):
            draw.rectangle(
                (left - offset, top - offset, right + offset, bottom + offset),
                outline=box_color,
            )

    boxed_path = get_unique_target_path(boxed_path)
    boxed_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(boxed_path, quality=95)


def clamp_box(box, image_size):
    width, height = image_size
    left, top, right, bottom = box
    left = max(0, min(width - 1, int(round(left))))
    top = max(0, min(height - 1, int(round(top))))
    right = max(0, min(width - 1, int(round(right))))
    bottom = max(0, min(height - 1, int(round(bottom))))
    return left, top, right, bottom


def get_unique_target_path(target_path):
    if not target_path.exists():
        return target_path

    counter = 2
    while True:
        candidate = target_path.with_name(f"{target_path.stem}_{counter}{target_path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def print_progress_bar(current, total, start_time, width=40):
    if total <= 0:
        return
    completed = int(width * current / total)
    bar = "#" * completed + "-" * (width - completed)
    percent = current / total * 100
    if current == 0:
        remaining_seconds = 0
    else:
        elapsed = time.monotonic() - start_time
        average_seconds_per_file = elapsed / current
        remaining_seconds = average_seconds_per_file * (total - current)
    eta = format_duration(remaining_seconds)
    sys.stdout.write(f"\rProgress: [{bar}] {current}/{total} ({percent:5.1f}%) ETA: {eta}")
    sys.stdout.flush()


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def format_datetime(value):
    return value.strftime("%Y-%m-%d %H:%M:%S")


def average(values):
    if not values:
        return 0
    return sum(values) / len(values)


def safe_path_name(value):
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip()
    return cleaned or "unknown_id"


def validate_timespan(value):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Timespan must be a number of seconds") from exc
    if value < 0:
        raise ValueError("Timespan must be zero or greater")
    return value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cluster original JPEG images around manually identified orca ID images."
    )
    parser.add_argument("--id-dir", default=ID_DIR, help="Directory containing manually identified JPEG images.")
    parser.add_argument("--original-dir", default=ORIGINAL_DIR, help="Directory containing all original JPEG images.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Directory where ID cluster folders will be written.")
    parser.add_argument("--timespan", type=float, default=TIMESPAN_SECONDS, help="Seconds before and after each ID image to include.")
    parser.add_argument(
        "--no-mtime-fallback",
        action="store_true",
        help="Only use EXIF timestamps. By default file modification time is used when EXIF is missing.",
    )
    parser.add_argument(
        "--draw-boxes",
        action=argparse.BooleanOptionalAction,
        default=DRAW_BOXES,
        help="Call the fin-detect API and draw boxes on each ID summary slide.",
    )
    parser.add_argument("--base-url", default=BASE_URL, help="Base URL for the fin detection API.")
    parser.add_argument("--detect-path", default=DETECT_PATH, help="Detection endpoint path.")
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification for API requests.",
    )
    parser.add_argument("--box-color", default=BOX_COLOR, help="Bounding box color for kept detections.")
    parser.add_argument("--discarded-box-color", default=DISCARDED_BOX_COLOR, help="Bounding box color for discarded detections.")
    parser.add_argument("--box-width", type=int, default=BOX_WIDTH, help="Bounding box line width.")
    parser.add_argument(
        "--max-box-movement-per-step",
        type=float,
        default=MAX_BOX_MOVEMENT_PER_STEP,
        help="Maximum allowed bounding-box center movement, in pixels, per image step from the manual ID image.",
    )
    parser.add_argument(
        "--max-box-size-change-ratio",
        type=float,
        default=MAX_BOX_SIZE_CHANGE_RATIO,
        help="Maximum allowed area ratio versus the manual box. 2.0 allows 50%-200% of manual area.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=REQUEST_TIMEOUT_SECONDS,
        help="API request timeout in seconds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cluster_id_images(
        args.id_dir,
        args.original_dir,
        args.output_dir,
        args.timespan,
        use_mtime_fallback=not args.no_mtime_fallback,
        draw_boxes=args.draw_boxes,
        base_url=args.base_url,
        detect_path=args.detect_path,
        verify_ssl=not args.no_verify_ssl,
        box_color=args.box_color,
        discarded_box_color=args.discarded_box_color,
        box_width=args.box_width,
        max_box_movement_per_step=args.max_box_movement_per_step,
        max_box_size_change_ratio=args.max_box_size_change_ratio,
        request_timeout=args.request_timeout,
    )


if __name__ == "__main__":
    main()
