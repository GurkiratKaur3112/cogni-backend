import re
import threading
import uuid
from pathlib import Path

from youtube_extractor import YouTubeFeatureExtractor
from scene_detector import (
    select_scene_frames
)

from vision_analyzer import (
    analyze_frame_and_find_ads,
    analyze_frames_batch
)

from pinecone_ads import search_ads

CHUNK_DURATION_SECONDS = 30
MAX_BATCH_FRAMES = 3

# 3 representative frames per chunk.
# Example for 0-30s:
# 5s, 15s, 25s
FRAMES_PER_CHUNK = 3


# --------------------------------------------------
# In-memory job storage
# --------------------------------------------------

jobs = {}

jobs_lock = threading.Lock()

def create_job(youtube_url: str):

    job_id = str(uuid.uuid4())

    job = {
        "job_id": job_id,
        "youtube_url": youtube_url,
        "status": "queued",
        "progress": 0,
        "total_chunks": 0,
        "completed_chunks": 0,
        "placements": [],
        "error": None
    }

    with jobs_lock:
        jobs[job_id] = job

    return job_id


def get_job(job_id: str):

    with jobs_lock:
        job = jobs.get(job_id)

        if job is None:
            return None

        # Return a copy so callers don't mutate
        # the internal object.
        return {
            **job,
            "placements": list(
                job["placements"]
            )
        }


def update_job(job_id: str, **updates):

    with jobs_lock:

        if job_id not in jobs:
            return

        jobs[job_id].update(updates)


def add_placement(job_id: str, placement: dict):
    """
    Add an advertisement placement.

    If the new advertisement is the same as the
    previous advertisement, don't create another
    placement. Instead, keep the existing placement
    open.

    If the advertisement changes, close the previous
    placement and create a new one.
    """

    with jobs_lock:

        if job_id not in jobs:
            return False

        placements = jobs[job_id]["placements"]

        # ------------------------------------------
        # No previous placement
        # ------------------------------------------

        if not placements:

            placements.append(placement)

            return True

        previous = placements[-1]

        previous_ad = previous.get(
            "ad",
            {}
        )

        current_ad = placement.get(
            "ad",
            {}
        )

        previous_id = previous_ad.get(
            "id"
        )

        current_id = current_ad.get(
            "id"
        )

        # ------------------------------------------
        # Same advertisement
        # ------------------------------------------

        if (
            previous_id
            and current_id
            and previous_id == current_id
        ):

            print(
                f"🔁 Same ad detected: "
                f"{current_ad.get('brand', '')}"
            )

            print(
                "   Extending existing placement."
            )

            return False

        # ------------------------------------------
        # Advertisement changed
        # ------------------------------------------

        previous["end_time"] = (
            placement["timestamp"]
        )

        previous["end_time_formatted"] = (
            format_timestamp(
                placement["timestamp"]
            )
        )

        # ------------------------------------------
        # Create new placement
        # ------------------------------------------

        placement["start_time"] = (
            placement["timestamp"]
        )

        placement["start_time_formatted"] = (
            format_timestamp(
                placement["timestamp"]
            )
        )

        placement["end_time"] = None

        placement["end_time_formatted"] = None

        placements.append(
            placement
        )

        return True
# --------------------------------------------------
# Timestamp helpers
# --------------------------------------------------

def get_timestamp_from_filename(filename):

    match = re.search(
        r"frame_(\d+)s",
        filename
    )

    if match:
        return int(match.group(1))

    return None


def format_timestamp(seconds):

    minutes = seconds // 60
    remaining_seconds = seconds % 60

    return (
        f"{minutes:02d}:"
        f"{remaining_seconds:02d}"
    )


# --------------------------------------------------
# Extract representative frames
# --------------------------------------------------

def get_chunk_frames(
    frame_files,
    chunk_start,
    chunk_end
):

    chunk_frames = []

    for frame_path in frame_files:

        timestamp = get_timestamp_from_filename(
            frame_path.name
        )

        if timestamp is None:
            continue

        if (
            chunk_start
            <= timestamp
            < chunk_end
        ):

            chunk_frames.append(
                frame_path
            )

    if not chunk_frames:
        return []

    # ------------------------------------------
    # Select representative frames
    # ------------------------------------------

    if len(chunk_frames) <= FRAMES_PER_CHUNK:
        return chunk_frames

    step = (
        len(chunk_frames)
        / FRAMES_PER_CHUNK
    )

    selected = []

    for i in range(FRAMES_PER_CHUNK):

        index = int(
            i * step
        )

        if index >= len(chunk_frames):
            index = len(chunk_frames) - 1

        selected.append(
            chunk_frames[index]
        )

    return selected


# --------------------------------------------------
# Analyze one chunk
# --------------------------------------------------

def process_chunk(
    job_id,
    frame_files,
    chunk_start,
    chunk_end
):

    print(f"\n{'=' * 70}")
    print(f"📦 JOB {job_id}")
    print(f"🎬 Processing chunk {chunk_start}s → {chunk_end}s")
    print(f"{'=' * 70}")

    # ------------------------------------------
    # Get frames belonging to this chunk
    # ------------------------------------------

    chunk_frames = get_chunk_frames(
        frame_files,
        chunk_start,
        chunk_end
    )

    # ------------------------------------------
    # Cheap scene-change detection
    # ------------------------------------------

    selected_frames = select_scene_frames(
        chunk_frames,
        threshold=0.25,
        min_gap_seconds=10
    )

    print(f"\n🖼️ Chunk frames: {len(chunk_frames)}")
    print(f"⭐ Selected frames: {len(selected_frames)}")

    if not selected_frames:
        print("⚠️ No useful frames found in this chunk.")
        return

    # ------------------------------------------
    # Batch selected frames
    # ------------------------------------------

    for batch_start in range(
        0,
        len(selected_frames),
        MAX_BATCH_FRAMES
    ):

        # ✅ This was wrongly indented
        batch = selected_frames[
            batch_start:
            batch_start + MAX_BATCH_FRAMES
        ]

        print(
            f"\n🚀 BATCH "
            f"{batch_start // MAX_BATCH_FRAMES + 1}"
        )

        print(f"Frames in batch: {len(batch)}")

        try:

            # ----------------------------------
            # ONE QWEN REQUEST
            # ----------------------------------

            batch_results = analyze_frames_batch(batch)

            # ----------------------------------
            # Pinecone search for each frame
            # ----------------------------------

            for result in batch_results:

                frame_index = result["frame_index"]

                if frame_index >= len(batch):
                    continue

                frame_path = batch[frame_index]

                timestamp = get_timestamp_from_filename(
                    frame_path.name
                )

                scene = result.get("scene", "")
                search_query = result.get("search_query", "")

                print(f"\n🖼️ Frame: {frame_path.name}")
                print(f"⏱️ Timestamp: {timestamp}s")
                print(f"🧠 Scene: {scene}")
                print(f"🔎 Query: {search_query}")

                # ----------------------------------
                # Pinecone Search
                # ----------------------------------

                results = search_ads(
                    search_query,
                    top_k=3
                )

                if results is None:
                    print(
                        "⚠️ No sufficiently relevant "
                        "advertisement found."
                    )
                    continue

                hits = results.result.hits

                if not hits:
                    print("⚠️ No advertisement found.")
                    continue

                best_ad = hits[0]
                fields = best_ad.fields

                placement = {
                    "timestamp": timestamp,
                    "timestamp_formatted": format_timestamp(timestamp),
                    "scene": scene,
                    "search_query": search_query,
                    "ad": {
                        "id": best_ad.id,
                        "brand": fields.get("brand", ""),
                        "title": fields.get("title", ""),
                        "category": fields.get("category", ""),
                        "description": fields.get("description", "")
                    },
                    "score": float(best_ad.score)
                }

                add_placement(job_id, placement)

                print(f"📢 AD FOUND: {fields.get('brand', '')}")
                print(f"   Timestamp: {timestamp}s")
                print(f"   Score: {best_ad.score:.4f}")

        except Exception as exc:

            print(f"❌ Batch processing failed: {exc}")
            continue

# --------------------------------------------------
# Complete background analysis
# --------------------------------------------------

def run_analysis_job(job_id):

    job = get_job(job_id)

    if not job:
        return

    youtube_url = job["youtube_url"]

    try:

        update_job(
            job_id,
            status="downloading",
            progress=0
        )

        extractor = YouTubeFeatureExtractor()

        video_id = extractor.extract_video_id(
            youtube_url
        )

        if not video_id:

            raise ValueError(
                "Invalid YouTube URL."
            )

        print(
            f"\n🎬 Starting job {job_id}"
        )

        print(
            f"Video ID: {video_id}"
        )

        # --------------------------------------
        # Download video
        # --------------------------------------

        video_path = (
            extractor.download_video_and_metadata(
                youtube_url,
                video_id
            )
        )

        update_job(
            job_id,
            status="extracting_frames",
            video_id=video_id
        )

        # --------------------------------------
        # Extract frames
        # --------------------------------------

        extractor.extract_frames(
            video_path,
            video_id,
            interval_seconds=5
        )

        frame_directory = (
            Path(
                extractor.dirs["frames"]
            )
            / video_id
        )

        frame_files = list(
            frame_directory.glob(
                "frame_*s.jpg"
            )
        )

        frame_files.sort(
            key=lambda path:
            get_timestamp_from_filename(
                path.name
            ) or 0
        )

        if not frame_files:

            raise RuntimeError(
                "No frames were extracted."
            )

        # --------------------------------------
        # Determine video duration
        # --------------------------------------

        timestamps = [
            get_timestamp_from_filename(
                path.name
            )
            for path in frame_files
        ]

        timestamps = [
            timestamp
            for timestamp in timestamps
            if timestamp is not None
        ]

        video_duration = (
            max(timestamps) + 5
        )

        total_chunks = (
            video_duration
            + CHUNK_DURATION_SECONDS
            - 1
        ) // CHUNK_DURATION_SECONDS

        update_job(
            job_id,
            status="processing",
            total_chunks=total_chunks,
            completed_chunks=0,
            progress=0
        )

        # --------------------------------------
        # Process chunks sequentially
        # --------------------------------------

        for chunk_number in range(
            total_chunks
        ):

            chunk_start = (
                chunk_number
                * CHUNK_DURATION_SECONDS
            )

            chunk_end = min(
                chunk_start
                + CHUNK_DURATION_SECONDS,
                video_duration
            )

            process_chunk(
                job_id,
                frame_files,
                chunk_start,
                chunk_end
            )

            completed_chunks = (
                chunk_number + 1
            )

            progress = int(
                (
                    completed_chunks
                    / total_chunks
                )
                * 100
            )

            update_job(
                job_id,
                completed_chunks=completed_chunks,
                progress=progress
            )

            print(
                f"\n📊 JOB PROGRESS: "
                f"{progress}%"
            )

        # --------------------------------------
        # Completed
        # --------------------------------------

# --------------------------------------
# Close final advertisement placement
# --------------------------------------

        with jobs_lock:

            if job_id in jobs:

                placements = jobs[job_id]["placements"]

                if placements:

                    last_placement = placements[-1]

                    if last_placement.get(
                        "end_time"
                    ) is None:

                        last_placement["end_time"] = (
                            video_duration
                        )

                        last_placement[
                            "end_time_formatted"
                        ] = format_timestamp(
                            video_duration
                        )
        update_job(
            job_id,
            status="completed",
            progress=100
        )

        print(
            f"\n🎉 JOB {job_id} COMPLETED"
        )

    except Exception as exc:

        print(
            f"\n❌ JOB {job_id} FAILED:"
        )

        print(exc)

        update_job(
            job_id,
            status="failed",
            error=str(exc)
        )


# --------------------------------------------------
# Start background job
# --------------------------------------------------

def start_background_job(
    youtube_url
):

    job_id = create_job(
        youtube_url
    )

    thread = threading.Thread(
        target=run_analysis_job,
        args=(job_id,),
        daemon=True
    )

    thread.start()

    return job_id