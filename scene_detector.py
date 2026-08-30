import cv2
import numpy as np


def calculate_frame_difference(
    frame1,
    frame2
):
    """
    Calculate visual difference between
    two frames using grayscale histogram
    comparison.
    """

    gray1 = cv2.cvtColor(
        frame1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        frame2,
        cv2.COLOR_BGR2GRAY
    )

    # Resize for faster processing
    gray1 = cv2.resize(
        gray1,
        (320, 180)
    )

    gray2 = cv2.resize(
        gray2,
        (320, 180)
    )

    # Calculate absolute pixel difference
    difference = cv2.absdiff(
        gray1,
        gray2
    )

    # Normalize difference to 0-1
    score = (
        np.mean(difference)
        / 255.0
    )

    return float(score)


def select_scene_frames(
    frame_files,
    threshold=0.20,
    min_gap_seconds=10
):

    if not frame_files:
        return []

    selected = []

    reference_frame = None

    last_selected_timestamp = -999

    for frame_path in frame_files:

        frame = cv2.imread(
            str(frame_path)
        )

        if frame is None:
            continue

        timestamp = get_timestamp(
            frame_path
        )

        # --------------------------------------
        # First frame
        # --------------------------------------

        if reference_frame is None:

            selected.append(
                frame_path
            )

            reference_frame = frame

            last_selected_timestamp = (
                timestamp
            )

            continue

        # --------------------------------------
        # Compare against last selected frame
        # --------------------------------------

        difference = (
            calculate_frame_difference(
                reference_frame,
                frame
            )
        )

        print(
            f"Frame {timestamp}s "
            f"difference: "
            f"{difference:.3f}"
        )

        # --------------------------------------
        # Scene change
        # --------------------------------------

        scene_changed = (
            difference >= threshold
        )

        enough_time_passed = (
            timestamp
            - last_selected_timestamp
            >= min_gap_seconds
        )

        if (
            scene_changed
            and enough_time_passed
        ):

            print(
                f"⭐ Scene change detected "
                f"at {timestamp}s"
            )

            selected.append(
                frame_path
            )

            reference_frame = frame

            last_selected_timestamp = (
                timestamp
            )

    return selected


def get_timestamp(frame_path):

    name = frame_path.name

    # frame_25s.jpg
    timestamp = (
        name
        .replace("frame_", "")
        .replace("s.jpg", "")
    )

    return int(timestamp)