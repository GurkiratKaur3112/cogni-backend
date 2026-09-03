import os
import cv2
import json
import re
import yt_dlp
import shutil



print("========================================")
print("yt-dlp:", yt_dlp.version.__version__)
print("Node:", shutil.which("node"))
print("========================================")


class YouTubeFeatureExtractor:

    def __init__(self, output_dir="data"):

        self.output_dir = output_dir

        self.dirs = {
            "videos": os.path.join(
                output_dir,
                "videos"
            ),

            "frames": os.path.join(
                output_dir,
                "frames"
            ),

            "metadata": os.path.join(
                output_dir,
                "metadata"
            )
        }

        for path in self.dirs.values():

            os.makedirs(
                path,
                exist_ok=True
            )

        # ==========================================================
        # YOUTUBE COOKIES
        # ==========================================================
        #
        # Render Secret File:
        #
        # /etc/secrets/youtube_cookies.txt
        #
        # Render Secret Files are READ-ONLY.
        #
        # Therefore we copy the cookie file to /tmp, which is
        # writable, and give the writable copy to yt-dlp.
        #
        # ==========================================================

        secret_cookie_file = os.getenv(
            "YOUTUBE_COOKIES_FILE",
            "/etc/secrets/youtube_cookies.txt"
        )

        self.cookies_file = None

        if os.path.exists(secret_cookie_file):

            writable_cookie_file = os.path.join(
                "/tmp",
                "youtube_cookies.txt"
            )

            try:

                shutil.copy2(
                    secret_cookie_file,
                    writable_cookie_file
                )

                self.cookies_file = (
                    writable_cookie_file
                )

                print(
                    "🍪 YouTube cookies loaded successfully."
                )

            except Exception as e:

                print(
                    f"⚠️ Could not copy YouTube cookies: {e}"
                )

        else:

            print(
                "⚠️ YouTube cookies file not found."
            )



    def extract_video_id(self, url):

        match = re.search(
            r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
            url
        )

        return (
            match.group(1)
            if match
            else None
        )


    def download_video_and_metadata(
        self,
        url,
        video_id
    ):

        print(
            f"\n[1/2] Fetching video: "
            f"{video_id}"
        )

        video_path = os.path.join(
            self.dirs["videos"],
            f"{video_id}.mp4"
        )

        metadata_path = os.path.join(
            self.dirs["metadata"],
            f"{video_id}.json"
        )

        ydl_opts = {

            # Prefer low-resolution video because this
            # is being used for scene/context analysis.
            "format":
                "best[height<=360][ext=mp4]"
                "/bestvideo[height<=360][ext=mp4]+bestaudio"
                "/best[height<=480][ext=mp4]"
                "/worst[ext=mp4]"
                "/worst",

            "outtmpl":
                video_path,

            "noplaylist":
                True,

            "quiet":
                False,

            "no_warnings":
                False,

            "socket_timeout":
                20,

            "retries":
                5,

            "fragment_retries":
                5,

            "file_access_retries":
                3,

            "http_headers": {

                "User-Agent":
                    (
                        "Mozilla/5.0 "
                        "(Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 "
                        "(KHTML, like Gecko) "
                        "Chrome/120.0.0.0 "
                        "Safari/537.36"
                    ),

                "Accept-Language":
                    "en-US,en;q=0.9"
            },
            
            "js_runtimes": {
                "node": {
                    "path": "/usr/bin/node"
                }
            },

            # Don't force Android/Web clients.
            # Let the current yt-dlp extractor decide.
            "extractor_args": {
                "youtube": {}
            }
        }


        # ==========================================================
        # USE WRITABLE COOKIE COPY
        # ==========================================================

        if self.cookies_file:

            ydl_opts["cookiefile"] = (
                self.cookies_file
            )

        with yt_dlp.YoutubeDL(
            ydl_opts
        ) as ydl:

            info = ydl.extract_info(
                url,
                download=True
            )

            metadata = {

                "id":
                    info.get("id"),

                "title":
                    info.get("title"),

                "description":
                    info.get("description"),

                "tags":
                    info.get("tags", []),

                "categories":
                    info.get("categories", []),

                "view_count":
                    info.get("view_count"),

                "like_count":
                    info.get("like_count"),

                "duration":
                    info.get("duration"),

                "channel":
                    info.get("uploader")
            }

            with open(
                metadata_path,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    metadata,
                    f,
                    indent=4,
                    ensure_ascii=False
                )

        print(
            f"✅ Video saved: "
            f"{video_path}"
        )

        print(
            f"✅ Metadata saved: "
            f"{metadata_path}"
        )

        return video_path


    def extract_frames(
        self,
        video_path,
        video_id,
        interval_seconds=5
    ):

        print(
            f"\n[2/2] Extracting frames "
            f"every {interval_seconds} seconds..."
        )

        frame_dir = os.path.join(
            self.dirs["frames"],
            video_id
        )

        os.makedirs(
            frame_dir,
            exist_ok=True
        )

        cap = cv2.VideoCapture(
            video_path
        )

        if not cap.isOpened():

            print(
                "❌ Could not open video."
            )

            return

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if not fps or fps <= 0:

            print(
                "❌ Could not determine FPS."
            )

            cap.release()

            return

        frame_interval = max(
            1,
            int(
                fps * interval_seconds
            )
        )

        count = 0
        saved_count = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            if count % frame_interval == 0:

                timestamp = int(
                    count / fps
                )

                frame_filename = os.path.join(
                    frame_dir,
                    f"frame_{timestamp}s.jpg"
                )

                success = cv2.imwrite(
                    frame_filename,
                    frame
                )

                if success:
                    saved_count += 1

            count += 1

        cap.release()

        print(
            f"✅ Extracted "
            f"{saved_count} frames."
        )

        print(
            f"📁 Frames: "
            f"{frame_dir}"
        )


if __name__ == "__main__":

    youtube_url = input(
        "Enter YouTube URL: "
    ).strip()

    extractor = (
        YouTubeFeatureExtractor()
    )

    video_id = (
        extractor.extract_video_id(
            youtube_url
        )
    )

    if not video_id:

        print(
            "❌ Invalid YouTube URL."
        )

    else:

        video_path = (
            extractor.download_video_and_metadata(
                youtube_url,
                video_id
            )
        )

        extractor.extract_frames(
            video_path,
            video_id,
            interval_seconds=5
        )
