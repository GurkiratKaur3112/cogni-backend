import base64
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from pinecone_ads import search_ads

from huggingface_hub import InferenceClient


load_dotenv()


# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# VISION_MODEL = os.getenv(
#     "GROQ_VISION_MODEL",
#     "qwen/qwen3.6-27b"
# )

TOP_K_ADS = int(
    os.getenv("TOP_K_ADS", "3")
)


# if not GROQ_API_KEY:
#     raise RuntimeError(
#         "GROQ_API_KEY is missing from your .env file."
#     )


# client = Groq(api_key=GROQ_API_KEY)

HF_TOKEN = os.getenv("HF_TOKEN")

HF_MODEL = os.getenv(
    "HF_MODEL",
    "Qwen/Qwen3-VL-4B-Instruct"
)

if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN is missing from .env"
    )

hf_client = InferenceClient(
    provider="featherless-ai",
    api_key=HF_TOKEN
)

def image_to_data_url(image_path: str) -> str:

    path = Path(image_path)

    extension = path.suffix.lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }

    mime_type = mime_types.get(
        extension,
        "image/jpeg"
    )

    with open(
        image_path,
        "rb"
    ) as image_file:

        encoded = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return (
        f"data:{mime_type};base64,"
        f"{encoded}"
    )


def encode_image(image_path):
    """
    Convert local image to base64 data URL.
    """

    with open(image_path, "rb") as image_file:

        encoded = base64.b64encode(
            image_file.read()
        ).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"

# def analyze_frame(image_path):
#     """
#     Analyze a frame and generate a clean semantic search query.
#     """

#     image_url = encode_image(image_path)

#     prompt = """
# Analyze this image for contextual advertising.

# We need to understand what type of advertisement would naturally
# fit the content shown in the image.

# Return EXACTLY these two lines:

# SCENE: <short description of the scene>

# QUERY: <5 to 12 important words for semantic advertisement search>

# Rules for QUERY:
# - Describe the context, activity, products, lifestyle or environment.
# - Do NOT explain your reasoning.
# - Do NOT mention that you are an AI.
# - Do NOT write paragraphs.
# - Do NOT use JSON.
# - Do NOT use markdown.
# - Do NOT include <think> tags.
# - Do NOT identify a brand unless the brand itself is the main contextual subject.
# - Focus on concepts useful for matching advertisements.

# Example:

# SCENE: Person running outdoors

# QUERY: running fitness sports shoes athletic clothing exercise

# Another example:

# SCENE: Friends eating pizza together

# QUERY: pizza food restaurant dinner friends food delivery

# Another example:

# SCENE: Person drinking coffee in a cafe

# QUERY: coffee cafe beverages morning relaxation lifestyle

# Return ONLY the two lines.
# """

#     response = client.chat.completions.create(
#         model=VISION_MODEL,

#         messages=[
#             {
#                 "role": "system",
#                 "content": (
#                     "You are a contextual advertising analyzer. "
#                     "Follow the requested output format exactly."
#                 )
#             },
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": prompt
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                             "url": image_url
#                         }
#                     }
#                 ]
#             }
#         ],

#         temperature=0,

#         max_completion_tokens=200
#     )

#     raw_output = response.choices[0].message.content.strip()

#     print("\nDEBUG GROQ OUTPUT:")
#     print(raw_output)

#     # Remove accidental think blocks
#     raw_output = re.sub(
#         r"<think>.*?</think>",
#         "",
#         raw_output,
#         flags=re.DOTALL | re.IGNORECASE
#     ).strip()

#     # Extract QUERY line
#     query_match = re.search(
#         r"QUERY\s*:\s*(.+)",
#         raw_output,
#         flags=re.IGNORECASE
#     )

#     scene_match = re.search(
#         r"SCENE\s*:\s*(.+)",
#         raw_output,
#         flags=re.IGNORECASE
#     )

#     if query_match:
#         query = query_match.group(1).strip()
#     else:
#         # Fallback if model doesn't follow format
#         query = raw_output

#     if scene_match:
#         scene = scene_match.group(1).strip()
#     else:
#         scene = ""

#     # Clean accidental quotes
#     query = query.strip("\"'")

#     return {
#         "scene": scene,
#         "search_query": query
#     }
    
def analyze_frame(image_path: str):

    image_data_url = image_to_data_url(
        image_path
    )

    prompt = """
Analyze this video frame for contextual advertising.

Return exactly two lines:

SCENE: <short description of the scene>

QUERY: <5 to 12 important words for semantic advertisement search>

Rules:

- Identify the main scene.
- Identify important objects.
- Identify the activity.
- Identify the environment.
- Identify products or lifestyle context.
- Create a semantic query useful for advertisement matching.
- Do NOT identify a brand unless the brand is clearly visible.
- Do NOT explain your reasoning.
- Do NOT return JSON.
- Do NOT use markdown.
- Do NOT write paragraphs.
- Do NOT include <think> tags.

Example:

SCENE: Person running outdoors

QUERY: running fitness sports shoes athletic clothing exercise

Example:

SCENE: Friends eating pizza

QUERY: pizza restaurant food delivery dinner friends

Example:

SCENE: Person drinking coffee in cafe

QUERY: coffee cafe beverages drinks relaxation lifestyle
"""

    try:

        response = hf_client.chat.completions.create(

            model=HF_MODEL,

            messages=[
                {
                    "role": "user",

                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data_url
                            }
                        }
                    ]
                }
            ],

            max_tokens=200,

            temperature=0
        )

        raw_output = (
            response.choices[0]
            .message
            .content
            .strip()
        )

        print("\nDEBUG HUGGING FACE OUTPUT:")
        print(raw_output)

        # --------------------------------------
        # Remove thinking blocks
        # --------------------------------------

        raw_output = re.sub(
            r"<think>.*?</think>",
            "",
            raw_output,
            flags=re.DOTALL |
                  re.IGNORECASE
        ).strip()

        # --------------------------------------
        # Extract scene
        # --------------------------------------

        scene_match = re.search(
            r"SCENE\s*:\s*(.+)",
            raw_output,
            flags=re.IGNORECASE
        )

        # --------------------------------------
        # Extract query
        # --------------------------------------

        query_match = re.search(
            r"QUERY\s*:\s*(.+)",
            raw_output,
            flags=re.IGNORECASE
        )

        scene = (
            scene_match.group(1).strip()
            if scene_match
            else ""
        )

        if query_match:

            query = (
                query_match
                .group(1)
                .strip()
            )

        else:

            query = raw_output

        query = query.strip(
            "\"'"
        )

        return {
            "scene": scene,
            "search_query": query
        }

    except Exception as exc:

        print(
            f"❌ Hugging Face vision failed: "
            f"{exc}"
        )

        raise    

def analyze_frame_and_find_ads(image_path):

    print("\n" + "=" * 70)

    print(
        f"🖼️ Processing frame: "
        f"{Path(image_path).name}"
    )

    context = analyze_frame(image_path)

    scene = context.get(
        "scene",
        ""
    )

    search_query = context.get(
        "search_query",
        ""
    )

    print("\n🧠 VISION ANALYSIS")
    print("-" * 70)

    print(
        f"Scene: {scene}"
    )

    print(
        f"Search Query: {search_query}"
    )

    print("\n🔎 SEARCHING AD DATABASE...")

    results = search_ads(
        search_query,
        top_k=TOP_K_ADS
    )

    print("\n📢 RELEVANT ADVERTISEMENTS")
    print("-" * 70)

    hits = results.result.hits

    if not hits:

        print(
            "No relevant advertisements found."
        )

        return {
            "scene": scene,
            "search_query": search_query,
            "ads": []
        }

    for rank, hit in enumerate(
        hits,
        start=1
    ):

        fields = hit.fields

        print(
            f"\n{rank}. "
            f"{fields.get('brand', 'Unknown')}"
        )

        print(
            f"   Title: "
            f"{fields.get('title', '')}"
        )

        print(
            f"   Category: "
            f"{fields.get('category', '')}"
        )

        print(
            f"   Similarity: "
            f"{hit.score:.4f}"
        )

        print(
            f"   Description: "
            f"{fields.get('description', '')}"
        )

    return {
        "scene": scene,
        "search_query": search_query,
        "ads": hits
    }
    
def analyze_frames_batch(image_paths):
    """
    Analyze multiple selected frames in ONE Qwen vision request.

    Returns:
    [
        {
            "frame_index": 0,
            "scene": "...",
            "search_query": "..."
        },
        ...
    ]
    """

    if not image_paths:
        return []

    image_contents = []

    for index, image_path in enumerate(image_paths):

        image_data_url = image_to_data_url(
            str(image_path)
        )

        image_contents.append({
            "type": "text",
            "text": f"FRAME_INDEX: {index}"
        })

        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": image_data_url
            }
        })

    prompt = """
Analyze all the video frames provided below.

Each image has a FRAME_INDEX.

For EVERY frame, return exactly:

FRAME_INDEX: <index>
SCENE: <short scene description>
QUERY: <5 to 12 words useful for advertisement search>

Rules:

- Analyze every frame.
- Keep each frame's result separate.
- Do not skip frames.
- Identify the main scene, objects, activity,
  environment and relevant lifestyle/product context.
- QUERY must contain useful semantic advertising keywords.
- Do NOT identify brands unless clearly visible.
- Do NOT explain your reasoning.
- Do NOT return JSON.
- Do NOT use markdown.
- Do NOT include <think> tags.

Example:

FRAME_INDEX: 0
SCENE: Person cooking in a kitchen
QUERY: cooking kitchen appliances cookware groceries

FRAME_INDEX: 1
SCENE: Person eating dinner
QUERY: food restaurant groceries food delivery dining

FRAME_INDEX: 2
SCENE: Person exercising outdoors
QUERY: fitness running sports shoes gym exercise
"""

    # Put the instruction first, then all images.
    content = [
        {
            "type": "text",
            "text": prompt
        }
    ]

    content.extend(image_contents)

    try:

        response = hf_client.chat.completions.create(
            model=HF_MODEL,

            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],

            max_tokens=500,

            temperature=0
        )

        raw_output = (
            response.choices[0]
            .message
            .content
            .strip()
        )

        print("\nDEBUG BATCH QWEN OUTPUT")
        print("=" * 70)
        print(raw_output)
        print("=" * 70)

        # Remove thinking blocks if returned.
        raw_output = re.sub(
            r"<think>.*?</think>",
            "",
            raw_output,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()

        # ------------------------------------------
        # Parse each frame
        # ------------------------------------------

        results = []

        pattern = re.compile(
            r"FRAME_INDEX\s*:\s*(\d+)"
            r"\s*"
            r"SCENE\s*:\s*(.+?)"
            r"\s*"
            r"QUERY\s*:\s*(.+?)(?="
            r"\s*FRAME_INDEX\s*:|$)",
            flags=re.IGNORECASE | re.DOTALL
        )

        matches = pattern.findall(
            raw_output
        )

        for match in matches:

            frame_index = int(
                match[0]
            )

            scene = (
                match[1]
                .strip()
                .replace("\n", " ")
            )

            query = (
                match[2]
                .strip()
                .replace("\n", " ")
            )

            results.append({
                "frame_index": frame_index,
                "scene": scene,
                "search_query": query
            })

        # ------------------------------------------
        # Sort by frame index
        # ------------------------------------------

        results.sort(
            key=lambda x:
            x["frame_index"]
        )

        return results

    except Exception as exc:

        print(
            f"❌ Batch Qwen vision failed: "
            f"{exc}"
        )

        raise