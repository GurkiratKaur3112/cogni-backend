
IMAGE_PATH = "data/frames/QPsxAUZLQ3c/frame_50s.jpg"

from vision_analyzer import analyze_frame



result = analyze_frame(
    IMAGE_PATH
)

print("\n==============================")
print("FINAL RESULT")
print("==============================")

print(
    "Scene:",
    result["scene"]
)

print(
    "Query:",
    result["search_query"]
)