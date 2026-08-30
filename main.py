import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl

from analysis_service import (
    start_background_job,
    get_job
)


app = FastAPI(
    title="Context-Aware Advertising API",
    description=(
        "AI-powered contextual "
        "advertisement recommendation backend."
    ),
    version="2.0.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        os.getenv("FRONTEND_URL")
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)

# --------------------------------------------------
# Request model
# --------------------------------------------------

class AnalyzeRequest(BaseModel):

    youtube_url: HttpUrl


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "context-aware-advertising"
    }


# --------------------------------------------------
# Start analysis
# --------------------------------------------------

@app.post("/api/analyze")
def analyze_video(
    request: AnalyzeRequest
):

    try:

        job_id = start_background_job(
            str(request.youtube_url)
        )

        return {
            "success": True,
            "job_id": job_id,
            "status": "queued"
        }

    except Exception as exc:

        print(
            f"❌ Could not start job: {exc}"
        )

        raise HTTPException(
            status_code=500,
            detail="Could not start analysis."
        )


# --------------------------------------------------
# Get analysis status
# --------------------------------------------------

@app.get("/api/analyze/{job_id}")
def get_analysis_status(
    job_id: str
):

    job = get_job(job_id)

    if job is None:

        raise HTTPException(
            status_code=404,
            detail="Analysis job not found."
        )

    return {
        "success": True,
        "data": job
    }