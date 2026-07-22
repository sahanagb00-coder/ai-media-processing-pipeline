import os
import uuid
import hashlib
import json
import logging
import threading
import time

from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import imagehash

from PIL import Image, ExifTags

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer
from sqlalchemy.orm import declarative_base, sessionmaker


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

DATABASE_URL = "sqlite:///./media_pipeline.db"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# DATABASE
# ============================================================

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()


class ImageJob(Base):
    __tablename__ = "image_jobs"

    id = Column(String, primary_key=True)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)

    status = Column(
        String,
        default="pending",
        nullable=False,
    )

    result = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
    )


class ImageHash(Base):
    __tablename__ = "image_hashes"

    image_id = Column(
        String,
        primary_key=True,
    )

    sha256 = Column(
        String,
        nullable=False,
        index=True,
    )

    phash = Column(
        String,
        nullable=False,
    )


Base.metadata.create_all(engine)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Intelligent Media Processing Pipeline",
    description=(
        "AI-powered image quality analysis, "
        "duplicate detection, metadata analysis, "
        "screenshot detection, and image integrity checks."
    ),
    version="1.0.0",
)


# ============================================================
# STATIC FRONTEND
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse("static/index.html")


# ============================================================
# SHA256 HASH
# ============================================================

def calculate_sha256(file_path: str):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for chunk in iter(
            lambda: file.read(8192),
            b"",
        ):

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# PERCEPTUAL HASH
# ============================================================

def calculate_phash(file_path: str):

    with Image.open(file_path) as image:

        return str(
            imagehash.phash(image)
        )


# ============================================================
# BLUR DETECTION
# ============================================================

def detect_blur(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    variance = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    ).var()

    threshold = 100

    return {
        "detected": bool(
            variance < threshold
        ),

        "laplacian_variance": round(
            float(variance),
            2,
        ),

        "threshold": threshold,
    }


# ============================================================
# BRIGHTNESS ANALYSIS
# ============================================================

def analyze_brightness(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    brightness = float(
        np.mean(gray)
    )

    if brightness < 50:

        issue = "very_low_light"

    elif brightness < 90:

        issue = "low_light"

    elif brightness > 230:

        issue = "overexposed"

    else:

        issue = "normal"

    return {
        "issue": issue,

        "mean_brightness": round(
            brightness,
            2,
        ),
    }


# ============================================================
# IMAGE DIMENSION ANALYSIS
# ============================================================

def analyze_dimensions(image):

    height, width = image.shape[:2]

    valid = (
        width >= 640
        and
        height >= 480
    )

    return {
        "width": width,

        "height": height,

        "valid": valid,

        "issue": (
            None
            if valid
            else
            "low_resolution"
        ),
    }


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def detect_duplicate(
    job_id,
    sha256,
    phash,
):

    db = SessionLocal()

    try:

        # Exact duplicate detection

        exact = db.query(
            ImageHash
        ).filter(

            ImageHash.sha256 == sha256,

            ImageHash.image_id != job_id,

        ).first()


        if exact:

            db.merge(
                ImageHash(
                    image_id=job_id,

                    sha256=sha256,

                    phash=phash,
                )
            )

            db.commit()

            return {
                "is_duplicate": True,

                "type": "exact",

                "matched_processing_id":
                    exact.image_id,
            }


        # Perceptual duplicate detection

        current_hash = imagehash.hex_to_hash(
            phash
        )

        existing_images = db.query(
            ImageHash
        ).filter(

            ImageHash.image_id != job_id,

        ).all()


        for item in existing_images:

            try:

                existing_hash = imagehash.hex_to_hash(
                    item.phash
                )

                distance = (
                    current_hash
                    -
                    existing_hash
                )


                if distance <= 5:

                    db.merge(
                        ImageHash(
                            image_id=job_id,

                            sha256=sha256,

                            phash=phash,
                        )
                    )

                    db.commit()

                    return {
                        "is_duplicate": True,

                        "type": "perceptual",

                        "matched_processing_id":
                            item.image_id,

                        "hash_distance":
                            int(distance),
                    }


            except Exception:

                continue


        # Store new image hash

        db.merge(
            ImageHash(
                image_id=job_id,

                sha256=sha256,

                phash=phash,
            )
        )

        db.commit()


        return {
            "is_duplicate": False,

            "type": None,

            "matched_processing_id": None,
        }


    finally:

        db.close()


# ============================================================
# VEHICLE NUMBER DETECTION
# ============================================================

def extract_vehicle_number(file_path):

    return {
        "detected": False,

        "valid_format": False,

        "text": None,

        "source": "OCR disabled",

        "reason": (
            "OCR is disabled to keep "
            "Render memory usage low."
        ),
    }


# ============================================================
# SCREENSHOT DETECTION
# ============================================================

def analyze_screenshot(image):

    height, width = image.shape[:2]

    ratio = width / height

    signals = []

    score = 0.0


    if ratio > 1.7 or ratio < 0.6:

        score += 0.3

        signals.append(
            "unusual_aspect_ratio"
        )


    common_widths = [
        1080,
        1170,
        1284,
        1440,
    ]

    if width in common_widths:

        score += 0.3

        signals.append(
            "common_screenshot_width"
        )


    common_heights = [
        1920,
        2532,
        2778,
        2560,
    ]

    if height in common_heights:

        score += 0.3

        signals.append(
            "common_screenshot_height"
        )


    return {
        "likelihood": round(
            min(score, 1.0),
            2,
        ),

        "signals": signals,
    }


# ============================================================
# METADATA ANALYSIS
# ============================================================

def analyze_metadata(file_path):

    metadata = {}

    try:

        with Image.open(file_path) as image:

            exif = image.getexif()


            for tag_id, value in exif.items():

                tag_name = ExifTags.TAGS.get(
                    tag_id,
                    str(tag_id),
                )

                metadata[tag_name] = str(
                    value
                )


        software = metadata.get(
            "Software"
        )


        return {
            "has_exif": bool(
                metadata
            ),

            "metadata_count": len(
                metadata
            ),

            "software": software,

            "metadata": metadata,
        }


    except Exception as error:

        return {
            "has_exif": False,

            "metadata_count": 0,

            "software": None,

            "metadata": {},

            "error": str(error),
        }


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(
    file_path,
    job_id,
):

    image = cv2.imread(
        file_path
    )


    if image is None:

        raise ValueError(
            "Unable to read image"
        )


    sha256 = calculate_sha256(
        file_path
    )


    phash = calculate_phash(
        file_path
    )


    results = {}


    # 1. Blur detection

    results["blur"] = detect_blur(
        image
    )


    # 2. Brightness analysis

    results["brightness"] = analyze_brightness(
        image
    )


    # 3. Dimension validation

    results["dimensions"] = analyze_dimensions(
        image
    )


    # 4. File hash

    results["file_hash"] = sha256


    # 5. Perceptual hash

    results["perceptual_hash"] = phash


    # 6. Duplicate detection

    results["duplicate"] = detect_duplicate(
        job_id,

        sha256,

        phash,
    )


    # 7. Vehicle number detection

    results["vehicle_number"] = extract_vehicle_number(
        file_path
    )


    # 8. Screenshot detection

    results["screenshot_analysis"] = analyze_screenshot(
        image
    )


    # 9. Metadata analysis

    results["metadata_analysis"] = analyze_metadata(
        file_path
    )


    # ========================================================
    # ISSUE SUMMARY
    # ========================================================

    issues = []


    if results["blur"].get(
        "detected",
        False,
    ):

        issues.append(
            "blurry_image"
        )


    if results["brightness"].get(
        "issue"
    ) in [

        "low_light",

        "very_low_light",

    ]:

        issues.append(
            "low_light"
        )


    if results["brightness"].get(
        "issue"
    ) == "overexposed":

        issues.append(
            "overexposed_image"
        )


    if not results["dimensions"].get(
        "valid",
        False,
    ):

        issues.append(
            "low_resolution"
        )


    if results["duplicate"].get(
        "is_duplicate",
        False,
    ):

        issues.append(
            "duplicate_image"
        )


    if results["screenshot_analysis"].get(
        "likelihood",
        0,
    ) >= 0.5:

        issues.append(
            "possible_screenshot"
        )


    results["summary"] = {

        "issues_detected": issues,

        "has_issues": bool(
            issues
        ),

        "total_checks": 8,

        "overall_status": (

            "failed"

            if issues

            else

            "passed"
        ),
    }


    return results


# ============================================================
# ASYNCHRONOUS IMAGE PROCESSING
# ============================================================

def process_image(
    job_id,
    max_retries=3,
):

    for attempt in range(
        1,

        max_retries + 1,
    ):

        db = SessionLocal()

        job = None


        try:

            job = db.query(
                ImageJob
            ).filter(

                ImageJob.id == job_id,

            ).first()


            if not job:

                return


            job.status = "processing"

            job.updated_at = datetime.utcnow()

            db.commit()


            results = analyze_image(
                job.file_path,

                job_id,
            )


            job.result = json.dumps(
                results
            )

            job.status = "completed"

            job.error_message = None

            job.updated_at = datetime.utcnow()

            db.commit()


            logger.info(
                "Image processing completed: %s",

                job_id,
            )

            return


        except Exception as error:

            logger.exception(
                "Image processing failed"
            )


            if attempt == max_retries:

                if job:

                    job.status = "failed"

                    job.error_message = str(
                        error
                    )

                    job.updated_at = datetime.utcnow()

                    db.commit()

            else:

                time.sleep(
                    2 ** attempt
                )


        finally:

            db.close()


# ============================================================
# UPLOAD API
# ============================================================

@app.post(
    "/api/v1/images"
)
async def upload_image(
    file: UploadFile = File(...),
):

    if file.content_type not in (
        ALLOWED_CONTENT_TYPES
    ):

        raise HTTPException(
            status_code=400,

            detail=(
                "Only JPEG, PNG and WEBP "
                "images are supported"
            ),
        )


    content = await file.read()


    if not content:

        raise HTTPException(
            status_code=400,

            detail="Uploaded file is empty",
        )


    if len(content) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,

            detail="File size exceeds 10 MB",
        )


    processing_id = str(
        uuid.uuid4()
    )


    extension = Path(
        file.filename
    ).suffix.lower()


    file_path = (
        UPLOAD_DIR
        /
        f"{processing_id}{extension}"
    )


    with open(
        file_path,

        "wb",
    ) as output:

        output.write(
            content
        )


    db = SessionLocal()


    try:

        job = ImageJob(

            id=processing_id,

            original_filename=file.filename,

            file_path=str(
                file_path
            ),

            content_type=file.content_type,

            file_size=len(
                content
            ),

            status="pending",
        )


        db.add(
            job
        )

        db.commit()


    finally:

        db.close()


    thread = threading.Thread(

        target=process_image,

        args=(processing_id,),

        daemon=True,
    )


    thread.start()


    return {

        "processing_id": processing_id,

        "status": "pending",

        "message": (
            "Image uploaded and queued "
            "for asynchronous processing"
        ),
    }


# ============================================================
# STATUS API
# ============================================================

@app.get(
    "/api/v1/images/{processing_id}/status"
)
def get_status(
    processing_id: str,
):

    db = SessionLocal()


    try:

        job = db.query(
            ImageJob
        ).filter(

            ImageJob.id == processing_id,

        ).first()


        if not job:

            raise HTTPException(
                status_code=404,

                detail="Processing ID not found",
            )


        return {

            "processing_id": processing_id,

            "status": job.status,

            "error": job.error_message,

            "created_at": job.created_at,

            "updated_at": job.updated_at,
        }


    finally:

        db.close()


# ============================================================
# RESULTS API
# ============================================================

@app.get(
    "/api/v1/images/{processing_id}/results"
)
def get_results(
    processing_id: str,
):

    db = SessionLocal()


    try:

        job = db.query(
            ImageJob
        ).filter(

            ImageJob.id == processing_id,

        ).first()


        if not job:

            raise HTTPException(
                status_code=404,

                detail="Processing ID not found",
            )


        if job.status == "failed":

            return {

                "processing_id": processing_id,

                "status": "failed",

                "analysis": None,

                "error": job.error_message,
            }


        if job.status != "completed":

            return {

                "processing_id": processing_id,

                "status": job.status,

                "analysis": None,
            }


        return {

            "processing_id": processing_id,

            "status": "completed",

            "analysis": json.loads(
                job.result
            ),
        }


    finally:

        db.close()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/api/v1/health"
)
def health():

    return {

        "service": (
            "Intelligent Media "
            "Processing Pipeline"
        ),

        "status": "running",

        "version": "1.0.0",
    }
