# ============================================================
# 1. Base Image
# ============================================================

FROM python:3.11-slim


# ============================================================
# 2. Working Directory
# ============================================================

WORKDIR /app


# ============================================================
# 3. Copy Dependency File
# ============================================================

COPY requirements.txt .


# ============================================================
# 4. Install Dependencies
# ============================================================

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# 5. Copy Project
# ============================================================

COPY . .


# ============================================================
# 6. Copy Serving Model
# ============================================================

# Local structure:
#
# src/serving/model/
# ├── MLmodel
# ├── model.xgb
# ├── feature_columns.txt
# ├── preprocessing.pkl
# ├── conda.yaml
# ├── python_env.yaml
# ├── requirements.txt
# └── metadata/
#
# Docker structure:
#
# /app/model/
# ├── MLmodel
# ├── model.xgb
# ├── feature_columns.txt
# ├── preprocessing.pkl
# └── ...
#
# inference.py checks /app/model first.

COPY src/serving/model /app/model


# ============================================================
# 7. Environment Configuration
# ============================================================

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app


# ============================================================
# 8. Expose FastAPI Port
# ============================================================

EXPOSE 8000


# ============================================================
# 9. Start Application
# ============================================================

CMD ["python", "-m", "uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]