# Use official Python runtime as base image
FROM python:3.11-slim

# Set maintainer label
LABEL maintainer="ml-app"
LABEL description="Iris Classifier ML Application"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first (for better layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code to container
COPY src/ ./src/
COPY pytest.ini .
COPY .flake8 .

# Create directories for models and outputs
RUN mkdir -p models

# Expose port (for future prediction API)
EXPOSE 8000

# Add healthcheck (optional, for future use)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command: Run training
CMD ["python", "src/train.py"]
