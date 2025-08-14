# Dockerfile for MedGemma API

# --- Stage 1: Build Stage ---
# Use a full Python image to build dependencies, which may have system requirements.
FROM python:3.10-slim as builder

# Set the working directory
WORKDIR /app

# Install system dependencies that might be needed by Python packages
# For example, if a library needed C compilers, you'd install them here.
# RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Copy the requirements file
COPY requirements.txt .

# Create a virtual environment and install dependencies
# This isolates dependencies and keeps the final image clean.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt


# --- Stage 2: Final Stage ---
# Use a slim base image for the final container to reduce size.
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy the application code
COPY medgemma_api_server.py .

# Set the PATH to include the virtual environment's binaries
ENV PATH="/opt/venv/bin:$PATH"

# Set the environment variable for the model path inside the container.
# The user will mount their local model directory to this path.
ENV MODEL_PATH="/models"

# Expose the port the application runs on
EXPOSE 8000

# Command to run the Uvicorn server
# This is the standard way to run FastAPI in production.
# --host 0.0.0.0 makes it accessible from outside the container.
CMD ["uvicorn", "medgemma_api_server:app", "--host", "0.0.0.0", "--port", "8000"]
