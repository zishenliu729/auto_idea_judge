# Use an official Python runtime as the base image
FROM python:3.10-slim

# Install system-level dependencies, including git
# WebIDE Docker bootstrap (2026-07-07): local imported base images may already
# contain git/build tools but not apt metadata; skip apt only when apt-get is absent.
RUN if python3 --version 2>/dev/null | grep -q "Python 3.12"; then \
        echo "WebIDE local base detected; skipping apt bootstrap"; \
    elif command -v apt-get >/dev/null 2>&1; then \
        apt-get update && apt-get install -y \
            build-essential \
            git \
        && apt-get clean && rm -rf /var/lib/apt/lists/*; \
    else \
        echo "apt-get not found; assuming local WebIDE base already has required tools"; \
    fi

# Set the working directory inside the container
WORKDIR /dgm

# Copy the entire repository into the container
COPY . .

# Install Python dependencies
# WebIDE Docker bootstrap (2026-07-07): imported Ubuntu/Python bases enforce
# PEP 668, so allow system-site installation inside this disposable DGM image.
RUN pip install --break-system-packages --no-cache-dir -r requirements.txt

# Keep the container running by default
CMD ["tail", "-f", "/dev/null"]