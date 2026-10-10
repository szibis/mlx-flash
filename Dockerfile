# MLX-Flash Docker
# NOTE: MLX requires Apple Silicon Metal GPU for inference.
# This container is for: testing, CI/CD, and packaging.
# For actual inference, run natively on macOS with Apple Silicon.

FROM python:3.13-slim

LABEL org.opencontainers.image.source="https://github.com/szibis/MLX-Flash"
LABEL org.opencontainers.image.description="MLX-Flash: MoE expert caching for Apple Silicon"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY pyproject.toml PYPI_README.md ./
COPY mlx_flash_compress/ ./mlx_flash_compress/
COPY tests/ ./tests/
COPY scripts/release_metadata.py scripts/local_runtime.py ./scripts/
COPY docs/ ./docs/
COPY assets/ ./assets/
COPY README.md ./

# Install Python deps (skip mlx/mlx-lm — not available on Linux)
RUN pip install --no-cache-dir \
    numpy psutil tabulate pytest lz4 zstandard safetensors huggingface-hub && \
    pip install --no-cache-dir --no-deps -e .

EXPOSE 8080

# Default: run tests that do not import MLX or require Apple hardware.
CMD ["python", "-m", "pytest", "tests/test_compression.py", "tests/test_cache_unit.py", "tests/test_release_metadata.py", "-v", "--tb=short"]
