FROM python:3.10-slim

# 1. Install uv directly from their official binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. Set environment variables for non-interactive installs
ENV UV_COMPILE_BYTECODE=1 UV_HTTP_TIMEOUT=300

WORKDIR /app

# 3. Copy lockfiles first
COPY uv.lock pyproject.toml ./

# 4. Install dependencies (Frozen ensures exact versions from your lockfile)
RUN uv sync --frozen --no-install-project --no-dev

# 5. Copy the rest of your code
COPY . .

# 6. Final sync
RUN uv sync --frozen --no-dev

EXPOSE 7860

# 7. Run using 0.0.0.0 so the HF Space can route external traffic to the container
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]