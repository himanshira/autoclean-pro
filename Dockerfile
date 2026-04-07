FROM python:3.10-slim

# 1. Install uv from the official binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 2. Environment variables
#    UV_COMPILE_BYTECODE  — pre-compile .pyc at build time (faster startup)
#    UV_HTTP_TIMEOUT      — avoid timeouts on slow HF networks
#    UV_PROJECT_ENVIRONMENT — tell uv exactly where to create/find the venv
#    PATH                 — required so 'uv run' resolves the venv from ANY
#                           working directory (docker exec, HF eval runner, etc.)
#                           Without this, uv run falls back to system python
#                           when CWD has no pyproject.toml in its parent chain
ENV UV_COMPILE_BYTECODE=1 \
    UV_HTTP_TIMEOUT=300 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 3. Copy lockfiles first — maximises Docker layer cache reuse
COPY uv.lock pyproject.toml ./

# 4. Install all dependencies into /app/.venv
RUN uv sync --frozen --no-install-project --no-dev

# 5. Copy the rest of the project (source code + data/ CSVs)
COPY . .

# 6. Final sync — installs the project package itself
RUN uv sync --frozen --no-dev

EXPOSE 7860

# 7. uv run activates /app/.venv — required by hackathon spec
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]