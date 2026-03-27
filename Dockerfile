FROM python:3.10-slim

# 1. Install uv directly from their official binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 2. Copy the lockfile and pyproject.toml first (for faster builds)
COPY uv.lock pyproject.toml ./

# 3. Install dependencies without the project itself yet
RUN uv sync --frozen --no-install-project

# 4. Copy the rest of your code (including the data folder!)
COPY . .

# 5. Final sync to include your project code
RUN uv sync --frozen

EXPOSE 7860

# 6. Run the server using uv to manage the path correctly
CMD ["uv", "run", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]