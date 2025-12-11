# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Set environment variables for uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# Add the virtual environment's bin to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# --- Install Docker CLI ---
# We need the Docker CLI to launch worker containers from the API server.
RUN apt-get update && apt-get install -y curl && \
    curl -fL "https://download.docker.com/linux/static/stable/x86_64/docker-26.1.4.tgz" | tar -xz --strip-components=1 -C /usr/local/bin docker/docker && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies into a virtual environment.
# This layer is cached and only rebuilds if the lockfile or pyproject.toml changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --locked --no-dev

# Copy the rest of the project source code
COPY . .

# The default command to run when the container starts.
# This will start the API server from the modular API package.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
