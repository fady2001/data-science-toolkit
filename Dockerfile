FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PORT=8000
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
WORKDIR /app

# Cache dependencies
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT []

CMD sh -c "uvicorn src.inference:app --host 0.0.0.0 --port $PORT"

EXPOSE ${PORT}
