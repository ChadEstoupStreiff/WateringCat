FROM dhi.io/uv:0-debian13-dev

WORKDIR /app

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/app/.venv

ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project

COPY src/ src/

ENTRYPOINT ["uv"]
CMD ["run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]