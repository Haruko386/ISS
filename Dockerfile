FROM python:3.14-slim AS builder

WORKDIR /build

COPY pyproject.toml README.md ./
COPY iss ./iss

RUN python -m pip install --no-cache-dir --upgrade build \
    && python -m build --wheel


FROM python:3.14-slim AS runtime

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/dist/*.whl /tmp/iss/

RUN python -m pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" "torch>=2.2,<3" \
    && python -m pip install --no-cache-dir /tmp/iss/*.whl \
    && rm -rf /tmp/iss

RUN useradd --create-home --uid 10001 iss

USER iss
WORKDIR /workspace

ENTRYPOINT ["iss"]
CMD ["doctor"]
