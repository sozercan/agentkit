# syntax=docker/dockerfile:1
# Minimal Phase A0 Foundry hosted Responses brokered conformance image.
# Build from the repository root so runtimes/common is available in context.
FROM python:3.12-slim

ENV PORT=8088 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/agentkit
COPY runtimes/common /tmp/agentkit-serve-common
RUN pip install --no-cache-dir /tmp/agentkit-serve-common \
    && rm -rf /tmp/agentkit-serve-common

EXPOSE 8088
ENTRYPOINT ["agentkit-foundry-conformance", "--host", "0.0.0.0", "--port", "8088"]
