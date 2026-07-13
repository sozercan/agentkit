# syntax=docker/dockerfile:1
# Minimal production AgentKit Foundry brokered-only image with static brokeredTools.
# Build from the repository root so runtimes/common is available in context.
FROM python:3.12-slim

ENV PORT=8088 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/agentkit
COPY runtimes/common /tmp/agentkit-serve-common
RUN pip install --no-cache-dir /tmp/agentkit-serve-common \
    && rm -rf /tmp/agentkit-serve-common
COPY test/foundry-brokered-agentkit/agent.yaml /agent/agent.yaml
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin agentkit \
    && chown -R 1000:1000 /agent /opt/agentkit

EXPOSE 8088
USER 1000
ENTRYPOINT ["agentkit-foundry-brokered", "--host", "0.0.0.0", "--port", "8088", "--config", "/agent/agent.yaml"]
