# syntax=docker/dockerfile:1
# Wrap a built AgentKit image with the Azure Foundry invocations protocol.
# Build the base image with test/foundry-hosted-agent/agentkitfile.yaml first.
ARG BASE_IMAGE=foundry-agentkit-base:test
FROM ${BASE_IMAGE}

USER 0
RUN /opt/agentkit/bin/pip install --no-cache-dir azure-ai-agentserver-invocations==1.0.0b5
COPY foundry_invocations.py /opt/agentkit/foundry_invocations.py
RUN chown 1000:1000 /opt/agentkit/foundry_invocations.py

ENV PORT=8088
EXPOSE 8088
USER 1000

ENTRYPOINT ["/opt/agentkit/bin/python", "/opt/agentkit/foundry_invocations.py"]
