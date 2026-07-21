#!/usr/bin/env bash
# Starts a local Apache Pulsar standalone instance without Docker.
# Requires Java 17+ on the host.
set -euo pipefail

PULSAR_VERSION="3.2.2"
INSTALL_DIR="${PULSAR_INSTALL_DIR:-$HOME/.local/pulsar}"

if [ ! -d "$INSTALL_DIR/apache-pulsar-${PULSAR_VERSION}" ]; then
  echo "Downloading Apache Pulsar ${PULSAR_VERSION}..."
  mkdir -p "$INSTALL_DIR"
  curl -L -o /tmp/pulsar.tar.gz \
    "https://archive.apache.org/dist/pulsar/pulsar-${PULSAR_VERSION}/apache-pulsar-${PULSAR_VERSION}-bin.tar.gz"
  tar -xzf /tmp/pulsar.tar.gz -C "$INSTALL_DIR"
fi

cd "$INSTALL_DIR/apache-pulsar-${PULSAR_VERSION}"
echo "Starting Pulsar standalone (Ctrl+C to stop)..."
bin/pulsar standalone
