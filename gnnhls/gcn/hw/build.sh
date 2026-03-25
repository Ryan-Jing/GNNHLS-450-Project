#!/bin/bash

# Exit on errors and fail if any command in a pipe fails.
set -euo pipefail

# Build the hardware artifact for deployment to the KV260.
# The host executable should be built separately on the target or cross-compiled for aarch64.
make gcn.xclbin | tee report_build.log
