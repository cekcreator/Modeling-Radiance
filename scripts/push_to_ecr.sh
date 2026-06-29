#!/usr/bin/env bash
# Push the Libera unfiltered radiances Docker image to AWS ECR.
#
# Required env vars:
#   AWS_ACCOUNT_ID  — your 12-digit AWS account ID
#   AWS_REGION      — e.g. us-east-1
#
# Optional env vars:
#   IMAGE_NAME  (default: libera-l2-unfiltered-radiances)
#   IMAGE_TAG   (default: latest)
#
# Usage:
#   AWS_ACCOUNT_ID=123456789012 AWS_REGION=us-east-1 bash scripts/push_to_ecr.sh
set -euo pipefail

: "${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID must be set}"
: "${AWS_REGION:?AWS_REGION must be set}"

IMAGE_NAME="${IMAGE_NAME:-libera-l2-unfiltered-radiances}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ECR_REPO="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
FULL_URI="${ECR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG} ..."
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo "Authenticating with ECR ..."
aws ecr get-login-password --region "${AWS_REGION}" | \
    docker login --username AWS --password-stdin "${ECR_REPO}"

echo "Pushing ${FULL_URI} ..."
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${FULL_URI}"
docker push "${FULL_URI}"

echo "Done: ${FULL_URI}"
