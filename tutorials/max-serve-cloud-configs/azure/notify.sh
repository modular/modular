#!/bin/bash
##===----------------------------------------------------------------------===##
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##===----------------------------------------------------------------------===##

RESOURCE_GROUP=$1
VM_PASSWORD=$2
PUBLIC_IP=$3
VM_NAME=maxServeVM
MAX_WAIT_MINUTES=30
START_TIME=$(date +%s)

fetch_logs() {
    echo "=== Azure VM Logs ==="
    echo "📋 Attempting to fetch logs from VM at ${PUBLIC_IP}..."

    LOGS=$(sshpass -p "$VM_PASSWORD" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=50 -v azureuser@$PUBLIC_IP "
        # First check if container is running
        CONTAINER_ID=\$(sudo docker ps -q -f ancestor=docker.modular.com/modular/max-nvidia-full:latest)
        if [ -n \"\$CONTAINER_ID\" ]; then
            echo '=== Docker Container Found ==='
            sudo docker logs \$CONTAINER_ID
        else
            echo '=== Checking Deployment Logs ==='
            # Check deployment logs if container isn't running yet
            sudo cat /var/log/azure/custom-script/handler.log 2>/dev/null || echo 'No handler.log';
            sudo cat /var/lib/waagent/custom-script/download/0/stdout 2>/dev/null || echo 'No stdout';
            sudo cat /var/lib/waagent/custom-script/download/0/stderr 2>/dev/null || echo 'No stderr';
        fi
    " 2>&1)

    if [ $? -ne 0 ]; then
        echo "❌ Failed to connect to VM or fetch logs"
        echo "Debug info: $LOGS"
        return 1
    fi

    echo "$LOGS"
    echo "===================="
}

check_server_status() {
    local logs=$1
    echo "🔍 Checking logs for server status..."

    if echo "$logs" | grep -q "Pulling from docker.modular.com/modular/max-nvidia-full"; then
        echo "⏳ Docker image is still being pulled..."
        return 1
    fi

    if curl -s -f "http://$PUBLIC_IP/v1/health" >/dev/null; then
        echo "✅ Server health check passed!"
        return 0
    fi

    if echo "$logs" | grep -q "Building\|Compiling\|Starting download of model"; then
        echo "⏳ Server is initializing (compiling model)..."
        return 1
    fi

    echo "⏳ Server still starting up..."
    return 1
}

echo "🔍 Starting monitoring for MAX server (max wait: ${MAX_WAIT_MINUTES} minutes)..."

if ! az vm show -g "$RESOURCE_GROUP" -n "$VM_NAME" &>/dev/null; then
    echo "❌ VM not found. Please check if VM exists in resource group ${RESOURCE_GROUP}"
    exit 1
fi

while true; do
    current_time=$(date +%s)
    elapsed_minutes=$(((current_time - START_TIME) / 60))

    if [ $elapsed_minutes -ge $MAX_WAIT_MINUTES ]; then
        echo "❌ Timeout after ${MAX_WAIT_MINUTES} minutes. Server might still be starting up."
        echo "👉 You can manually check the logs by:"
        echo "   ssh azureuser@$PUBLIC_IP"
        echo "   sudo cat /var/log/azure/custom-script/handler.log"
        echo "   sudo cat /var/lib/waagent/custom-script/download/0/stdout"
        echo "   sudo cat /var/lib/waagent/custom-script/download/0/stderr"
        exit 1
    fi

    echo "⏳ Checking logs... (${elapsed_minutes}/${MAX_WAIT_MINUTES} minutes)"

    LOGS=$(fetch_logs)

    if [ -n "$LOGS" ]; then
        if check_server_status "$LOGS"; then
            echo "✅ Server is ready! (took ${elapsed_minutes} minutes)"
            echo "📋 Latest logs:"
            echo "$LOGS"
            exit 0
        fi
    else
        echo "⏳ Logs not yet available..."
    fi

    echo "⏳ Server still starting up... checking again in 60 seconds"
    echo "-------------------------------------------"
    sleep 60
done
