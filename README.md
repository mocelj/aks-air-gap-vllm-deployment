# AKS (Isolated) GPU + LLM Deployment Scenarios

This folder contains **public/sanitized** Kubernetes manifests and a Dockerfile demonstrating three scenarios:

1) GPU nodepool validation (CUDA pod + interactive `nvidia-smi`)
2) Offline model deployment from Hugging Face (weights baked into the image)
3) Offline model deployment using NVIDIA NIM with weights staged on shared storage

All manifests in this folder use placeholders like `<YOUR-…>`; replace them with your environment values before applying.
The original (environment-specific) versions live under `internal-resources/` and should not be committed.

## Before you run anything (replace placeholders)

- `<YOUR-ACR-NAME>`: your Azure Container Registry name (without `.azurecr.io`)
- `<YOUR-AZURE-FILES-DNS>`: your Azure Files NFS endpoint DNS name
- `<YOUR-AZURE-FILES-PATH>`: the exported NFS path (share + optional subfolder)
- `<SERVICE_IP>`: the internal LoadBalancer IP shown by `kubectl get svc ... -o wide`
- `<YOUR-NGC-API-KEY>` / `HF_TOKEN`: keep these out of source control

---

## 1) Test the GPU nodepool (CUDA pod)

Use the CUDA pod to validate that:
- the pod schedules onto the GPU nodepool (node selector + tolerations)
- the NVIDIA device plugin is working
- the container can see the GPU

Manifest:
- [gpu-test-cuda-pod.yaml](gpu-test-cuda-pod.yaml)

Apply and validate:

```bash
kubectl apply -f gpu-test-cuda-pod.yaml
kubectl get pod cuda -o wide
kubectl exec -it cuda -- nvidia-smi
```

Cleanup:

```bash
kubectl delete pod cuda
```

---

## 2) Deploy a model prepared for offline usage from Hugging Face

Goal: build a container image that already contains the model weights, then run it in an isolated cluster **without downloading anything at runtime**.

### Prerequisites

- Create a Hugging Face API token (`HF_TOKEN`).
- Select the model you want to use on Hugging Face.
- For gated models (e.g., Llama 3), you must request access and get usage approved on Hugging Face.

### How the image is built

The Docker build uses the Hugging Face CLI to download the model artifacts during build-time and stores them in the image filesystem. Runtime is configured for offline mode.

Dockerfile:
- [Dockerfile.hf-llama3-offline](Dockerfile.hf-llama3-offline)

Build with ACR (build happens in Azure; the token is passed as a build-arg):

```bash
az acr build \
  --registry <YOUR-ACR-NAME> \
  --image llama3-vllm-fat:8b-instruct \
  --build-arg HF_TOKEN=$HF_TOKEN \
  .
```

### Deploy

Manifest:
- [deployment-llama3-vllm-fat.yaml](deployment-llama3-vllm-fat.yaml)

Apply:

```bash
kubectl apply -f deployment-llama3-vllm-fat.yaml
```

### Test

Show the service and take note of the exposed (internal) service IP:

```bash
kubectl get svc vllm-gptoss -o wide
```

From a host that can reach the internal load balancer IP (for example, an Azure Jumpbox in the same VNet), you can interact with the vLLM OpenAI-compatible endpoints:

```bash
curl -v "http://<SERVICE_IP>:8000/v1/models"
```

```bash
curl -X 'POST' "http://<SERVICE_IP>:8000/v1/chat/completions" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {
      "content": "You are a polite and respectful chatbot helping people plan to find nice restaurants.",
      "role": "system"
    },
    {
      "content": "What should I go for lunch close to the Microsoft office in Pratteln?",
      "role": "user"
    }
  ],
  "model": "meta/llama3-8b-instruct",
  "max_tokens": 512,
  "top_p": 1,
  "n": 1,
  "stream": false,
  "frequency_penalty": 0.0
}'
```

---

## 3) Use the same model via NVIDIA NIM (isolated + staged weights)

Goal: run NVIDIA NIM in an isolated cluster while ensuring **no runtime dependency on external registries/model downloads**.

### Prerequisites

- You need an NVIDIA NGC API key to access NIM images from NGC (e.g., `nvcr.io`), **or** you can pull via a cached ACR if you have an ACR pull-through cache / caching rule configured for `nvcr.io`.

### Weight staging approach (offline-first)

High-level flow:

1. Use a connected machine (jumpbox) with internet access to download the model weights using NVIDIA’s documented commands.
2. Copy the downloaded model artifacts and the NIM profile/model metadata to a shared NFS location.
3. In the isolated cluster, mount that NFS location into the NIM container at the cache path so NIM can start without reaching out externally.

### Jumpbox recommendation (fast downloads)

If you are using a Jumpbox in Azure, the fastest way to download the weights is to use the same GPU VM size as you intend to use for inference (for example, a VM with an A100 GPU).

To avoid manual installation of most of the software, we recommend using the **DSVM** (Data Science Virtual Machine) marketplace Linux image. You will still need to install the NVIDIA container runtime.

#### Ubuntu: install NVIDIA container runtime

```bash
# Add NVIDIA package repositories

# Set up GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Add the repository to your sources list
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use the runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

#### Mount the shared file share (NFS)

After the container runtime is installed, mount the shared drive that will hold the NIM cache (model weights + profile).

```bash
# Shared Drive (replace placeholders with your values)
sudo mkdir -p /mount/<YOUR-AZURE-FILES-MOUNT>
sudo mount -t nfs <YOUR-AZURE-FILES-DNS>:/<YOUR-AZURE-FILES-PATH> /mount/<YOUR-AZURE-FILES-MOUNT> -o vers=4,minorversion=1,sec=sys
```

#### Prepare to download the weights (connected Jumpbox)

Use the following steps on the connected Jumpbox to authenticate, pull the NIM image, and prepare a local cache directory.

```bash
# ACR + NIM image settings
export ACR_NAME="<YOUR-ACR-NAME>"
export LOCAL_NIM_CACHE=~/nim-llama3-cache
mkdir -p "$LOCAL_NIM_CACHE" && chmod -R a+w "$LOCAL_NIM_CACHE"
export TARGET_IMAGE="$ACR_NAME.azurecr.io/nvcr.io/nim/meta/llama3-8b-instruct:latest"

# NGC key (do NOT hardcode in scripts; keep it in a secure secret store)
export NGC_API_KEY="<YOUR-NGC-API-KEY>"

# Login + pull
az login --identity
az acr login --name "$ACR_NAME"
docker pull "$TARGET_IMAGE"
```

#### Explore available profiles and download weights to the Jumpbox

Run these commands on the temporary GPU Jumpbox (connected VM) to inspect available profiles and download the model artifacts into the local cache directory.

Choose the optimal profile automatically (downloads to cache):

```bash
docker run --rm \
  --runtime=nvidia \
  --gpus all \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -u "$(id -u)" \
  -e NGC_API_KEY \
  "$TARGET_IMAGE" \
  download-to-cache
```

List available model profiles:

```bash
docker run --rm \
  --runtime=nvidia \
  --gpus all \
  -e NGC_API_KEY \
  "$TARGET_IMAGE" \
  list-model-profiles
```

Download an explicit profile:

```bash

 docker run --rm \
   --runtime=nvidia \
   --gpus all \
   -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
   -u "$(id -u)" \
   -e NGC_API_KEY \
   "$TARGET_IMAGE" \
   download-to-cache --profile <PROFILE_NAME>
```

After the download completes, copy the downloaded cache/profiles into the **shared NFS folder** that will be mounted by the AKS deployment manifest (so the isolated cluster can start without downloading anything).

For example, if you mounted the share at `/mount/<YOUR-AZURE-FILES-MOUNT>`:

```bash
sudo rsync -avh "$LOCAL_NIM_CACHE/" "/mount/<YOUR-AZURE-FILES-MOUNT>/"
```

Tip: ensure the destination corresponds to the `volumes[].nfs.path` used in [deployment-nim-llama3-isolated.yaml](deployment-nim-llama3-isolated.yaml).

### Deployment behavior to enforce offline startup

Manifest:
- [deployment-nim-llama3-isolated.yaml](deployment-nim-llama3-isolated.yaml)

Key points:
- The pod mounts the shared NFS directory into the path specified by `NIM_CACHE_PATH`.
- If model weights are already present on the mounted path, NIM uses them instead of downloading from NGC.
- Do **not** provide the NVIDIA API key to the running workload in the isolated cluster when you want to force offline behavior.
  - In the manifest, the `NGC_API_KEY` secret reference is intentionally commented out.
- Ensure the NFS cache is complete.
  - If files are missing, NIM may attempt to download them (which will fail in an isolated cluster).

Apply:

```bash
kubectl apply -f deployment-nim-llama3-isolated.yaml
```

### Test

Show the service and take note of the exposed (internal) service IP:

```bash
kubectl get svc vllm-nim-llama3-service -o wide
```

From a host that can reach the internal load balancer IP (for example, an Azure Jumpbox in the same VNet), you can interact with the OpenAI-compatible endpoints:

```bash
curl -v "http://<SERVICE_IP>:8000/v1/models"
```

```bash
curl -X 'POST' "http://<SERVICE_IP>:8000/v1/chat/completions" \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {
      "content": "You are a polite and respectful chatbot helping people plan to find nice restaurants.",
      "role": "system"
    },
    {
      "content": "What should I go for lunch close to the Microsoft office in Pratteln?",
      "role": "user"
    }
  ],
  "model": "meta/llama3-8b-instruct",
  "max_tokens": 512,
  "top_p": 1,
  "n": 1,
  "stream": false,
  "frequency_penalty": 0.0
}'
```

If the request fails due to an unknown model ID, use the `/v1/models` output to pick a valid `model` value.

---

## License

MIT License


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
