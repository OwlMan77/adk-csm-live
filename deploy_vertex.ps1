# Deploy OmniVoice to Vertex AI: build -> Artifact Registry -> Model Registry -> Endpoint
# Run from the adk-csm-live project root in PowerShell.
#
# Reads HF_TOKEN and project config from .env automatically.
#
# Usage:
#   .\deploy_vertex.ps1
#   .\deploy_vertex.ps1 -ImageTag v2 -MachineType g2-standard-8

param(
    [string]$ProjectId   = "",
    [string]$Region      = "",
    [string]$Repository  = "omnivoice",
    [string]$ImageTag    = "latest",
    [string]$MachineType = "g2-standard-4"
)

$ErrorActionPreference = "Stop"

# --- Load .env ----------------------------------------------------------------
$envFile = Join-Path $PSScriptRoot ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | Where-Object { $_ -match '^\s*[^#]\S+=\S' } | ForEach-Object {
        $parts = $_ -split '=', 2
        [System.Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), "Process")
    }
}

# --- Resolve defaults from .env -----------------------------------------------
if (-not $ProjectId) { $ProjectId = $env:GOOGLE_CLOUD_PROJECT }
if (-not $ProjectId) { throw "GOOGLE_CLOUD_PROJECT is not set in .env or environment." }
if (-not $Region)    { $Region = if ($env:OMNIVOICE_LOCATION) { $env:OMNIVOICE_LOCATION } else { $env:GOOGLE_CLOUD_LOCATION } }
if (-not $Region)    { throw "GOOGLE_CLOUD_LOCATION (or OMNIVOICE_LOCATION) is not set in .env or environment." }

# --- Config -------------------------------------------------------------------
$ImageName           = "omnivoice-vertex"
$ModelDisplayName    = "OmniVoice"
$EndpointDisplayName = "omnivoice-endpoint"
$AcceleratorType     = "nvidia-l4"
$AcceleratorCount    = 1

$ArHost      = "$Region-docker.pkg.dev"
$Image       = "$ArHost/$ProjectId/$Repository/$ImageName`:$ImageTag"
$Accelerator = "type=$AcceleratorType,count=$AcceleratorCount"

$HfToken = $env:HF_TOKEN
if (-not $HfToken) { throw "HF_TOKEN is not set in .env or environment." }

# --- Preflight ----------------------------------------------------------------
foreach ($cmd in "docker", "gcloud") {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        throw "$cmd not found on PATH."
    }
}

Write-Host "Project  : $ProjectId"
Write-Host "Region   : $Region"
Write-Host "Image    : $Image"
Write-Host "Machine  : $MachineType + ${AcceleratorCount}x $AcceleratorType"
Write-Host ""

# --- 1. Build -----------------------------------------------------------------
Write-Host "[1/6] Checking if Docker image exists locally..."
docker image inspect $Image 2>$null | Out-Null
if ($LASTEXITCODE -eq 0) {
    Write-Host "   Image found locally - skipping build."
} else {
    Write-Host "   Image not found - building (model weights baked in, ~10 min)..."
    docker build `
        --build-arg "HF_TOKEN=$HfToken" `
        --tag $Image `
        --file "vertex\Dockerfile" `
        .
    if ($LASTEXITCODE -ne 0) { throw "docker build failed." }
}

# --- 2. Artifact Registry repo (idempotent) -----------------------------------
Write-Host "[2/6] Ensuring Artifact Registry repository '$Repository' exists..."
$prev = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
gcloud artifacts repositories create $Repository `
    --repository-format=docker `
    --location=$Region `
    --project=$ProjectId `
    --description="OmniVoice serving containers" `
    --quiet 2>$null
$ErrorActionPreference = $prev

# --- 3. Push ------------------------------------------------------------------
Write-Host "[3/6] Pushing image to Artifact Registry..."
gcloud auth configure-docker $ArHost --quiet
docker push $Image
if ($LASTEXITCODE -ne 0) { throw "docker push failed." }

# --- 4. Register model --------------------------------------------------------
Write-Host "[4/6] Checking if model '$ModelDisplayName' already exists..."
$prev = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
$ModelResource = gcloud ai models list `
    --region=$Region `
    --project=$ProjectId `
    --filter="displayName=$ModelDisplayName" `
    --format="value(name)" `
    --limit=1 2>$null | Select-Object -First 1
$ErrorActionPreference = $prev

if ($ModelResource) {
    Write-Host "   Model found - skipping upload."
} else {
    Write-Host "   Model not found - uploading..."
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    $ModelResource = gcloud ai models upload `
        --region=$Region `
        --project=$ProjectId `
        "--display-name=$ModelDisplayName" `
        "--container-image-uri=$Image" `
        --container-health-route="/health" `
        --container-predict-route="/predict" `
        --container-ports=8080 `
        --format="value(model)" 2>$null
    $ErrorActionPreference = $prev
    if ($LASTEXITCODE -ne 0) { throw "Model upload failed (exit $LASTEXITCODE)." }
}
Write-Host "   Model: $ModelResource"

# --- 5. Create endpoint with dedicated DNS ------------------------------------
Write-Host "[5/6] Creating Vertex AI endpoint with dedicated DNS..."
$prev = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
$EndpointResource = gcloud ai endpoints create `
    --region=$Region `
    --project=$ProjectId `
    "--display-name=$EndpointDisplayName" `
    --format="value(name)" 2>$null
$ErrorActionPreference = $prev
if ($LASTEXITCODE -ne 0) { throw "Endpoint creation failed (exit $LASTEXITCODE)." }

$EndpointId = Split-Path $EndpointResource -Leaf
Write-Host "   Endpoint: $EndpointResource"

# --- 6. Deploy model to endpoint ----------------------------------------------
Write-Host "[6/6] Deploying model (GPU provisioning takes ~10-15 min)..."
gcloud ai endpoints deploy-model $EndpointId `
    --region=$Region `
    --project=$ProjectId `
    "--model=$ModelResource" `
    "--display-name=$ModelDisplayName" `
    "--machine-type=$MachineType" `
    "--accelerator=$Accelerator" `
    --min-replica-count=1 `
    --max-replica-count=2 `
    --traffic-split=0=100
if ($LASTEXITCODE -ne 0) { throw "Model deployment failed (exit $LASTEXITCODE)." }

# --- Done ---------------------------------------------------------------------
Write-Host ""
Write-Host "Deployment complete!"
Write-Host ""
Write-Host "Add this to your .env:"
Write-Host "  OMNIVOICE_ENDPOINT_ID=$EndpointId"
Write-Host ""
Write-Host "Then test with:"
Write-Host "  uv run python scripts/test_csm.py --text `"Hello, this is a test.`""
