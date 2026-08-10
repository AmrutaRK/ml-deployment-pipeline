# Stage 04: CI/CD with GitHub Actions

## What CI/CD actually means, broken into its two halves

**CI - Continuous Integration:** automatically building and testing code
every time it changes, to catch problems immediately rather than
discovering them later. "Integration" refers to the original idea of
frequently merging (integrating) code changes and verifying they still
work together - as opposed to everyone working in isolation for weeks and
discovering conflicts/breakage all at once at the end.

**CD - Continuous Deployment:** automatically shipping code that passes
CI out to a real, running environment - no manual `docker build` /
`push` / `kubectl` commands typed by hand.

Put together: **push code → it's automatically built, tested, and (if it
passes) deployed - with zero manual steps in between.**

## Why this matters, concretely, compared to what's been done so far

Every deployment up to this point has been manually run using a
sequence of commands from a terminal. That works, but it doesn't scale
and it's error-prone - a step skipped, a typo in an image tag, forgetting
to test before pushing. CI/CD removes the human from that repetitive
loop entirely: the *same* sequence of steps runs identically every single
time, defined once in a file instead of remembered and retyped.

## What triggers the workflow

```yaml
on:
  push:
    branches: [main]
    paths:
      - '01-onnx-fastapi-docker-gcp/yolo-detection/**'
```

This means: the workflow only runs when code is pushed to the `main`
branch, AND only when files inside `01-onnx-fastapi-docker-gcp/yolo-detection/` 
actually changed. Editing the Kubernetes manifests or this README won't trigger a YOLO
rebuild - deliberately scoped, so unrelated changes don't cause
unnecessary rebuilds/redeploys.

## Walking through what actually happens, step by step

1. **Checkout code** - GitHub spins up a brand new, empty virtual
   machine for this run. It has nothing on it - not even the repo -
   until this step explicitly clones it.

2. **Authenticate to GCP** - uses the `GCP_SA_KEY` secret set up
   earlier, so this anonymous temporary machine can act as the GCP
   service account for the rest of the steps.

3. **Set up Cloud SDK** - installs `gcloud` on the runner (not present
   by default).

4. **Configure Docker for Artifact Registry** - lets Docker push images
   using the GCP credentials from step 2.

5. **Build the image** - the exact same `docker build` command run
   manually before, tagged two ways: with the specific git commit hash
   AND `:latest`. The commit-hash tag matters more than it might seem -
   it means every build is uniquely, permanently identifiable, so a
   rollback can target an exact known-good version instead of guessing.

6. **Smoke test** - this is the actual "CI" part earning its name. The
   built image is run for real, right here, and `/health` is hit with
   curl. If it fails, the whole workflow stops immediately - a broken
   image never reaches push or deploy. This is the single most important
   step for actually catching problems before they become production
   incidents.

7. **Push to Artifact Registry** - only runs if the smoke test passed.
   Same registry used for manual deploys before.

8. **Get GKE credentials** - equivalent of the `gcloud container
   clusters get-credentials` command run manually, so `kubectl` in the
   next steps can reach the real cluster.

9. **Deploy - `kubectl set image`** - tells the *existing* Deployment
   to switch to the new image. This triggers Kubernetes' built-in rolling
   update behavior (from stage 03) - new Pods with the new image start
   up, old Pods are terminated gradually, with no downtime window where
   the service is fully down.

10. **Verify rollout succeeded** - waits and confirms the new Pods
    actually became healthy, rather than just firing the deploy command
    and assuming it worked. If new Pods keep failing their liveness
    probe, this step fails loudly in the workflow log - visible
    immediately, not discovered later.

## Why smoke testing (step 6) matters so much

Without it, a broken image could get pushed and deployed automatically,
with nobody noticing until real users hit errors. The smoke test is the
actual safety net that makes "continuous deployment" reasonable to trust
rather than reckless - fully automatic deployment without automatic
verification first would just be automating the ability to break
production faster.

## Setup: GCP authentication

**Step 1: Create the service account**
```bash
gcloud iam service-accounts create github-actions-deployer \
  --display-name="GitHub Actions Deployer"
```

**Step 2: Grant it exactly the permissions it needs (not more)**
```bash
PROJECT_ID=$(gcloud config get-value project)

# Permission to push/pull Docker images to Artifact Registry
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

# Permission to deploy to your GKE cluster
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-deployer@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/container.developer"
```

Why scoped roles, not "Owner": giving GitHub's servers full admin access to your GCP account would be a real security risk if that credential ever leaked. Scoping it to just "push images" + "deploy to GKE" limits the blast radius if anything goes wrong — a genuinely important habit, not just caution for its own sake.

**Step 3: Create a key for this service account**
```bash
gcloud iam service-accounts keys create github-actions-key.json \
  --iam-account=github-actions-deployer@$PROJECT_ID.iam.gserviceaccount.com
```

This downloads a JSON file — this is effectively the service account's password. Treat it like one. **Never commit this file to the repo.**

**Step 4: Add it to GitHub as a Secret**

In your GitHub repo: **Settings → Secrets and variables → Actions → New repository secret**

Add two secrets:
- `GCP_SA_KEY` — paste the entire contents of `github-actions-key.json`
- `GCP_PROJECT_ID` — your GCP project ID

## Secrets used (set up earlier, in GitHub repo settings)

- `GCP_SA_KEY` - the service account key, scoped to only
  `artifactregistry.writer` and `container.developer` - deliberately
  minimal permissions, not broad admin access.
- `GCP_PROJECT_ID` - the GCP project ID.

## How to actually see this run

Push a change to anything inside `01-onnx-fastapi-docker-gcp/yolo-detection/`, then go to the
repo's **Actions** tab on GitHub - the workflow run appears there, with
each of the 10 steps above shown live, including full logs for anything
that fails.
