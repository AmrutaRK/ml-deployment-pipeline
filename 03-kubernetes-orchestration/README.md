# Stage 03: Kubernetes Orchestration

Runs both the ResNet classification service and the YOLO detection
service together, in one Kubernetes cluster - two independent workloads,
each with their own scaling and self-healing, sharing the same underlying
infrastructure. This is a better demonstration of what orchestration
actually does than running just one service would be.

---

## What Kubernetes is, and why it exists

Docker solves "how do I package and run one container reliably." It does
NOT solve what happens when there are many containers, spread across many
machines, that need to survive crashes, handle traffic spikes, and get
updated without downtime - all without a human manually intervening every
time something goes wrong. That's the actual problem Kubernetes exists to
solve: automated management of many containers, at scale, based on rules
declared once rather than commands issued repeatedly.

## The core idea: declarative, not imperative

`docker run` is an imperative instruction - "do this one specific thing,
right now." Kubernetes works differently: you write a YAML file declaring
a *desired end state* - "there should always be 3 copies of this running"
- and Kubernetes' job becomes continuously making reality match that
statement, forever, however that requires: starting new Pods, killing
broken ones, whatever it takes, without further commands from a human.

That's exactly what `replicas: 3` in the Deployment manifests below means:
not "start 3 containers now," but "keep 3 running, permanently, and fix
it yourself if reality ever drifts from that."

## Core building blocks, explained with a restaurant-chain analogy

**Pod** - the smallest unit Kubernetes manages, usually one container
(e.g. one running copy of the ResNet or YOLO service). Analogy: one
physical restaurant location, chef inside, currently open. Pods are
disposable by design - Kubernetes kills and replaces them freely, and
that's intentional, not a failure state.

**Deployment** - a standing rule for how many Pods of something should
exist. Analogy: the franchise rule "there must always be exactly 3
locations open in this city." If one location burns down, the franchise
system notices and opens a replacement automatically - this is what
"self-healing" means in practice. Deployments also manage rolling
updates: pushing a new image version replaces old Pods with new ones
gradually, without taking the whole service offline.

**Service** - Pods get recreated constantly, each with a new internal
address, so nothing can reliably point directly at a specific Pod. A
Service is a stable, unchanging address that automatically routes
traffic to whichever Pods are currently alive and healthy. Analogy: the
restaurant chain's single phone number / delivery app - customers call
the chain, not a specific location, and get routed to whichever location
is currently open.

**ReplicaSet** - the mechanism that actually maintains the replica count
for a Deployment. Rarely written by hand - Deployments create and manage
these automatically. Mentioned mainly so the term isn't mysterious when
it shows up in `kubectl` output.

**Node** - one physical (or virtual) machine in the cluster, capable of
running Pods. Analogy: one city block where restaurant locations can
physically be built.

**Cluster** - all the nodes together, managed as one system. Analogy: the
whole metro area, under one franchise operations system.

**HorizontalPodAutoscaler (HPA)** - automatically increases or decreases
replica count based on real load (CPU usage here). Analogy: "if lines get
too long, open more locations; if it's dead quiet, close some down" -
automatic, based on real demand, not a fixed number.

**Namespace** - a way to logically separate groups of resources (e.g.
`dev` vs `prod`) within one cluster. Not used in this example (everything
here lives in the `default` namespace), but extremely common in real
company setups - worth knowing the term.

## What Kubernetes gives you that plain `docker run` doesn't

- **Self-healing** - a crashed Pod gets automatically replaced
- **Scaling** - more/fewer replicas based on real load, automatically
- **Rolling updates** - new image versions roll out gradually, with
  automatic rollback if health checks start failing
- **Service discovery** - a stable name to reach a group of Pods,
  regardless of individual Pods being recreated constantly
- **Efficient packing** - many containers sharing a smaller number of
  physical machines, rather than one container per machine

## What it doesn't do

Kubernetes runs container images - it doesn't build them. `docker build`
and `docker push` still happen exactly as before; Kubernetes only takes
over from "image sitting in Artifact Registry" onward.

## How this compares to Cloud Run (what was used in stage 01)

| | Cloud Run | Kubernetes |
|---|---|---|
| Who manages scaling/healing | Google, invisibly | You, via explicit YAML rules |
| Idle cost | Scales to zero - $0 | Cluster keeps costing, even at low traffic |
| Control | Minimal - hand over an image, done | Fine-grained - explicit replica counts, resource limits, autoscaling thresholds |
| Good for | A single simple service | Many interdependent services needing coordination |

Neither is strictly "better" - Cloud Run was the right choice for a
single service with no coordination needs, which is exactly why it was
used first. Kubernetes earns its added complexity here specifically
because there are now two independent services being run and scaled
together.

---

## What's in this folder

- `manifests/resnet-deployment.yaml` / `resnet-service.yaml` / `resnet-hpa.yaml`
- `manifests/yolo-deployment.yaml` / `yolo-service.yaml` / `yolo-hpa.yaml`

Each service gets its own Deployment (its own replica rules), its own
Service (its own external IP), and its own HPA (scaling tuned to that
service's actual resource needs - YOLO's requests/limits are set higher
than ResNet's, since detection + NMS is heavier per request than
classification).

## How to actually run this, on GKE

```bash
# Create a small, cheap GKE cluster (no GPU needed for this stage)
gcloud container clusters create-auto mlops-cluster --region=europe-west3

# Point kubectl at it
gcloud container clusters get-credentials mlops-cluster --region=europe-west3

# Deploy both services
kubectl apply -f manifests/resnet-deployment.yaml
kubectl apply -f manifests/resnet-service.yaml
kubectl apply -f manifests/resnet-hpa.yaml
kubectl apply -f manifests/yolo-deployment.yaml
kubectl apply -f manifests/yolo-service.yaml
kubectl apply -f manifests/yolo-hpa.yaml

# Watch Pods come up - should see 3 resnet18 Pods and 2 yolo-detection Pods
kubectl get pods

# Get external IPs once the LoadBalancer Services are ready (can take a minute)
kubectl get services
```

Each Service gets its own external IP - visiting the ResNet one hits the
classification service, the YOLO one hits the detection service,
completely independently, even though both are running in the same
cluster.

**Useful commands once running:**
```bash
kubectl get deployments        # see replica counts for both services
kubectl get hpa                # see current vs target CPU utilization
kubectl describe pod <name>    # debug a specific Pod
kubectl logs <pod-name>        # view a specific Pod's logs
kubectl delete -f manifests/   # tear everything down
```

**Cost note:** unlike Cloud Run, a GKE cluster isn't free while idle - you
pay for the underlying compute as long as the cluster exists. 
Delete the cluster when done:
```bash
gcloud container clusters delete mlops-cluster --region=europe-west3
```