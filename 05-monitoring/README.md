# Stage 05: Monitoring with Prometheus + Grafana

Adds real observability to the ResNet and YOLO services already running
on Kubernetes - actual request counts, latency, and error rates, visible
on a dashboard, instead of flying blind on how the services are actually
performing.

## Why this wasn't needed for the Cloud Run stage

Cloud Run gives you basic metrics automatically, for free, with nothing
to set up. Plain Kubernetes gives you none of that out of the box - it
manages *running* your containers, not *observing* them. Prometheus +
Grafana is the standard pairing that fills that gap.

## What each piece actually does

**Prometheus** - collects and stores metrics over time (a "time series
database" specialized for numbers-that-change-over-time, like request
count or latency). Critically, it works by **pulling**, not receiving
pushes: on a schedule (`scrape_interval: 15s` here), Prometheus itself
reaches out to each service's `/metrics` endpoint and grabs whatever
numbers are currently there. Your FastAPI services don't send anything
anywhere - they just passively expose current numbers whenever asked,
and Prometheus is the one doing the asking.

**Grafana** - a separate tool, purely for *visualizing* data that already
lives somewhere else (Prometheus, in this case). Grafana doesn't collect
or store metrics itself - it connects to Prometheus as a "data source"
and turns raw numbers into readable graphs and dashboards.

Restaurant analogy: **Prometheus is the inventory clerk** who walks
through the kitchen every 15 seconds, writing down exactly how many
orders came in, how long each took, noting anything that went wrong -
and keeps a running logbook of all of it over time. **Grafana is the
manager's office wall**, covered in charts built from that logbook, so
the manager can see trends at a glance instead of reading raw numbers.

## How the metrics actually get INTO your services

This required one small code change, already made in this stage:

```python
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

Two lines. `instrument(app)` wraps every existing route (`/predict`,
`/health`, etc.) so request count, latency, and in-progress request
counts get tracked automatically, with zero changes to `/predict` itself.
`expose(app)` adds the actual `GET /metrics` endpoint Prometheus will
scrape. 

**This change needs to be deployed** the same way any other code change
does - rebuild the image, push it, redeploy (the CI/CD pipeline from
stage 04 handles exactly this for the YOLO service and ResNet
service).

## What's in this folder

- `manifests/prometheus-configmap.yaml` - Prometheus's own config,
  telling it which two services to scrape and how often
- `manifests/prometheus-deployment.yaml` - runs Prometheus itself,
  mounting that config in
- `manifests/grafana-deployment.yaml` - runs Grafana, which will be
  pointed at Prometheus as its data source

## A new concept here: internal Service DNS, not external IPs

The Prometheus scrape config targets things like
`resnet18-service.default.svc.cluster.local:80` - this is DIFFERENT from
the external LoadBalancer IPs used to hit the services from a browser
earlier. Kubernetes automatically gives every Service an internal DNS
name, reachable only from *inside* the cluster. Prometheus doesn't need
to leave the cluster to reach your services - it's running inside the
same cluster, so it uses this internal address instead of going out to
the internet and back in.

## How to run this, on the existing GKE cluster

**First, create the Grafana admin password as a Kubernetes Secret** — this is deliberately a one-time imperative command, not a manifest file, so the actual password is never written to disk or committed to the repo:

```bash
kubectl create secret generic grafana-admin-secret \
  --from-literal=admin-password='<pick-your-own-password>'
```

```bash
kubectl apply -f manifests/prometheus-configmap.yaml
kubectl apply -f manifests/prometheus-deployment.yaml
kubectl apply -f manifests/grafana-deployment.yaml

# Get external IPs for both dashboards (can take a minute to provision)
kubectl get services prometheus-service grafana-service
```

### Verify Prometheus is actually scraping successfully

Open `http://<prometheus-external-ip>:9090`, go to **Status → Targets**.
Both `resnet18-service` and `yolo-detection-service` should show as
`UP`. If either shows `DOWN`, that specific service likely hasn't been
redeployed with the `/metrics` change yet.

### Set up Grafana

1. Open `http://<grafana-external-ip>:3000`
2. Log in: username `admin`, password whatever you set when creating
   `grafana-admin-secret` above
3. **Connections → Data sources → Add data source → Prometheus**
4. URL: `http://prometheus-service.default.svc.cluster.local:9090`
   (internal DNS again, same reasoning as above)
5. Click **Save & test** - should confirm the connection works

### A few starter PromQL queries to try on a dashboard panel

PromQL is Prometheus's query language - a few useful starting points:

```promql
# Requests per second, per service, over the last 5 minutes
rate(http_requests_total[5m])

# 95th percentile request latency - "95% of requests finish faster than this"
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Current number of requests being actively processed right now
http_requests_in_progress
```
