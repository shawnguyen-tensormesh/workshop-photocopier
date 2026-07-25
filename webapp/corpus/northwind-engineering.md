# Northwind Platform — Engineering Runbook (excerpt)

## Service Topology

The Northwind order platform runs three core services: the storefront API, the order worker, and
the ledger. The storefront API is stateless and horizontally scaled behind a load balancer; it
never writes to the database directly, instead enqueuing work onto the order queue. The order
worker consumes the queue, validates inventory against the tiers service, and commits accepted
orders to the ledger. The ledger is the single source of truth for order state and is the only
service permitted to write to the primary Postgres cluster.

Each service exposes a health endpoint at /healthz and a Prometheus metrics endpoint at /metrics.
The platform SLO is 99.9% availability measured monthly, with a p99 order-commit latency target of
400 milliseconds under normal load.

## Deployment

Deploys are gated by CI: unit tests, an integration suite against an ephemeral database, and a
schema-compatibility check. A release is promoted to staging automatically on a green main build,
but promotion to production requires a manual approval from an on-call engineer. Rollbacks are
performed with the same pipeline by re-deploying the previous image digest; never hand-edit a
running deployment. The canary receives 5% of traffic for ten minutes before a full rollout, and
an automatic rollback triggers if the canary error rate exceeds 2%.

## Incident Response

When an alert fires, the on-call engineer acknowledges within five minutes and opens an incident
channel. Severity 1 means customer-facing outage or data loss; Severity 2 means degraded service
without full outage; Severity 3 is a minor issue with no customer impact. Sev-1 incidents require a
written post-incident review within three business days, published to the whole engineering org.
The primary on-call escalates to the secondary after fifteen minutes without acknowledgement, and
to the engineering manager after thirty.

## Data Retention

Order records are retained for seven years for tax and audit purposes. Application logs are kept
for 30 days in hot storage and archived to cold storage for one year. Personally identifiable
information is encrypted at rest and may be deleted on verified customer request within 30 days,
except where retention is legally required. Backups of the ledger run every six hours and are
tested by a monthly restore drill; a restore that misses the six-hour objective is treated as a
Sev-2 incident.
