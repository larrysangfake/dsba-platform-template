# Stock API Runbook

## 📌 Overview

This runbook provides procedures for managing, operating, and debugging the Stock API service in both local and production environments.

---

## 🚀 Startup Procedures

### Local Development

```bash
# Start services locally using Docker Compose
docker-compose -f docker-compose.yml up
```

### Production (GKE)

```bash
# Deploy the Stock API to Google Kubernetes Engine (GKE)
kubectl apply -f deploy/kubernetes.yaml
```

---

## 📴 Shutdown Procedures

### Graceful Shutdown

```bash
# Scale down the deployment to zero pods
kubectl scale deployment stock-api --replicas=0
```

### Emergency Stop

```bash
# Force delete all pods of the Stock API
kubectl delete pod -l app=stock-api
```

---

## 🔄 Model Updates

### Trigger Model Retraining (via GitHub Actions)

```bash
curl -X POST -H "workflow" \
  https://api.github.com/repos/larrysang/stock-api/actions/workflows/model-training.yml/dispatches \
  -d '{"inputs":{"stock_ticker":"AAPL"}}'
```

### Verify New Model Deployment

```bash
# Inspect model files inside a running pod
kubectl exec -it <pod-name> -- ls -lh /app/models
```

---

## 🛠️ Debugging Guide

| Symptom        | Diagnostic Command                                        |
| -------------- | --------------------------------------------------------- |
| API 500 Errors | `kubectl logs -l app=stock-api --tail=100`                |
| High Latency   | `kubectl top pods`                                        |
| Model Failures | `kubectl exec -it <pod-name> -- python validate_model.py` |

---

## 🔑 Secrets Management

### Database Credentials

- **Location**: Stored in Kubernetes secrets under `stock-api-db`.
- **Access**:
  ```bash
  kubectl get secret stock-api -o yaml
  ```

### API Key Rotation Procedure

1. Generate a new API key from the third-party provider.
2. Update the corresponding Kubernetes secret:
   ```bash
   kubectl create secret generic stock-api-key --from-literal=API_KEY=<new_key> --dry-run=client -o yaml | kubectl apply -f -
   ```
3. Restart affected pods:
   ```bash
   kubectl rollout restart deployment/stock-api
   ```

---

## 📊 Observability

- Access **Grafana** dashboard:

  ```bash
  kubectl port-forward svc/grafana 3000:80
  ```

  Open: [http://localhost:3000](http://localhost:3000) (Default credentials: `admin/admin`)

- Key Metrics to Monitor:

  - `container_memory_usage_bytes{container="stock-api"}`: Memory usage
  - `increase(http_requests_total{endpoint="/predict"}[5m])`: Prediction request rate

---

## 📅 Contact List

| Name        | Role                       | Contact Info                                                 |
| ----------- | -------------------------- | ------------------------------------------------------------ |
| Baichuan DU | ML Engineer                | [baichuan.du@essec.edu](mailto\:baichuan.du@essec.edu)       |
| Linhui SANG | DevOps Specialist          | [larry.sang@student-cs.fr](mailto\:larry.sang@student-cs.fr) |
| Binong HAN  | Investment Data Scientist  | [b00761132@essec.edu](mailto\:b00761132@essec.edu)           |

---

## 📖 References

- [API Documentation](../docs/api.md)
- [Incident Response Playbook](../docs/incident-response.md)
- [Prometheus Alerting Rules](../monitor/alerts.yml)

---

*Last updated: (28/03/2025)





