
# SmartShop Advisor – Agent 

Asistente inteligente que recomienda productos a partir de un texto

## Características
- Entrada por texto (descripción de lo que quieres)
- Backend en FastAPI

## Instrucciones

1. Instalar dependencias:
```bash
pip install -r requirements.txt


- Integración de logs: cualquier contenedor que escriba logs en `/var/lib/docker/containers` será recolectado por Promtail (si ejecutas en Linux). Para Windows/ Docker Desktop ajusta la ruta o usa el volumen `app_logs` si lo prefieres.

  En `logging_assets/monitoring/promtail/config.yml` ya se establece `job: "docker"` para facilitar consultas en Grafana:

  - Consulta ejemplo en el panel de logs: `{job="docker"}` o `{job="docker"} |= "ERROR"`

## Alertas y dashboards

-- Reglas de alerta: `logging_assets/monitoring/prometheus/rules/alerts.yml` contiene reglas de ejemplo (`HighErrorRate`, `HighLatencyP95`). Puedes añadir más reglas en ese archivo o crear nuevos ficheros `.yml` dentro de `logging_assets/monitoring/prometheus/rules/`.

-- Alertmanager: la configuración de `logging_assets/monitoring/alertmanager/config.yml` define `receivers` vacíos por defecto. Añade `email_configs`, `slack_configs` o `webhook_configs` para recibir notificaciones.

-- Dashboards: los dashboards están en `logging_assets/grafana/provisioning/dashboards/` y se importan automáticamente al arranque de Grafana. Si quieres añadir un dashboard nuevo, simplemente coloca el `.json` en esa carpeta.

## 📊 Acceder a los servicios

Arranca el stack:

```powershell
docker-compose up --build
```

- **Aplicación FastAPI**: http://localhost:8000
  - `GET /` - Endpoint raíz
  - `GET /random_event` - Evento aleatorio
  - `GET /metrics` - Métricas Prometheus 

- **Prometheus**: http://localhost:9090 — Status → Targets debe mostrar los servicios configurados

- **Grafana**: http://localhost:3000
  - Usuario: `admin` / Contraseña: `admin`
  - Dashboard: "Observabilidad FastAPI + Prometheus + Loki" (importado automáticamente)

- **Loki**: http://localhost:3100

Si no ves logs en Grafana, asegúrate de que Promtail tenga acceso a los logs del host. En Linux usamos `/var/lib/docker/containers` montado en el servicio `promtail`.

## 📝 Características del Dashboard

✅ **Métricas en tiempo real:**
   - Total de errores (últimos 5 minutos)
   - Total de solicitudes (últimos 5 minutos)
   - Tasa de solicitudes por segundo
   - Latencia p95 por endpoint
   - Solicitudes por endpoint y status code

✅ **Logs en tiempo real:**
   - Integración con Loki
   - Visualización de todos los logs de la aplicación

## 🔧 Variables de entorno de Grafana

```yaml
GF_SECURITY_ADMIN_PASSWORD: admin
GF_SECURITY_ADMIN_USER: admin
GF_PATHS_PROVISIONING: /etc/grafana/provisioning
GF_USERS_ALLOW_SIGN_UP: false
```