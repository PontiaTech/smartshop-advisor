# 🛍️ SmartShop Advisor – Multimodal Fashion Search Agent

**Autores**: Paula Campreciós, Gaizka Menéndez, Steven Rodríguez

**SmartShop Advisor** es un asistente inteligente de búsqueda y recomendación de productos de moda que permite a los usuarios encontrar artículos relevantes a partir de **consultas en lenguaje natural** y, opcionalmente, **imágenes de referencia**.

El sistema combina técnicas de **búsqueda semántica (RAG)**, **LLMs**, y **recuperación web**, ofreciendo una experiencia cercana a la de un asesor de compra digital.

La solución se despliega íntegramente mediante **Docker Compose** y está diseñada como un **MVP funcional y extensible**.

---

## ✨ Características principales

* 🔎 **Búsqueda multimodal**: texto o texto + imagen
* 🧠 **Recuperación semántica (RAG)** basada en embeddings
* 🌍 **Soporte multilingüe** (detección y traducción automática con LLM)
* 🧵 **Gestión de contexto conversacional** (follow-up questions)
* 🛒 **Recomendaciones enriquecidas** con imágenes y metadatos
* 📦 **Arquitectura desacoplada por servicios**
* 📊 **Observabilidad completa**: métricas, logs, alertas y dashboards

---

## 🏗️ Arquitectura general

El sistema está compuesto por varios servicios orquestados mediante `docker-compose`, siguiendo una arquitectura desacoplada:

```
┌──────────────┐      ┌──────────────────┐
│   UI Gradio  │ ---> │  API SmartShop   │
└──────────────┘      └──────────────────┘
                              │
               ┌──────────────┼──────────────┐
               │              │              │
        ┌──────────┐   ┌────────────┐   ┌──────────┐
        │  Chroma  │   │ PostgreSQL │   │ SerpAPI  │
        └──────────┘   └────────────┘   └──────────┘
```

---

## 🧩 Servicios

### 🔹 Chroma – Base de datos vectorial

Chroma actúa como el **núcleo del sistema de recuperación semántica**, almacenando los embeddings de los productos del catálogo.

* Almacena representaciones semánticas de los productos
* Permite búsquedas por similitud a partir de lenguaje natural
* Persistencia garantizada mediante volumen Docker
* Conserva el conocimiento incluso tras reinicios del sistema

---

### 🔹 PostgreSQL – Base de datos relacional (`db-smartshopadvisor`)

En la fase inicial del diseño se contempló el uso de una base de datos relacional para almacenar información estructurada como:

* Perfiles de usuario
* Historial de conversaciones
* Registros de interacción
* Metadatos de productos

Tras priorizar la correcta implementación del flujo **RAG** y debido a limitaciones de tiempo y alcance, estas funcionalidades no se integraron completamente en la versión final.

Actualmente, la base de datos relacional queda **preparada para futuras extensiones**, aunque el sistema operativo se apoya principalmente en la base de datos vectorial.

---

### 🔹 Servicio de ingesta (`ingest-smartshopadvisor`)

Este servicio se utiliza para **poblar la base de datos vectorial** con el catálogo interno de productos.

* Script principal: `ingest_to_chroma_robust`
* Datasets utilizados: **Mango, Zara y Zalando**

#### Preprocesamiento aplicado

* Limpieza de registros erróneos o incompletos
* Eliminación de duplicados
* Selección y análisis de columnas relevantes

#### Campos almacenados por producto

* `product_name`
* `description`
* `family_raw`
* `raw_color`
* `source`
* `url`
* `image`

Adicionalmente, se crea un campo `text`, resultado de la concatenación de los atributos anteriores, para mejorar la recuperación semántica.

#### Embeddings

* Modelo: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
* Inserción por lotes: 2000 productos
* Identificador único por producto mediante `uuid`

---

### 🔹 API y lógica de negocio (`api-smartshopadvisor`)

La API está implementada con **FastAPI** y centraliza toda la lógica del sistema.

#### Endpoint principal

* `POST /chat`

Este endpoint orquesta el flujo completo de recomendación:

1. Recepción de la consulta del usuario y el historial conversacional
2. Resolución de contexto para consultas de seguimiento (follow-ups)
3. Normalización y sanitización de queries (RAG y web)
4. Recuperación interna (RAG) mediante Chroma
5. Clasificación de resultados mediante un regulador basado en LLM:

   * Mejor coincidencia
   * Productos similares
   * Productos descartados
6. Traducción de resultados al idioma solicitado
7. Búsqueda web complementaria (SerpAPI)
8. Construcción de un prompt estructurado con reglas anti-alucinación
9. Generación de la respuesta final mediante el LLM

La API devuelve:

* Respuesta textual
* Lista de productos seleccionados con metadatos e imágenes

---

### 🔹 Interfaz de usuario (`ui-gradio`)

La interfaz se implementa mediante **Gradio** y proporciona un punto de interacción directo e intuitivo con el sistema.

* Entrada por texto o texto + imagen
* Presentación estructurada de resultados
* Soporte visual mediante imágenes de producto
* Separación clara entre presentación y lógica de negocio
* Comunicación exclusiva con la API

Esta combinación de información textual y visual reduce la ambigüedad del lenguaje natural y acerca la experiencia a un escenario real de compra.

---

## 🚀 Puesta en marcha

### Arrancar el stack completo

```bash
docker-compose up --build
```

---

## 🌐 Acceso a los servicios

* **API FastAPI**: [http://localhost:8000](http://localhost:8000)

  * `POST /chat`
  * `GET /metrics`

* **UI Gradio**: [http://localhost:7860](http://localhost:7860)

* **Prometheus**: [http://localhost:9090](http://localhost:9090)
  *Status → Targets*

* **Grafana**: [http://localhost:3000](http://localhost:3000)

  * Usuario: `admin`
  * Contraseña: `admin`
  * Dashboard: *Observabilidad FastAPI + Prometheus + Loki*

* **Loki**: [http://localhost:3100](http://localhost:3100)

---

## 📊 Observabilidad

### Logs

* Recolección mediante **Promtail**
* Logs Docker montados desde `/var/lib/docker/containers`

Consultas de ejemplo en Grafana:

```
{job="docker"}
{job="docker"} |= "ERROR"
```

### Métricas

* Total de solicitudes
* Tasa de errores
* Latencia p95 por endpoint
* Solicitudes por status code

### Alertas

* Definidas en:

```
logging_assets/monitoring/prometheus/rules/alerts.yml
```

* Alertmanager preparado para email, Slack o webhooks

---

## 🔮 Trabajo futuro

* Autenticación y perfiles de usuario
* Persistencia del historial conversacional
* Personalización avanzada de recomendaciones
* Ampliación del catálogo mediante scraping
* Ranking avanzado de resultados
* Mejora del razonamiento multimodal
