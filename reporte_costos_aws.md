# Reporte de Costos AWS — NotionMap
**Versión:** 1.0 | **Fecha:** Junio 2026 | **Región base:** us-east-1 (N. Virginia)

---

## 1. Resumen Ejecutivo

| Escenario | AWS/mes | Gemini API/mes | Evaluaciones/mes | **TOTAL/mes** |
|-----------|--------:|---------------:|-----------------:|-------------:|
| MVP / Desarrollo | $32 | $1.50 | $0.20 | **~$34** |
| Producción Pequeña (equipo interno) | $98 | $6.00 | $1.00 | **~$105** |
| Producción Media (empresarial) | $265 | $30.00 | $1.50 | **~$297** |

> **Nota crítica:** El proyecto usa `torch` + `sentence-transformers` para embeddings locales.
> Esto requiere mínimo **3 GB de RAM libre** en tiempo de ejecución.
> Instancias con menos de 4 GB (t3.micro, t3.small) **no son viables**.

---

## 2. Análisis de Recursos del Proyecto

### Consumo real de memoria en runtime

| Componente | RAM estimada | CPU |
|---|---|---|
| FastAPI + Uvicorn | ~150 MB | Baja |
| PyTorch (overhead base) | ~500 MB | Media en inferencia |
| HuggingFace `all-MiniLM-L6-v2` cargado | ~450 MB | Media durante embedding |
| ChromaDB + SQLite en memoria | ~150 MB | Baja |
| LangChain + dependencias | ~200 MB | Baja |
| **Sistema operativo + buffer** | ~500 MB | — |
| **TOTAL EN REPOSO** | **~1.95 GB** | — |
| **TOTAL EN PICO (generando roadmap)** | **~2.8 GB** | Alta (5-15 seg) |

### Stack y sus implicaciones de costo

```
┌────────────────────────────────────────────────────┐
│  GRATIS (corre en tu servidor, sin costo variable) │
│  ✓ Embeddings HuggingFace all-MiniLM-L6-v2         │
│  ✓ ChromaDB (file-based, sin servidor externo)     │
│  ✓ FastAPI / Uvicorn                               │
├────────────────────────────────────────────────────┤
│  PAGO POR USO (API externa Google)                 │
│  $ Gemini 2.5 Flash — generación de roadmaps       │
│  $ Gemini 2.5 Flash — evaluaciones RAGAS + Judge   │
└────────────────────────────────────────────────────┘
```

---

## 3. Opciones de Arquitectura AWS

### Opción A — EC2 Simple (Recomendada para iniciar)

```
Internet
    │
    ▼
[Elastic IP]
    │
    ▼
┌─────────────────────┐
│  EC2 t3.large       │  ← FastAPI + ChromaDB + HuggingFace
│  2 vCPU / 8 GB RAM  │
│  Ubuntu 22.04 LTS   │
└──────────┬──────────┘
           │
    ┌──────▼──────┐
    │  EBS gp3    │  ← OS + App + vectorstore/chroma_db
    │  50 GB      │
    └─────────────┘
           │
    ┌──────▼──────┐
    │  S3 Bucket  │  ← PDFs originales + backups + snapshots
    └─────────────┘
```

**Características:**
- Un solo servidor, sin load balancer
- Puerto 8000 expuesto directamente (o nginx como proxy en el mismo servidor)
- ChromaDB persiste en disco EBS
- Ideal para equipos pequeños o demos

---

### Opción B — EC2 con ALB (Producción pequeña)

```
Internet
    │
    ▼
┌──────────────────────┐
│  Route 53            │  ← DNS: notionmap.tuempresa.com
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  ALB (Application    │  ← HTTPS (certificado ACM gratis)
│  Load Balancer)      │    + Health checks
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│  EC2 t3.large        │  ← FastAPI + app completa
│  2 vCPU / 8 GB       │
└──────────┬───────────┘
           │
    ┌──────▼──────┐    ┌──────────────┐
    │  EBS gp3    │    │  S3 Bucket   │
    │  80 GB      │    │  + Lifecycle │
    └─────────────┘    └──────────────┘
```

**Añade sobre Opción A:**
- ALB con SSL/TLS mediante AWS Certificate Manager (gratis)
- Health checks automáticos
- Dominio personalizado con Route 53

---

### Opción C — ECS Fargate (Sin servidores, Escalable)

```
Internet → ALB → ECS Fargate Task
                    │
                    ├── Container: FastAPI App
                    │   (2 vCPU / 8 GB)
                    │
                    └── EFS Mount ← vectorstore/chroma_db
                                    (persistencia ChromaDB)
```

**Ventaja:** Sin gestión de EC2, escala automáticamente.
**Desventaja:** ~20% más caro que EC2 equivalente + EFS añade latencia a ChromaDB.

---

### Opción D — Alta Disponibilidad (Empresarial)

```
Internet → Route 53 → CloudFront (frontend estático)
                            │
                       ALB + WAF
                            │
              ┌─────────────┴─────────────┐
              │                           │
         EC2 t3.large               EC2 t3.large
         (AZ us-east-1a)            (AZ us-east-1b)
              │                           │
              └───────────┬───────────────┘
                          │
                  ┌───────▼────────┐
                  │ RDS PostgreSQL │  ← pgvector reemplaza ChromaDB
                  │  db.t3.medium  │    (HA + backups automáticos)
                  └────────────────┘
```

**Nota:** Esta opción requiere migrar ChromaDB → pgvector en RDS.

---

## 4. Desglose de Costos AWS por Componente

> Todos los precios en USD, región us-east-1, Junio 2026.

### 4.1 Cómputo — EC2

| Instancia | vCPU | RAM | On-Demand/mes | Reserved 1yr/mes | Ahorro |
|-----------|------|-----|-------------:|----------------:|--------|
| t3.medium | 2 | 4 GB | $30.37 | $18.98 | 38% |
| **t3.large** ✓ | **2** | **8 GB** | **$60.74** | **$37.96** | **38%** |
| t3.xlarge | 4 | 16 GB | $121.47 | $75.92 | 38% |
| m5.large | 2 | 8 GB | $70.08 | $43.80 | 38% |
| m5.xlarge | 4 | 16 GB | $140.16 | $87.60 | 38% |

> **Recomendación:** `t3.large` cubre los ~2.8 GB de pico con margen seguro.
> `t3.medium` (4GB) puede funcionar en MVP pero sin margen para picos.

### 4.2 Almacenamiento — EBS

| Volumen | Uso | Costo/mes |
|---------|-----|----------:|
| 30 GB gp3 | OS + App + vectorstore (MVP) | $2.40 |
| 50 GB gp3 | OS + App + vectorstore + logs | $4.00 |
| 100 GB gp3 | Producción con crecimiento | $8.00 |
| Snapshots EBS (30 GB) | Backup semanal | $1.50 |

> gp3 = $0.08/GB-mes. Incluye 3,000 IOPS y 125 MB/s gratis.

### 4.3 Almacenamiento — S3

| Uso | Volumen estimado | Costo/mes |
|-----|-----------------|----------:|
| PDFs originales | 5 GB | $0.12 |
| Backups vectorstore | 2 GB | $0.05 |
| Logs archivados (S3 IA) | 10 GB | $0.13 |
| **Total S3** | | **$0.30** |

> S3 Standard: $0.023/GB-mes | S3 IA: $0.0125/GB-mes

### 4.4 Red y Balanceo

| Servicio | Descripción | Costo/mes |
|---------|-------------|----------:|
| Elastic IP (en uso) | IP fija para EC2 | $0.00 |
| Elastic IP (no asociada) | Penalización AWS | $3.65 |
| ALB - hora base | 730 hrs × $0.008 | $5.84 |
| ALB - LCU (100 usuarios/día) | ~0.5 LCU promedio | $2.92 |
| NAT Gateway (si VPC privada) | $0.045/hr + datos | $32.85 |
| CloudFront (frontend, 50 GB) | $0.085/GB | $4.25 |
| Transferencia datos salida | Primeros 100 GB gratis | $0.00 |

> **Recomendación:** Evitar NAT Gateway en MVP. Usar subnet pública con Security Groups estrictos.

### 4.5 DNS y Certificados

| Servicio | Costo/mes |
|---------|----------:|
| Route 53 — Hosted Zone | $0.50 |
| Route 53 — Consultas DNS (1M) | $0.40 |
| AWS Certificate Manager (SSL) | **Gratis** |
| **Total** | **$0.90** |

### 4.6 Monitoreo — CloudWatch

| Recurso | Costo/mes |
|---------|----------:|
| Métricas EC2 básicas (CPU, RAM, disco) | $0.00 (incluidas) |
| Métricas personalizadas (5 métricas) | $1.50 |
| Logs (FastAPI + app, 5 GB/mes) | $2.50 |
| Alarmas (5 alarmas) | $0.50 |
| Dashboard (1 dashboard) | $3.00 |
| **Total CloudWatch** | **$7.50** |

### 4.7 Seguridad (Opcional pero recomendado)

| Servicio | Costo/mes |
|---------|----------:|
| AWS WAF (1 Web ACL, reglas básicas) | $5.00 |
| AWS Secrets Manager (1 secreto: GOOGLE_API_KEY) | $0.40 |
| GuardDuty (detección amenazas) | $4.00 |
| **Total Seguridad** | **$9.40** |

---

## 5. Costos API Externa — Google Gemini

### Precios Gemini 2.5 Flash (sin thinking mode)

| Tipo | Precio |
|------|--------|
| Tokens de entrada | $0.15 / 1M tokens |
| Tokens de salida | $0.60 / 1M tokens |

### Estimación de tokens por operación

**Por cada roadmap generado:**
```
Prompt sistema + instrucciones formato:  ~500 tokens
Contexto ChromaDB (4 chunks × 250 tok):  ~1,000 tokens
Pregunta del usuario:                     ~50 tokens
─────────────────────────────────────────
Total INPUT:                             ~1,550 tokens

Roadmap JSON (8-12 pasos):               ~800 tokens
─────────────────────────────────────────
Total OUTPUT:                            ~800 tokens

Costo por query:
  Input:  1,550 × $0.15/1M = $0.000233
  Output:   800 × $0.60/1M = $0.000480
  ────────────────────────────────────
  TOTAL:                    ≈ $0.00071 (~$0.001/query)
```

### Costo mensual Gemini según volumen de uso

| Queries/día | Queries/mes | Costo Gemini/mes |
|------------:|------------:|-----------------:|
| 10 | 300 | $0.21 |
| 50 | 1,500 | $1.07 |
| 200 | 6,000 | $4.27 |
| 500 | 15,000 | $10.67 |
| 1,000 | 30,000 | $21.33 |
| 5,000 | 150,000 | $106.65 |

> El costo de Gemini es prácticamente despreciable hasta ~500 queries/día.
> Los embeddings (HuggingFace local) NO generan costo variable adicional.

---

## 6. Costos de Evaluación — RAGAS + LLM Judge

### Tokens consumidos por run completo (7 muestras del dataset)

#### RAGAS (4 métricas × 7 muestras)
```
Llamadas LLM internas RAGAS:
  - faithfulness:      ~3 llamadas × 7 muestras = 21 llamadas
  - answer_relevancy:  ~2 llamadas × 7 muestras = 14 llamadas
  - context_precision: ~3 llamadas × 7 muestras = 21 llamadas
  - context_recall:    ~3 llamadas × 7 muestras = 21 llamadas
  Total: ~77 llamadas internas

Tokens estimados por llamada:
  Input promedio:  ~1,800 tokens
  Output promedio: ~400 tokens

Total RAGAS:
  Input:  77 × 1,800 = 138,600 tokens × $0.15/1M = $0.021
  Output: 77 × 400  =  30,800 tokens × $0.60/1M = $0.018
  ──────────────────────────────────────────────────────
  Subtotal RAGAS:                                 $0.039
```

#### LLM Judge (1 llamada × 7 muestras)
```
Por muestra:
  Input:  pregunta + contexto + roadmap + ground truth ≈ 2,500 tokens
  Output: JSON con 5 criterios + justificaciones      ≈   600 tokens

Total Judge:
  Input:  7 × 2,500 = 17,500 tokens × $0.15/1M = $0.003
  Output: 7 ×   600 =  4,200 tokens × $0.60/1M = $0.003
  ─────────────────────────────────────────────────────
  Subtotal Judge:                                $0.006
```

#### Costo total por run de evaluación
```
  RAGAS:           $0.039
  LLM Judge:       $0.006
  ─────────────────────────
  Total por run:  ~$0.045
```

#### Costo mensual según frecuencia de evaluación

| Frecuencia | Runs/mes | Costo/mes |
|-----------|----------:|----------:|
| Una vez al mes | 1 | $0.05 |
| Semanal | 4 | $0.18 |
| Cada 2 días | 15 | $0.68 |
| Diaria | 30 | $1.35 |
| Con cada deploy (CI/CD, 10/mes) | 10 | $0.45 |

> **Recomendación:** Correr evaluación en cada merge a `main`. A ~10 deploys/mes = $0.45/mes.

---

## 7. Costo Total por Escenario

### Escenario 1 — MVP / Desarrollo

**Objetivo:** Probar el sistema, demos internas. ~50 queries/día.

| Componente | Servicio | Costo/mes |
|-----------|---------|----------:|
| Cómputo | EC2 t3.medium (on-demand) | $30.37 |
| Almacenamiento | EBS gp3 30GB | $2.40 |
| Backup | S3 5GB | $0.12 |
| IP Fija | Elastic IP | $0.00 |
| Monitoreo | CloudWatch básico | $0.00 |
| **Infraestructura AWS** | | **$32.89** |
| LLM Generación | Gemini (50 q/día) | $1.07 |
| Evaluaciones | RAGAS+Judge (semanal) | $0.18 |
| **Total APIs** | | **$1.25** |
| **TOTAL MENSUAL** | | **~$34** |
| **TOTAL ANUAL** | | **~$408** |

**Arquitectura:** Opción A (EC2 Simple). Sin dominio, sin ALB.

---

### Escenario 2 — Producción Pequeña (Equipo Interno)

**Objetivo:** Equipo de 10-30 personas. ~200 queries/día.

| Componente | Servicio | Costo/mes |
|-----------|---------|----------:|
| Cómputo | EC2 t3.large (on-demand) | $60.74 |
| Almacenamiento | EBS gp3 50GB | $4.00 |
| Backup | EBS Snapshot + S3 10GB | $1.80 |
| Red | ALB | $8.76 |
| DNS + SSL | Route 53 + ACM | $0.90 |
| Monitoreo | CloudWatch (logs + métricas) | $5.50 |
| Seguridad | Secrets Manager | $0.40 |
| **Infraestructura AWS** | | **$82.10** |
| LLM Generación | Gemini (200 q/día) | $4.27 |
| Evaluaciones | RAGAS+Judge (en cada deploy) | $0.45 |
| **Total APIs** | | **$4.72** |
| Dominio | (externo, ej. GoDaddy) | ~$1.00 |
| **TOTAL MENSUAL** | | **~$88** |
| **TOTAL ANUAL** | | **~$1,056** |
| **Con Reserved EC2 (1 año)** | | **~$65/mes → ~$780/año** |

**Arquitectura:** Opción B (EC2 + ALB).

---

### Escenario 3 — Producción Media (Empresarial)

**Objetivo:** Organización de 100+ usuarios. ~1,000 queries/día.

| Componente | Servicio | Costo/mes |
|-----------|---------|----------:|
| Cómputo | 2× EC2 t3.large (Reserved 1yr) | $75.92 |
| Almacenamiento | 2× EBS gp3 80GB | $12.80 |
| Base de datos | RDS PostgreSQL t3.medium (pgvector) | $52.56 |
| Frontend CDN | CloudFront (100GB) | $8.50 |
| Red | ALB | $8.76 |
| DNS + SSL | Route 53 + ACM | $0.90 |
| Monitoreo | CloudWatch completo | $10.00 |
| Seguridad | WAF + Secrets + GuardDuty | $9.40 |
| S3 | PDFs + backups + logs (50GB) | $1.15 |
| **Infraestructura AWS** | | **$179.99** |
| LLM Generación | Gemini (1,000 q/día) | $21.33 |
| Evaluaciones | RAGAS+Judge (diaria CI/CD) | $1.35 |
| Soporte AWS | Developer Support (mín.) | $29.00 |
| **Total APIs + Soporte** | | **$51.68** |
| **TOTAL MENSUAL** | | **~$232** |
| **TOTAL ANUAL** | | **~$2,784** |

**Arquitectura:** Opción D (Alta disponibilidad con RDS pgvector).
> Nota: Migrar de ChromaDB a pgvector requiere modificar `src/ingest.py` y `src/rag_engine.py`.

---

## 8. Comparativa de Arquitecturas

| | Opción A (EC2 Simple) | Opción B (EC2+ALB) | Opción C (Fargate) | Opción D (HA) |
|---|:---:|:---:|:---:|:---:|
| **Complejidad setup** | Baja | Media | Media | Alta |
| **Disponibilidad** | ~99.5% | ~99.9% | ~99.9% | ~99.99% |
| **Escalabilidad** | Manual | Manual | Automática | Automática |
| **HTTPS** | Manual (nginx) | Nativo (ACM) | Nativo | Nativo |
| **Costo mínimo** | ~$34/mes | ~$88/mes | ~$95/mes | ~$232/mes |
| **Gestión servidor** | Alta | Alta | Ninguna | Media |
| **Recomendado para** | MVP/Dev | Equipo pequeño | Sin DevOps | Empresa |

---

## 9. Optimizaciones para Reducir Costos

### Corto plazo (aplicar inmediatamente)

1. **Reserved Instances** — Comprometer 1 año en EC2 ahorra **38%**
   - t3.large: $60.74/mes → $37.96/mes = **-$272/año**

2. **Savings Plans** — Alternativa flexible a Reserved, ahorra ~35%

3. **Spot Instances para evaluaciones** — Los jobs de RAGAS/Judge no son críticos.
   Correrlos en EC2 Spot puede ahorrar **70%** en cómputo de evaluación.

4. **S3 Lifecycle Rules** — Logs >30 días a S3-IA, >90 días a S3 Glacier
   - Ahorro: ~60% en almacenamiento de logs

5. **Free Tier AWS** — Primer año incluye:
   - 750 hrs EC2 t2/t3.micro (no aplica para este proyecto por RAM)
   - 5 GB S3 Standard
   - 1M requests Lambda
   - 15 GB transferencia saliente

### Mediano plazo

6. **Compresión de vectores ChromaDB** — Reducir tamaño del vectorstore
   con `quantize_embeddings=True` (experimental en ChromaDB 1.4+)

7. **Caché de respuestas** — Añadir Redis ElastiCache (t3.micro ~$12/mes)
   para cachear roadmaps de preguntas frecuentes. Reduce llamadas a Gemini.
   Break-even: si >17 queries/día son repetidas.

8. **Modelo de embedding más ligero** — `all-MiniLM-L6-v2` (90MB) es buen balance.
   `paraphrase-MiniLM-L3-v2` (45MB) es más rápido pero menos preciso.

### Largo plazo

9. **Gemini API — Batch Mode** — Para evaluaciones RAGAS, usar
   `batchEmbedContents` reduce costos un 50% en procesamiento masivo.

10. **Auto Scaling basado en colas** — SQS + Lambda para picos de tráfico
    sin pagar por capacidad idle.

---

## 10. Estimación de Costos de Despliegue Inicial (One-time)

| Tarea | Tiempo estimado | Costo servicio |
|-------|----------------|---------------|
| Setup EC2 + EBS + Security Groups | 2-4 hrs | $0 |
| Configuración nginx + SSL | 1-2 hrs | $0 |
| Migración vectorstore a EBS | 30 min | $0 |
| Configuración CloudWatch | 1 hr | $0 |
| Configuración Route 53 + dominio | 1 hr | $0.50 (hosted zone) |
| Registro dominio (externo) | — | ~$12/año |
| **Total one-time** | **5-8 hrs** | **~$12.50** |

---

## 11. Recomendación Final

```
┌─────────────────────────────────────────────────────────────────┐
│  ETAPA 1 — HOY (MVP):                                           │
│  EC2 t3.medium + EBS 30GB = ~$34/mes                           │
│  Validar el producto con usuarios reales                        │
├─────────────────────────────────────────────────────────────────┤
│  ETAPA 2 — 3 MESES (Producción):                               │
│  EC2 t3.large Reserved + ALB + Route 53 = ~$65/mes             │
│  Añadir dominio, HTTPS, monitoreo, backups                      │
├─────────────────────────────────────────────────────────────────┤
│  ETAPA 3 — 6+ MESES (Escala):                                  │
│  2× t3.large + RDS pgvector + CloudFront = ~$232/mes           │
│  Solo si superas 500 queries/día o necesitas SLA empresarial    │
└─────────────────────────────────────────────────────────────────┘
```

**Resumen del costo real de operación mensual (Etapa 2):**
- AWS infraestructura: **~$65**
- Google Gemini API: **~$4-6** (200 queries/día)
- Evaluaciones RAGAS+Judge: **~$0.45**
- **Total operativo: ~$70/mes**

Este costo es significativamente bajo porque la parte más costosa computacionalmente
(embeddings con HuggingFace) corre localmente en el servidor sin costo variable adicional.

---

*Reporte generado para el proyecto NotionMap. Precios basados en AWS us-east-1 y Google AI Studio, Junio 2026. Los precios pueden variar. Verificar en [aws.amazon.com/pricing](https://aws.amazon.com/pricing) y [ai.google.dev/pricing](https://ai.google.dev/pricing).*
