# Costo MVP — NotionMap
**Equipo:** 10 personas | **Uso estimado:** ~20 queries/día | **Región:** us-east-1

---

## Resumen

| | |
|---|---:|
| AWS infraestructura | $32.89 / mes |
| Google Gemini API | $0.43 / mes |
| Evaluaciones RAGAS + Judge | $0.18 / mes |
| **TOTAL** | **~$34 / mes** |
| **TOTAL ANUAL** | **~$408 / año** |

---

## Desglose AWS

| Servicio | Detalle | $/mes |
|---------|---------|------:|
| EC2 t3.medium | 2 vCPU / 4 GB RAM — app + ChromaDB + embeddings | $30.37 |
| EBS gp3 30 GB | OS + app + vectorstore | $2.40 |
| S3 (5 GB) | PDFs originales + backups | $0.12 |
| Elastic IP | IP fija pública | $0.00 |
| CloudWatch básico | Métricas EC2 incluidas | $0.00 |
| **Total AWS** | | **$32.89** |

---

## Desglose APIs Externas

| Servicio | Cálculo | $/mes |
|---------|---------|------:|
| Gemini 2.5 Flash | 20 queries/día × 30 días × $0.00071/query | $0.43 |
| RAGAS + LLM Judge | 4 runs/mes × $0.045/run | $0.18 |
| **Total APIs** | | **$0.61** |

---

## Arquitectura MVP

```
Internet
    │
    ▼
[Elastic IP]  →  EC2 t3.medium  →  EBS 30 GB
                 Ubuntu 22.04       vectorstore/
                 FastAPI :8000      chroma_db/
                    │
                    ▼
                S3 Bucket
                (PDFs + backups)
```

Sin ALB · Sin dominio · Sin WAF · Acceso directo por IP

---

## Ahorro posible

| Optimización | Ahorro/mes |
|---|---:|
| Reserved EC2 t3.medium (1 año) | -$11.39 → **$19.00/mes** |
| Total con Reserved | **~$22/mes** |

---

> Precios AWS us-east-1, Junio 2026. Gemini: $0.15/1M tokens entrada · $0.60/1M tokens salida.
