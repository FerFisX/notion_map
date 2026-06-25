"""
dataset.py — Preguntas de prueba con ground truth y orden esperado de pasos.
Basado en los 3 PDFs cargados: llaves candidatas, Power Query, N8N con IA.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalSample:
    question:            str
    ground_truth:        str
    expected_keywords:   List[str]
    category:            str
    expected_step_order: List[str] = field(default_factory=list)
    # ^ pasos esperados en orden lógico (para validar secuencia)


EVAL_SAMPLES: List[EvalSample] = [

    # ── Llaves candidatas ─────────────────────────────────────────────────
    EvalSample(
        question="¿Qué es una llave candidata en bases de datos?",
        ground_truth=(
            "Una llave candidata es un atributo o conjunto mínimo de atributos "
            "que identifican de forma única cada tupla en una relación. "
            "Puede haber varias llaves candidatas y de ellas se elige la llave primaria."
        ),
        expected_keywords=["llave", "candidata", "única", "tupla", "relación", "primaria"],
        category="bases_de_datos",
        expected_step_order=[
            "Definir el concepto de llave",
            "Identificar atributos únicos",
            "Verificar minimalidad",
            "Seleccionar llave primaria",
        ],
    ),
    EvalSample(
        question="¿Cómo se identifican las llaves candidatas en una tabla?",
        ground_truth=(
            "Se deben encontrar todos los conjuntos de atributos que cumplan unicidad "
            "(no hay dos filas iguales) y minimalidad (no sobra ningún atributo). "
            "Se analiza la dependencia funcional entre columnas."
        ),
        expected_keywords=["unicidad", "minimalidad", "dependencia", "funcional", "atributos"],
        category="bases_de_datos",
        expected_step_order=[
            "Listar atributos de la tabla",
            "Verificar unicidad por atributo",
            "Verificar minimalidad",
            "Documentar llaves candidatas encontradas",
        ],
    ),
    EvalSample(
        question="¿Cuál es la diferencia entre llave primaria y llave candidata?",
        ground_truth=(
            "Todas las llaves primarias son llaves candidatas, pero no al revés. "
            "De todas las llaves candidatas se elige una como llave primaria. "
            "Las demás se llaman llaves alternativas."
        ),
        expected_keywords=["primaria", "candidata", "alternativa", "elegir", "diferencia"],
        category="bases_de_datos",
        expected_step_order=[
            "Definir llave candidata",
            "Definir llave primaria",
            "Comparar características",
            "Identificar llaves alternativas",
        ],
    ),

    # ── Power Query ───────────────────────────────────────────────────────
    EvalSample(
        question="¿Cuál es la complejidad en tiempo de las operaciones en Power Query?",
        ground_truth=(
            "Las operaciones básicas como filtros y selección de columnas son O(n). "
            "Los merge/join pueden ser O(n²) en el peor caso. "
            "Las operaciones de agrupación son O(n log n). "
            "El rendimiento depende del motor de evaluación lazy de Power Query."
        ),
        expected_keywords=["O(n)", "complejidad", "join", "merge", "agrupación", "lazy"],
        category="power_query",
        expected_step_order=[
            "Entender el modelo de evaluación lazy",
            "Analizar complejidad de filtrado",
            "Analizar complejidad de merge/join",
            "Analizar complejidad de agrupación",
            "Aplicar optimizaciones",
        ],
    ),
    EvalSample(
        question="¿Cómo optimizar una consulta lenta en Power Query?",
        ground_truth=(
            "Aplicar filtros y eliminar columnas lo antes posible en el pipeline. "
            "Evitar columnas personalizadas con lógica compleja. "
            "Usar tipos de datos correctos. Deshabilitar la carga de tablas intermedias."
        ),
        expected_keywords=["filtros", "pipeline", "columnas", "optimizar", "tipos", "carga"],
        category="power_query",
        expected_step_order=[
            "Diagnosticar consulta con el Performance Analyzer",
            "Aplicar filtros al inicio del pipeline",
            "Eliminar columnas innecesarias",
            "Revisar y corregir tipos de datos",
            "Deshabilitar cargas de tablas intermedias",
            "Medir y comparar mejora",
        ],
    ),

    # ── N8N con agentes IA ─────────────────────────────────────────────────
    EvalSample(
        question="¿Cómo crear un flujo con agentes IA en N8N?",
        ground_truth=(
            "En N8N se crea un workflow con nodos. Para agentes IA se usa el nodo AI Agent "
            "conectado a un LLM (OpenAI, Anthropic). Se definen herramientas disponibles "
            "para el agente y se configura la memoria conversacional si se necesita."
        ),
        expected_keywords=["workflow", "nodo", "AI Agent", "LLM", "herramientas", "memoria"],
        category="n8n_ia",
        expected_step_order=[
            "Crear nuevo workflow en N8N",
            "Agregar nodo Trigger",
            "Agregar nodo AI Agent",
            "Configurar credenciales del LLM",
            "Definir herramientas del agente",
            "Configurar memoria conversacional",
            "Activar y probar el flujo",
        ],
    ),
    EvalSample(
        question="¿Cómo testear un flujo de agente IA en N8N?",
        ground_truth=(
            "Se usa el modo de ejecución manual para probar paso a paso. "
            "Se revisan los inputs y outputs de cada nodo. "
            "Se pueden ver los logs de ejecución en tiempo real. "
            "Se prueban casos borde y mensajes inesperados."
        ),
        expected_keywords=["manual", "nodo", "logs", "ejecución", "testear", "casos"],
        category="n8n_ia",
        expected_step_order=[
            "Activar modo de ejecución manual",
            "Ejecutar el workflow con datos de prueba",
            "Revisar output de cada nodo",
            "Revisar logs de ejecución",
            "Probar casos borde",
            "Documentar resultados",
        ],
    ),
]
