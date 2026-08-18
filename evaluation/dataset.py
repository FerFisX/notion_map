"""
dataset.py — Preguntas de prueba con ground truth y orden esperado de pasos.
Basado en los 3 PDFs cargados: llaves candidatas, Power Query, N8N con IA.
"""

from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class EvalSample:
    question:            str
    ground_truth:        str
    expected_keywords:   List[str]
    category:            str
    expected_step_order: List[str] = field(default_factory=list)
    # ^ pasos esperados en orden lógico (para validar secuencia)
    expected_elements: List[dict[str, Any]] = field(default_factory=list)
    # ^ universo opcional y estable para evaluar cobertura con Completeness
    requires_web: bool = False
    # ^ el benchmark de generación debe verificar uso real de contexto web


EVAL_SAMPLES: List[EvalSample] = [

    # Llaves candidatas
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

    # Power Query
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

    # N8N con agentes IA
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

    # Llaves candidatas (ampliación)
    EvalSample(
        question="¿Qué propiedades debe cumplir una llave candidata?",
        ground_truth=(
            "Una llave candidata debe cumplir unicidad: no puede haber dos tuplas con "
            "los mismos valores en los atributos de la llave. Y minimalidad: ningún "
            "atributo puede eliminarse sin perder la propiedad de unicidad."
        ),
        expected_keywords=["unicidad", "minimalidad", "atributos", "tupla", "única"],
        category="bases_de_datos",
        expected_step_order=[
            "Definir la propiedad de unicidad",
            "Definir la propiedad de minimalidad",
            "Verificar unicidad en los atributos",
            "Verificar minimalidad del conjunto",
        ],
    ),
    EvalSample(
        question="¿Puede una tabla tener más de una llave candidata?",
        ground_truth=(
            "Sí, una tabla puede tener varias llaves candidatas. Cada una identifica "
            "de forma única las filas. De entre todas se elige una como llave primaria "
            "y las restantes quedan como llaves alternativas."
        ),
        expected_keywords=["varias", "candidatas", "primaria", "alternativas", "única"],
        category="bases_de_datos",
        expected_step_order=[
            "Identificar todos los conjuntos únicos y mínimos",
            "Listar las llaves candidatas",
            "Elegir la llave primaria",
            "Marcar las demás como alternativas",
        ],
    ),

    # Power Query (ampliación)
    EvalSample(
        question="¿Qué es la evaluación lazy en Power Query?",
        ground_truth=(
            "La evaluación lazy significa que Power Query no ejecuta las transformaciones "
            "hasta que se necesita el resultado. Esto permite optimizar el plan de ejecución "
            "y aplicar query folding delegando operaciones al origen de datos."
        ),
        expected_keywords=["lazy", "evaluación", "transformaciones", "query folding", "origen"],
        category="power_query",
        expected_step_order=[
            "Entender qué es la evaluación lazy",
            "Identificar cuándo se materializa el resultado",
            "Aprovechar el query folding",
            "Validar el plan de ejecución",
        ],
    ),
    EvalSample(
        question="¿Por qué un merge en Power Query puede ser lento?",
        ground_truth=(
            "Un merge compara filas entre dos tablas y en el peor caso tiene complejidad "
            "cuadrática O(n²). Es lento cuando las tablas son grandes, cuando no hay query "
            "folding y cuando se hace antes de filtrar y reducir columnas."
        ),
        expected_keywords=["merge", "join", "O(n²)", "cuadrática", "folding", "filtrar"],
        category="power_query",
        expected_step_order=[
            "Entender el costo de comparar filas",
            "Reducir filas antes del merge",
            "Eliminar columnas innecesarias",
            "Verificar si hay query folding",
            "Medir el impacto",
        ],
    ),

    # N8N con agentes IA (ampliación)
    EvalSample(
        question="¿Qué nodos se necesitan para un agente IA en N8N?",
        ground_truth=(
            "Se necesita un nodo Trigger para iniciar el flujo, el nodo AI Agent como "
            "núcleo, un nodo de modelo de lenguaje (Chat Model) conectado al agente, y "
            "opcionalmente nodos de herramientas y de memoria."
        ),
        expected_keywords=["Trigger", "AI Agent", "Chat Model", "herramientas", "memoria"],
        category="n8n_ia",
        expected_step_order=[
            "Agregar nodo Trigger",
            "Agregar nodo AI Agent",
            "Conectar el nodo Chat Model",
            "Agregar nodos de herramientas",
            "Agregar nodo de memoria",
        ],
    ),
    EvalSample(
        question="¿Cómo se conecta un LLM a un agente en N8N?",
        ground_truth=(
            "Se agrega un nodo Chat Model (por ejemplo OpenAI o Anthropic), se configuran "
            "las credenciales de la API y se conecta la salida del modelo a la entrada "
            "de Chat Model del nodo AI Agent."
        ),
        expected_keywords=["Chat Model", "credenciales", "API", "AI Agent", "conectar"],
        category="n8n_ia",
        expected_step_order=[
            "Agregar el nodo Chat Model",
            "Configurar las credenciales de la API",
            "Seleccionar el modelo a usar",
            "Conectar el modelo al nodo AI Agent",
            "Probar la conexión",
        ],
    ),
    EvalSample(
        question="¿Cómo se agrega memoria conversacional a un agente en N8N?",
        ground_truth=(
            "Se añade un nodo de memoria (por ejemplo Window Buffer Memory) y se conecta "
            "a la entrada de memoria del nodo AI Agent. Esto permite que el agente recuerde "
            "los mensajes anteriores de la conversación."
        ),
        expected_keywords=["memoria", "Window Buffer", "AI Agent", "conversación", "mensajes"],
        category="n8n_ia",
        expected_step_order=[
            "Agregar el nodo de memoria",
            "Configurar el tamaño de la ventana",
            "Conectar la memoria al nodo AI Agent",
            "Probar que el agente recuerda el contexto",
        ],
    ),

    # Muestra representativa para generación real de roadmaps
    EvalSample(
        question=(
            "¿Cómo puedo aprender e implementar Time Intelligence en DAX para "
            "comparar ventas entre diferentes periodos?"
        ),
        ground_truth=(
            "Definir el objetivo de comparación temporal, preparar una tabla calendario "
            "continua y marcada, relacionarla con ventas, validar una medida base, crear "
            "medidas para periodos comparables y comprobarlas con resultados conocidos."
        ),
        expected_keywords=[
            "Time Intelligence", "DAX", "tabla calendario", "medida base",
            "periodo anterior", "validación",
        ],
        category="dax_time_intelligence",
        expected_step_order=[
            "Definir el objetivo de comparación temporal",
            "Preparar la tabla calendario y sus relaciones",
            "Crear y validar la medida base",
            "Crear las medidas de comparación temporal",
            "Validar los resultados con periodos conocidos",
        ],
        expected_elements=[
            {"name": "Objetivo de comparación temporal", "importance": "important"},
            {"name": "Tabla calendario continua, marcada y relacionada", "importance": "critical"},
            {"name": "Medida base validada", "importance": "critical"},
            {"name": "Medidas para periodo anterior y variación", "importance": "critical"},
            {"name": "Validación con periodos conocidos", "importance": "critical"},
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo documentar un sistema utilizando niveles de abstracción "
            "conceptual, lógico y de implementación?"
        ),
        ground_truth=(
            "Definir el objetivo y la audiencia, describir capacidades y conceptos del "
            "dominio, representar componentes y contratos lógicos, documentar tecnologías "
            "y configuraciones de implementación, y validar la trazabilidad entre niveles."
        ),
        expected_keywords=[
            "conceptual", "lógico", "implementación", "trazabilidad",
            "audiencia", "contratos",
        ],
        category="niveles_abstraccion",
        expected_step_order=[
            "Definir el objetivo y la audiencia",
            "Documentar el nivel conceptual",
            "Documentar el nivel lógico",
            "Documentar el nivel de implementación",
            "Validar la trazabilidad y consistencia",
        ],
        expected_elements=[
            {"name": "Objetivo y audiencia de la documentación", "importance": "important"},
            {"name": "Nivel conceptual con dominio y capacidades", "importance": "critical"},
            {"name": "Nivel lógico con componentes, relaciones y contratos", "importance": "critical"},
            {"name": "Nivel de implementación con tecnologías y configuraciones", "importance": "critical"},
            {"name": "Trazabilidad y consistencia entre niveles", "importance": "critical"},
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo diseñar y evaluar prompts para mejorar la calidad de las "
            "respuestas de un sistema RAG?"
        ),
        ground_truth=(
            "Definir la tarea y el contrato de salida, separar instrucciones, pregunta y "
            "contexto recuperado, exigir respuestas sustentadas y manejo de evidencia "
            "insuficiente, construir casos de prueba y mejorar el prompt con métricas."
        ),
        expected_keywords=[
            "prompt", "RAG", "contexto recuperado", "grounding",
            "formato de salida", "casos de prueba",
        ],
        category="prompt_engineering_rag",
        expected_step_order=[
            "Definir la tarea y el contrato de salida",
            "Estructurar instrucciones, pregunta y contexto",
            "Definir reglas de grounding",
            "Crear casos de prueba representativos",
            "Evaluar resultados e iterar el prompt",
        ],
        expected_elements=[
            {"name": "Tarea y contrato de salida explícitos", "importance": "critical"},
            {"name": "Separación entre instrucciones, pregunta y contexto", "importance": "critical"},
            {"name": "Reglas de grounding y contexto insuficiente", "importance": "critical"},
            {"name": "Casos de prueba representativos", "importance": "critical"},
            {"name": "Evaluación con métricas e iteración", "importance": "important"},
        ],
    ),

    # Casos actuales para validar el fallback web de generación
    EvalSample(
        question=(
            "¿Cómo puedo actualizar una integración tras el lanzamiento de GPT-5.6 "
            "del 9 de julio de 2026 para elegir entre gpt-5.6-sol, gpt-5.6-terra y "
            "gpt-5.6-luna, configurar reasoning effort y comparar sus precios actuales?"
        ),
        ground_truth=(
            "Verificar la documentación vigente y los IDs disponibles, comparar el "
            "alcance y precio actual de Sol, Terra y Luna, configurar reasoning effort "
            "según la complejidad, y validar calidad, latencia y consumo real de tokens "
            "antes de actualizar la integración."
        ),
        expected_keywords=[
            "GPT-5.6 Sol", "GPT-5.6 Terra", "GPT-5.6 Luna", "latencia",
            "costo", "tokens", "evaluación",
        ],
        category="openai_current_models",
        expected_step_order=[
            "Verificar disponibilidad e IDs vigentes",
            "Comparar Sol, Terra y Luna",
            "Configurar reasoning effort",
            "Estimar costo y latencia actuales",
            "Probar y actualizar la integración",
        ],
        requires_web=True,
    ),
    EvalSample(
        question=(
            "¿Cómo puedo migrar una aplicación desde GPT-5.4 o GPT-5.5 hacia "
            "GPT-5.6 y configurar reasoning effort, prompt caching y Responses API?"
        ),
        ground_truth=(
            "Inventariar el modelo y parámetros actuales, seleccionar el tier GPT-5.6, "
            "migrar mediante Responses API, configurar reasoning effort de forma "
            "explícita, conservar prefijos estables para prompt caching y comparar "
            "calidad, latencia, tokens y costo antes del despliegue."
        ),
        expected_keywords=[
            "GPT-5.6", "Responses API", "reasoning effort", "prompt caching",
            "evaluación", "latencia", "tokens",
        ],
        category="openai_current_models",
        expected_step_order=[
            "Inventariar la configuración actual",
            "Seleccionar el modelo GPT-5.6",
            "Migrar a Responses API",
            "Configurar reasoning y caching",
            "Evaluar y desplegar gradualmente",
        ],
        requires_web=True,
    ),
]
