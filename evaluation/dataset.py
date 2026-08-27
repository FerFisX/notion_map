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
            "¿Cómo puedo comparar, usando la documentación oficial vigente, los modelos "
            "Anthropic disponibles en Amazon Bedrock, sus capacidades, límites y precios "
            "para elegir uno para un sistema RAG?"
        ),
        ground_truth=(
            "Consultar el catálogo, las páginas de precios y las cuotas oficiales de "
            "Amazon Bedrock para identificar modelos Anthropic e inference profiles "
            "disponibles en la región; comparar contexto, capacidades, límites, precio, "
            "latencia y calidad mediante un benchmark RAG antes de seleccionar el modelo."
        ),
        expected_keywords=[
            "Amazon Bedrock", "Anthropic", "modelo", "precio", "límites",
            "latencia", "RAG",
        ],
        category="bedrock_current_models",
        expected_step_order=[
            "Definir requisitos del sistema RAG",
            "Consultar catálogo y disponibilidad vigentes",
            "Comparar capacidades, límites y precios",
            "Ejecutar un benchmark común",
            "Seleccionar y documentar el modelo",
        ],
        requires_web=True,
    ),
    EvalSample(
        question=(
            "¿Cómo puedo comparar, usando documentación oficial vigente, el acceso a "
            "modelos Claude mediante Amazon Bedrock y la API de Anthropic considerando "
            "modelos disponibles, regiones, precios, autenticación y límites?"
        ),
        ground_truth=(
            "Definir los requisitos de integración y consultar documentación y precios "
            "vigentes de AWS y Anthropic; comparar modelos, disponibilidad regional, "
            "autenticación, cuotas, límites, costos y operación mediante una prueba "
            "equivalente antes de elegir el canal de acceso a Claude."
        ),
        expected_keywords=[
            "Claude", "Amazon Bedrock", "Anthropic API", "regiones", "precios",
            "autenticación", "límites",
        ],
        category="llm_provider_current",
        expected_step_order=[
            "Definir requisitos de integración",
            "Consultar documentación oficial vigente",
            "Comparar modelos, regiones y autenticación",
            "Comparar precios, cuotas y límites",
            "Probar y seleccionar el proveedor",
        ],
        requires_web=True,
    ),

    # Ampliación del benchmark de generación: Power BI y DAX
    EvalSample(
        question=(
            "¿Cómo puedo crear una tabla calendario en Power BI que soporte años "
            "fiscales, semanas ISO y comparaciones de Time Intelligence en DAX?"
        ),
        ground_truth=(
            "Definir el rango de fechas, crear una tabla calendario continua, agregar "
            "atributos fiscales e ISO, marcarla como tabla de fechas, relacionarla con "
            "el modelo y validar medidas temporales."
        ),
        expected_keywords=[
            "tabla calendario", "año fiscal", "semana ISO", "DAX", "relación",
        ],
        category="dax_time_intelligence",
        expected_step_order=[
            "Definir los requisitos temporales",
            "Crear la tabla calendario continua",
            "Agregar atributos fiscales e ISO",
            "Marcar y relacionar la tabla",
            "Validar medidas de Time Intelligence",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo construir y validar medidas YTD, QTD y MTD en DAX sin "
            "obtener resultados incorrectos cuando faltan fechas?"
        ),
        ground_truth=(
            "Preparar una tabla calendario continua, validar relaciones y medida base, "
            "crear medidas YTD, QTD y MTD, controlar periodos incompletos y comprobar "
            "resultados contra datos conocidos."
        ),
        expected_keywords=["YTD", "QTD", "MTD", "calendario", "periodos", "validación"],
        category="dax_time_intelligence",
        expected_step_order=[
            "Validar la tabla calendario",
            "Validar la medida base",
            "Crear medidas acumuladas",
            "Controlar fechas faltantes",
            "Comprobar resultados",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo decidir entre DATEADD y SAMEPERIODLASTYEAR para comparar "
            "periodos en un modelo de Power BI?"
        ),
        ground_truth=(
            "Comprender el contexto de filtro y los requisitos de comparación, revisar "
            "las diferencias entre ambas funciones, probar calendarios y granularidades "
            "reales, y elegir según el comportamiento validado."
        ),
        expected_keywords=[
            "DATEADD", "SAMEPERIODLASTYEAR", "contexto de filtro", "calendario",
        ],
        category="dax_time_intelligence",
        expected_step_order=[
            "Definir el escenario temporal",
            "Revisar el contexto de filtro",
            "Comparar las funciones",
            "Probar casos representativos",
            "Seleccionar y documentar la función",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo diagnosticar una medida de Time Intelligence en DAX que "
            "funciona por mes pero muestra un total anual incorrecto?"
        ),
        ground_truth=(
            "Reproducir el error, inspeccionar la medida base y el contexto de filtro, "
            "validar calendario y relaciones, aislar la lógica de agregación y comprobar "
            "la corrección en diferentes niveles de granularidad."
        ),
        expected_keywords=[
            "contexto de filtro", "total", "granularidad", "medida base", "calendario",
        ],
        category="dax_time_intelligence",
        expected_step_order=[
            "Reproducir el resultado incorrecto",
            "Validar calendario y relaciones",
            "Inspeccionar contexto y medida base",
            "Corregir la lógica del total",
            "Validar todas las granularidades",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo implementar incremental refresh en Power BI conservando "
            "query folding y verificando que realmente reduzca el tiempo de actualización?"
        ),
        ground_truth=(
            "Definir la ventana de retención y actualización, crear parámetros de fecha, "
            "aplicar filtros plegables, configurar la política incremental, publicar y "
            "medir particiones y tiempos de actualización."
        ),
        expected_keywords=[
            "incremental refresh", "query folding", "parámetros", "particiones", "medición",
        ],
        category="power_bi",
        expected_step_order=[
            "Definir la política de actualización",
            "Crear parámetros de fecha",
            "Preservar query folding",
            "Configurar y publicar la política",
            "Medir el resultado",
        ],
    ),

    # Ampliación del benchmark de generación: niveles de abstracción
    EvalSample(
        question=(
            "¿Cómo puedo transformar un requerimiento de negocio en modelos conceptual, "
            "lógico y físico sin perder trazabilidad entre niveles?"
        ),
        ground_truth=(
            "Definir objetivos y lenguaje de negocio, modelar conceptos y relaciones, "
            "traducirlos a componentes lógicos, decidir implementaciones físicas y mantener "
            "identificadores y revisiones de trazabilidad."
        ),
        expected_keywords=[
            "requerimiento", "conceptual", "lógico", "físico", "trazabilidad",
        ],
        category="niveles_abstraccion",
        expected_step_order=[
            "Definir el requerimiento de negocio",
            "Construir el modelo conceptual",
            "Derivar el modelo lógico",
            "Diseñar el modelo físico",
            "Validar la trazabilidad",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo mantener consistencia y trazabilidad cuando cambia un requisito "
            "que afecta varios niveles de abstracción de un sistema?"
        ),
        ground_truth=(
            "Registrar el cambio, analizar impactos desde el nivel conceptual hasta la "
            "implementación, actualizar artefactos vinculados, validar consistencia y "
            "obtener aprobación de los responsables."
        ),
        expected_keywords=["cambio", "impacto", "trazabilidad", "consistencia", "artefactos"],
        category="niveles_abstraccion",
        expected_step_order=[
            "Registrar el cambio",
            "Analizar el impacto entre niveles",
            "Actualizar artefactos relacionados",
            "Validar consistencia",
            "Aprobar y comunicar el cambio",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo usar el modelo C4 para documentar un sistema sin mezclar "
            "decisiones de contexto, contenedores, componentes y código?"
        ),
        ground_truth=(
            "Definir audiencia y alcance, crear diagramas de contexto, contenedores y "
            "componentes de forma progresiva, documentar código solo cuando aporte valor "
            "y validar que cada vista mantenga su nivel de abstracción."
        ),
        expected_keywords=["C4", "contexto", "contenedores", "componentes", "código"],
        category="niveles_abstraccion",
        expected_step_order=[
            "Definir audiencia y alcance",
            "Crear la vista de contexto",
            "Crear la vista de contenedores",
            "Detallar componentes relevantes",
            "Validar separación entre niveles",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo revisar una documentación técnica para detectar conceptos que "
            "están descritos en un nivel de abstracción incorrecto?"
        ),
        ground_truth=(
            "Definir criterios para cada nivel, inventariar afirmaciones y diagramas, "
            "clasificarlos según su propósito, detectar mezclas, reubicar detalles y "
            "validar la lectura con cada audiencia."
        ),
        expected_keywords=["audiencia", "nivel", "conceptos", "detalles", "revisión"],
        category="niveles_abstraccion",
        expected_step_order=[
            "Definir criterios por nivel",
            "Inventariar el contenido",
            "Clasificar conceptos y detalles",
            "Corregir mezclas de abstracción",
            "Validar con las audiencias",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo diseñar y documentar una API desde su capacidad de negocio "
            "hasta sus contratos lógicos y detalles de despliegue?"
        ),
        ground_truth=(
            "Definir la capacidad y consumidores, modelar recursos y operaciones, diseñar "
            "contratos lógicos y errores, especificar seguridad e implementación, y "
            "mantener trazabilidad entre la necesidad y el despliegue."
        ),
        expected_keywords=["API", "capacidad", "contratos", "seguridad", "despliegue"],
        category="niveles_abstraccion",
        expected_step_order=[
            "Definir capacidad y consumidores",
            "Modelar recursos y operaciones",
            "Diseñar contratos lógicos",
            "Definir seguridad e implementación",
            "Validar trazabilidad y despliegue",
        ],
    ),

    # Ampliación del benchmark de generación: Prompt Engineering y RAG
    EvalSample(
        question=(
            "¿Cómo puedo diseñar un prompt de RAG que cite evidencia y reconozca "
            "explícitamente cuando el contexto recuperado es insuficiente?"
        ),
        ground_truth=(
            "Definir el contrato de respuesta, separar contexto e instrucciones, exigir "
            "citas vinculadas a evidencia, establecer una respuesta de insuficiencia y "
            "probar casos con evidencia completa, parcial y ausente."
        ),
        expected_keywords=["RAG", "evidencia", "citas", "contexto insuficiente", "pruebas"],
        category="prompt_engineering_rag",
        expected_step_order=[
            "Definir el contrato de respuesta",
            "Estructurar contexto e instrucciones",
            "Agregar reglas de citación",
            "Definir manejo de evidencia insuficiente",
            "Probar escenarios de cobertura",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo crear un dataset de evaluación para comparar versiones de prompts "
            "en un sistema RAG sin sobreajustarlo a unos pocos ejemplos?"
        ),
        ground_truth=(
            "Definir objetivos y segmentos, muestrear preguntas representativas y casos "
            "difíciles, separar desarrollo y prueba, registrar referencias y criterios, "
            "y comparar versiones con métricas y revisión humana."
        ),
        expected_keywords=["dataset", "segmentos", "casos difíciles", "métricas", "revisión"],
        category="prompt_engineering_rag",
        expected_step_order=[
            "Definir objetivos y segmentos",
            "Recolectar casos representativos",
            "Separar desarrollo y prueba",
            "Definir referencias y criterios",
            "Comparar versiones del prompt",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo proteger un sistema RAG contra prompt injection contenida "
            "en documentos recuperados sin impedir respuestas útiles?"
        ),
        ground_truth=(
            "Establecer jerarquía de instrucciones, tratar documentos como datos no "
            "confiables, delimitar contexto, restringir herramientas y datos sensibles, "
            "probar ataques y monitorear incidentes."
        ),
        expected_keywords=[
            "prompt injection", "instrucciones", "contexto", "herramientas", "pruebas",
        ],
        category="prompt_engineering_rag",
        expected_step_order=[
            "Definir el modelo de amenaza",
            "Separar instrucciones y documentos",
            "Restringir herramientas y datos",
            "Crear pruebas adversariales",
            "Monitorear y mejorar defensas",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo implementar query rewriting en un RAG preservando la intención "
            "original y evitando consultas refinadas demasiado amplias?"
        ),
        ground_truth=(
            "Capturar intención y restricciones, generar una reescritura acotada, conservar "
            "la consulta original, comparar recuperación con ambas versiones, aplicar "
            "umbrales y evaluar casos de deriva."
        ),
        expected_keywords=["query rewriting", "intención", "restricciones", "deriva", "retrieval"],
        category="prompt_engineering_rag",
        expected_step_order=[
            "Capturar intención y restricciones",
            "Generar una reescritura acotada",
            "Conservar la consulta original",
            "Comparar la recuperación",
            "Evaluar y controlar la deriva",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo priorizar y comprimir contexto recuperado para un RAG cuando "
            "los documentos relevantes superan la ventana disponible del modelo?"
        ),
        ground_truth=(
            "Medir presupuesto de contexto, recuperar y rerankear fragmentos, eliminar "
            "redundancia, resumir conservando evidencia, asignar espacio a instrucciones "
            "y respuesta, y validar pérdida de información."
        ),
        expected_keywords=["contexto", "reranking", "redundancia", "resumen", "tokens"],
        category="prompt_engineering_rag",
        expected_step_order=[
            "Definir el presupuesto de contexto",
            "Recuperar y rerankear fragmentos",
            "Eliminar redundancia",
            "Comprimir preservando evidencia",
            "Evaluar la pérdida de información",
        ],
    ),

    # Ampliación del benchmark de generación: n8n y bases de datos
    EvalSample(
        question=(
            "¿Cómo puedo agregar aprobación humana a un workflow de n8n antes de que "
            "un agente de IA ejecute una acción sensible?"
        ),
        ground_truth=(
            "Identificar acciones sensibles, pausar el workflow, enviar una solicitud con "
            "contexto suficiente, validar identidad y decisión, continuar o cancelar de "
            "forma segura y conservar auditoría."
        ),
        expected_keywords=["n8n", "aprobación", "pausa", "seguridad", "auditoría"],
        category="n8n_ia",
        expected_step_order=[
            "Identificar acciones sensibles",
            "Insertar el punto de aprobación",
            "Solicitar y validar la decisión",
            "Continuar o cancelar con seguridad",
            "Registrar la auditoría",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo diseñar retries, manejo de errores e idempotencia en un workflow "
            "de n8n que llama APIs externas?"
        ),
        ground_truth=(
            "Clasificar errores transitorios y permanentes, configurar backoff y límites, "
            "usar claves idempotentes, manejar ramas de error, persistir estado y probar "
            "repeticiones sin duplicar efectos."
        ),
        expected_keywords=["retry", "backoff", "idempotencia", "errores", "APIs"],
        category="n8n_ia",
        expected_step_order=[
            "Clasificar los errores",
            "Definir política de retries",
            "Implementar idempotencia",
            "Diseñar ramas de recuperación",
            "Probar fallos y duplicados",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo configurar memoria y herramientas en un agente de n8n evitando "
            "que datos de una conversación se filtren a otra?"
        ),
        ground_truth=(
            "Definir límites de sesión, usar identificadores aislados, restringir datos "
            "guardados y herramientas, aplicar expiración y permisos, y probar separación "
            "entre usuarios y conversaciones."
        ),
        expected_keywords=["memoria", "sesión", "aislamiento", "herramientas", "permisos"],
        category="n8n_ia",
        expected_step_order=[
            "Definir límites de sesión",
            "Configurar identificadores aislados",
            "Restringir memoria y herramientas",
            "Aplicar expiración y permisos",
            "Probar aislamiento entre conversaciones",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo identificar llaves candidatas compuestas a partir de datos, "
            "restricciones y dependencias funcionales de una tabla?"
        ),
        ground_truth=(
            "Inventariar atributos y restricciones, obtener cierres de atributos, buscar "
            "superllaves, eliminar atributos redundantes para comprobar minimalidad y "
            "validar candidatos con datos y reglas del negocio."
        ),
        expected_keywords=[
            "llave compuesta", "dependencia funcional", "cierre", "unicidad", "minimalidad",
        ],
        category="bases_de_datos",
        expected_step_order=[
            "Inventariar atributos y restricciones",
            "Calcular cierres de atributos",
            "Identificar superllaves",
            "Comprobar minimalidad",
            "Validar llaves candidatas",
        ],
    ),
    EvalSample(
        question=(
            "¿Cómo puedo normalizar un esquema hasta tercera forma normal preservando "
            "sus llaves candidatas y dependencias funcionales importantes?"
        ),
        ground_truth=(
            "Documentar dependencias y candidatos, verificar primera y segunda forma "
            "normal, eliminar dependencias transitivas, comprobar unión sin pérdida y "
            "preservación de dependencias, y validar el esquema resultante."
        ),
        expected_keywords=["3FN", "llaves candidatas", "dependencias", "unión", "normalización"],
        category="bases_de_datos",
        expected_step_order=[
            "Documentar dependencias y llaves",
            "Verificar primera forma normal",
            "Resolver dependencias parciales",
            "Resolver dependencias transitivas",
            "Validar la descomposición",
        ],
    ),

    # Casos temporales para comprobar el uso de contexto web vigente
    EvalSample(
        question=(
            "¿Cómo puedo comparar las licencias, capacidades y precios vigentes de Power BI "
            "para seleccionar una opción de despliegue empresarial?"
        ),
        ground_truth=(
            "Consultar documentación y precios oficiales vigentes, definir usuarios, "
            "capacidad y distribución, comparar alternativas y restricciones, estimar "
            "costo total y validar la decisión con una prueba representativa."
        ),
        expected_keywords=["Power BI", "licencias", "capacidad", "precio", "despliegue"],
        category="power_bi_current",
        expected_step_order=[
            "Definir requisitos de despliegue",
            "Consultar opciones y precios vigentes",
            "Comparar capacidades y restricciones",
            "Estimar el costo total",
            "Validar y documentar la decisión",
        ],
        requires_web=True,
    ),
    EvalSample(
        question=(
            "¿Cómo puedo revisar la actualización más reciente de Power BI Desktop y "
            "decidir qué cambios debo probar antes de adoptarla en producción?"
        ),
        ground_truth=(
            "Consultar las notas oficiales más recientes, identificar cambios relevantes "
            "y deprecaciones, analizar impacto en modelos y reportes, probar compatibilidad "
            "en un entorno controlado y planificar despliegue o rollback."
        ),
        expected_keywords=["Power BI Desktop", "actualización", "compatibilidad", "pruebas"],
        category="power_bi_current",
        expected_step_order=[
            "Consultar notas oficiales vigentes",
            "Identificar cambios relevantes",
            "Analizar impacto y riesgos",
            "Probar en un entorno controlado",
            "Planificar adopción o rollback",
        ],
        requires_web=True,
    ),
    EvalSample(
        question=(
            "¿Cómo puedo verificar los nodos y capacidades de agentes de IA disponibles "
            "en la versión actual de n8n antes de actualizar un workflow existente?"
        ),
        ground_truth=(
            "Verificar versión instalada y documentación vigente, revisar cambios de nodos, "
            "credenciales y memoria, identificar breaking changes, clonar y probar el flujo, "
            "y preparar migración y rollback."
        ),
        expected_keywords=["n8n", "versión", "AI Agent", "breaking changes", "migración"],
        category="n8n_current",
        expected_step_order=[
            "Inventariar versión y workflow actuales",
            "Consultar capacidades vigentes",
            "Identificar cambios incompatibles",
            "Probar una copia del workflow",
            "Planificar actualización y rollback",
        ],
        requires_web=True,
    ),
    EvalSample(
        question=(
            "¿Cómo puedo comparar modelos LLM alojados disponibles actualmente para un RAG "
            "considerando precio, latencia, contexto, grounding y privacidad?"
        ),
        ground_truth=(
            "Definir carga y requisitos, consultar especificaciones y precios vigentes, "
            "preseleccionar modelos, ejecutar un benchmark común de calidad y rendimiento, "
            "evaluar privacidad y calcular costo total antes de decidir."
        ),
        expected_keywords=["LLM", "precio", "latencia", "contexto", "privacidad", "benchmark"],
        category="llm_current_models",
        expected_step_order=[
            "Definir requisitos del RAG",
            "Consultar modelos y precios vigentes",
            "Diseñar un benchmark común",
            "Evaluar calidad, rendimiento y privacidad",
            "Seleccionar y documentar el modelo",
        ],
        requires_web=True,
    ),
    EvalSample(
        question=(
            "¿Cómo puedo actualizar una integración de OpenAI usando la documentación "
            "vigente para seleccionar modelo, Responses API, reasoning y prompt caching?"
        ),
        ground_truth=(
            "Revisar documentación y modelos disponibles, inventariar la integración, "
            "seleccionar modelo y parámetros compatibles, migrar a Responses API, preservar "
            "prefijos cacheables, probar calidad, costo y latencia, y desplegar gradualmente."
        ),
        expected_keywords=[
            "OpenAI", "Responses API", "reasoning", "prompt caching", "latencia",
        ],
        category="openai_current_models",
        expected_step_order=[
            "Consultar documentación vigente",
            "Inventariar la integración actual",
            "Seleccionar modelo y parámetros",
            "Migrar y configurar caching",
            "Evaluar y desplegar gradualmente",
        ],
        requires_web=True,
    ),
]
