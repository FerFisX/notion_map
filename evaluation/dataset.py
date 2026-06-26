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
]
