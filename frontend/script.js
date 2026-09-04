// --- 1. CONFIGURACIÓN CYTOSCAPE ---
var cy = cytoscape({
    container: document.getElementById('cy'),
    style: [
        { selector: 'node[type="main"]', style: { 'label': 'data(label)', 'shape': 'round-rectangle', 'width': '220px', 'height': '60px', 'background-color': 'white', 'border-width': 2, 'border-color': '#333', 'text-valign': 'center', 'text-halign': 'center', 'text-wrap': 'wrap', 'text-max-width': '180px', 'font-size': '12px', 'font-weight': 'bold' } },
        // Color según la fuente predominante del nodo
        { selector: 'node[type="main"][source="corpus"]', style: { 'background-color': '#e6f7ff', 'border-color': '#1890ff' } },
        { selector: 'node[type="main"][source="web"]',    style: { 'background-color': '#fff7e6', 'border-color': '#fa8c16' } },
        { selector: 'node[type="main"][source="hybrid"]', style: { 'background-color': '#f3e8ff', 'border-color': '#722ed1' } },
        { selector: 'node[type="sub"]', style: { 'label': 'data(label)', 'shape': 'tag', 'width': '140px', 'height': '40px', 'background-color': '#fffbe6', 'border-color': '#ffe58f', 'border-width': 1, 'text-valign': 'center', 'text-halign': 'center', 'text-wrap': 'wrap', 'text-max-width': '120px', 'font-size': '10px', 'color': '#555' } },
        { selector: 'edge', style: { 'width': 2, 'line-color': '#ccc', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier' } },
        { selector: ':selected', style: { 'border-width': 4, 'border-color': '#1890ff' } }
    ]
});

const API_URL = "/generate-roadmap";

// --- 2. MEMORIA GLOBAL (CACHÉ) ---
// Aquí guardamos los datos de cada nodo para no tener que pedirlos de nuevo
// Clave: ID del DOM del nodo en la barra lateral. Valor: Objeto JSON con steps.
const nodeDataCache = {};
let activeDomId = null; // Cuál nodo de la barra lateral estamos viendo actualmente
let activeSourceMode = 'auto';
let generationInProgress = false;
let loadingMessageTimer = null;

const loadingMessages = [
    'Preparando la consulta...',
    'Buscando información...',
    'Organizando los pasos...',
    'Validando las fuentes...',
    'Finalizando detalles...'
];

function startLoadingState() {
    if (generationInProgress) return false;

    hideUserMessage();
    generationInProgress = true;
    const loading = document.getElementById('loading');
    const loadingMessage = document.getElementById('loadingMessage');
    const generateButton = document.getElementById('generateButton');
    const sourceMode = document.getElementById('sourceMode');
    let messageIndex = 0;

    loadingMessage.innerText = loadingMessages[messageIndex];
    loading.classList.remove('hidden');
    generateButton.disabled = true;
    generateButton.innerText = 'Generando...';
    sourceMode.disabled = true;

    loadingMessageTimer = window.setInterval(() => {
        messageIndex = Math.min(messageIndex + 1, loadingMessages.length - 1);
        loadingMessage.innerText = loadingMessages[messageIndex];
        if (messageIndex === loadingMessages.length - 1) {
            window.clearInterval(loadingMessageTimer);
            loadingMessageTimer = null;
        }
    }, 20000);

    return true;
}

function stopLoadingState() {
    if (loadingMessageTimer) {
        window.clearInterval(loadingMessageTimer);
        loadingMessageTimer = null;
    }

    generationInProgress = false;
    document.getElementById('loading').classList.add('hidden');
    document.getElementById('loadingMessage').innerText = loadingMessages[0];
    const generateButton = document.getElementById('generateButton');
    generateButton.disabled = false;
    generateButton.innerText = 'Generar Roadmap';
    document.getElementById('sourceMode').disabled = false;
}

// --- 3. FUNCIÓN PRINCIPAL DE CARGA ---
function createRequestError(data, response) {
    const error = new Error(
        data?.message || data?.detail || data?.title || `HTTP ${response.status}`
    );
    error.status = String(data?.status || 'TECHNICAL_ERROR').toUpperCase();
    error.httpStatus = response.status;
    return error;
}

function hideUserMessage() {
    const message = document.getElementById('userMessage');
    if (!message) return;
    message.classList.add('hidden');
    message.classList.remove('technical');
    document.getElementById('userMessageActions').replaceChildren();
}

function addUserMessageAction(label, callback) {
    const button = document.createElement('button');
    button.type = 'button';
    button.innerText = label;
    button.addEventListener('click', callback);
    document.getElementById('userMessageActions').appendChild(button);
}

function retryCurrentQuestion() {
    hideUserMessage();
    handleSearch();
}

function editCurrentQuestion() {
    hideUserMessage();
    document.getElementById('queryInput').focus();
}

function selectSourceMode(mode) {
    const selector = document.getElementById('sourceMode');
    selector.value = mode;
    activeSourceMode = mode;
    selector.dispatchEvent(new Event('change'));
    hideUserMessage();
    selector.focus();
}

function focusSourceMode() {
    document.getElementById('sourceMode').focus();
}

function showUserError(error) {
    const status = String(error?.status || 'TECHNICAL_ERROR').toUpperCase();
    const message = document.getElementById('userMessage');
    const title = document.getElementById('userMessageTitle');
    const text = document.getElementById('userMessageText');

    message.classList.remove('hidden', 'technical');
    document.getElementById('userMessageActions').replaceChildren();

    if (status === 'NO_ROADMAP') {
        title.innerText = 'Esta solicitud no necesita un roadmap';
        text.innerText = 'Reformula la pregunta como un proceso u objetivo técnico si deseas generar una guía paso a paso.';
        addUserMessageAction('Editar pregunta', editCurrentQuestion);
        return;
    }

    if (status === 'INSUFFICIENT_EVIDENCE') {
        if (activeSourceMode === 'corpus') {
            title.innerText = 'No encontramos suficiente información en la base interna';
            text.innerText = 'No podemos crear un roadmap confiable con las fuentes disponibles. Prueba con el modo Automático o Fuentes Web.';
            addUserMessageAction('Cambiar a Automático', () => selectSourceMode('auto'));
            addUserMessageAction('Cambiar a Web', () => selectSourceMode('web'));
        } else if (activeSourceMode === 'web') {
            title.innerText = 'No encontramos fuentes web suficientes';
            text.innerText = 'No fue posible recuperar información confiable en este momento. Puedes intentarlo nuevamente o reformular la pregunta.';
            addUserMessageAction('Intentar nuevamente', retryCurrentQuestion);
        } else {
            title.innerText = 'No encontramos evidencia suficiente';
            text.innerText = 'Las fuentes disponibles no permiten crear un roadmap confiable. Intenta reformular la pregunta o seleccionar otra fuente.';
            addUserMessageAction('Intentar nuevamente', retryCurrentQuestion);
            addUserMessageAction('Cambiar modo', focusSourceMode);
        }
        addUserMessageAction('Editar pregunta', editCurrentQuestion);
        return;
    }

    if (status === 'TIMEOUT') {
        title.innerText = 'La generación está tomando más tiempo del esperado';
        text.innerText = 'El proceso se detuvo para evitar una espera excesiva. Inténtalo nuevamente o prueba con otro modo de fuentes.';
        addUserMessageAction('Intentar nuevamente', retryCurrentQuestion);
        addUserMessageAction('Cambiar modo', focusSourceMode);
        return;
    }

    if (status === 'GROUNDING_REVIEW_REQUIRED') {
        title.innerText = 'No pudimos verificar toda la información';
        text.innerText = 'El roadmap no fue mostrado porque algunos pasos no tenían respaldo suficiente en las fuentes recuperadas.';
        addUserMessageAction('Intentar nuevamente', retryCurrentQuestion);
        addUserMessageAction('Editar pregunta', editCurrentQuestion);
        addUserMessageAction('Cambiar modo', focusSourceMode);
        return;
    }

    message.classList.add('technical');
    title.innerText = 'No pudimos completar la solicitud';
    text.innerText = 'Ocurrió un problema inesperado. Inténtalo nuevamente y, si continúa, informa al equipo técnico.';
    addUserMessageAction('Intentar nuevamente', retryCurrentQuestion);
}

document.getElementById('dismissUserMessage').addEventListener('click', hideUserMessage);

async function fetchRoadmap(question, parentDomId = null, nodeLabel = "Inicio", nodeDesc = "") {
    if (!startLoadingState()) return;

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: question,
                source_mode: activeSourceMode
            })
        });
        const data = await response.json();

        if (!response.ok) {
            throw createRequestError(data, response);
        }
        if (data.status && data.status !== 'ACCEPTED') {
            throw createRequestError(data, response);
        }

        if (!data.steps && data.title !== "Error") {
            throw createRequestError(
                {
                    status: 'TECHNICAL_ERROR',
                    message: 'The roadmap response did not include a valid steps collection.'
                },
                response
            );
        }

        // LÓGICA DE ÁRBOL
        if (parentDomId) {
            // Caso A: Estamos profundizando (Drill Down)
            // 1. Agregamos los hijos a la barra lateral
            const newChildrenIds = appendChildrenToSidebar(parentDomId, data.steps);
            
            // 2. Guardamos la data en caché asociada al PRIMER hijo generado? 
            // NO, la data pertenece al nodo PADRE que acabamos de clickear.
            // PERO, para navegar, asociaremos esta "vista" al nodo padre en el caché.
            nodeDataCache[parentDomId].drillDownData = data; 
            
            // 3. Visualizamos el nuevo mapa
            renderGraph(data);
            setActiveNode(parentDomId); // Mantenemos el foco en el padre expandido o pasamos al hijo?
            // Generalmente al hacer drill down, quieres ver el detalle.
            // Vamos a dejar marcado el padre como "Expandido y Viendo Detalle".

        } else {
            // Caso B: Búsqueda Nueva (Raíz)
            document.getElementById('nodeList').innerHTML = ''; // Limpiar árbol
            const rootId = 'root-node';
            
            // Creamos un nodo raíz virtual en la caché
            nodeDataCache[rootId] = { 
                label: "🏠 " + (question.length > 20 ? question.substring(0,20)+"..." : question), 
                data: data,
                description: "Vista Principal"
            };

            // Renderizamos los items iniciales en la barra lateral
            // (Técnicamente son hijos de la raíz)
            // Para simplificar, los ponemos directo en el container y los asociamos a la raíz.
            data.steps.forEach(step => createSidebarNode(step, document.getElementById('nodeList'), rootId));
            
            // Renderizamos gráfico inicial
            renderGraph(data);
            updateBreadcrumbs([{id: rootId, label: "Inicio"}]);
            updateContextBanner("Inicio", "Vista Principal");
        }

    } catch (e) {
        console.error('Roadmap request failed', {
            status: e.status || 'TECHNICAL_ERROR',
            detail: e.message,
            error: e
        });
        showUserError(e);
    } finally {
        stopLoadingState();
    }
}

// --- 4. GESTIÓN DE LA BARRA LATERAL (EL ÁRBOL) ---

function createSidebarNode(step, container, parentDomId) {
    // Generamos un ID único para el DOM
    const domId = `node-${step.id}-${Math.random().toString(36).substr(2, 5)}`;
    
    // Guardamos la info básica en caché (para poder recuperarla al hacer click)
    // OJO: Al principio, este nodo NO tiene "drillDownData" (hijos), solo tiene su propia info.
    // Cuando le hacemos click, mostramos su PROPIO detalle si lo tuviéramos, 
    // pero aquí la lógica es: Click en nodo -> Mostrar gráfico donde ÉL es el protagonista (sus hijos).
    
    nodeDataCache[domId] = {
        label: step.label,
        description: step.description,
        evidenceIds: step.evidence_ids || [],
        parentId: parentDomId,
        data: null // Aquí se guardará el gráfico de sus hijos cuando se cargue
    };

    // Estructura HTML
    const wrapper = document.createElement('div');
    wrapper.id = domId;
    wrapper.className = 'tree-node-wrapper';

    const row = document.createElement('div');
    row.className = 'node-item';
    row.dataset.id = domId; // Referencia
    row.innerHTML = `<span class="node-icon">▶</span> <span class="node-text">${step.label}</span>`;
    
    const childrenContainer = document.createElement('div');
    childrenContainer.className = 'nested-group';
    childrenContainer.id = domId + "-children";

    // --- CLIC EN EL NODO (NAVEGACIÓN) ---
    row.onclick = (e) => {
        e.stopPropagation();
        handleNodeClick(domId);
    };

    wrapper.appendChild(row);
    wrapper.appendChild(childrenContainer);
    container.appendChild(wrapper);
}

function appendChildrenToSidebar(parentDomId, steps) {
    const parentWrapper = document.getElementById(parentDomId);
    const childrenContainer = document.getElementById(parentDomId + "-children");
    const icon = parentWrapper.querySelector('.node-icon');

    // Cambiar icono a expandido
    if(icon) icon.innerText = "▼";
    childrenContainer.classList.add('open');

    // Crear nodos hijos
    steps.forEach(step => createSidebarNode(step, childrenContainer, parentDomId));
}


// --- 5. LÓGICA DE NAVEGACIÓN (EL CEREBRO) ---

function handleNodeClick(domId) {
    const nodeInfo = nodeDataCache[domId];
    setActiveNode(domId);

    // ESCENARIO 1: Ya tenemos los datos de sus hijos (ya se hizo drill down antes)
    if (nodeInfo.data) {
        console.log("Cargando desde caché:", nodeInfo.label);
        renderGraph(nodeInfo.data);
        generateBreadcrumbs(domId);
        updateContextBanner(
            nodeInfo.label,
            nodeInfo.description,
            nodeInfo.evidenceIds
        );
        
        // Asegurar que la carpeta esté abierta visualmente
        const childrenContainer = document.getElementById(domId + "-children");
        if(childrenContainer.hasChildNodes()) {
            childrenContainer.classList.add('open');
            const icon = document.getElementById(domId).querySelector('.node-icon');
            if(icon) icon.innerText = "▼";
        }
    } 
    // ESCENARIO 2: Es la primera vez (Hacer Drill Down real)
    else {
        if (!startLoadingState()) return;
        console.log("Descargando datos para:", nodeInfo.label);
        const icon = document.getElementById(domId).querySelector('.node-icon');
        if(icon) icon.innerText = "⏳";

        // Llamada a la API
        fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                question: `Detalles técnicos paso a paso de: "${nodeInfo.label}". Contexto: ${nodeInfo.description || ""}`,
                source_mode: activeSourceMode
            })
        })
        .then(async res => {
            const data = await res.json();
            if (!res.ok) throw createRequestError(data, res);
            if (data.status && data.status !== 'ACCEPTED') {
                throw createRequestError(data, res);
            }
            return data;
        })
        .then(data => {
            if (!data.steps) {
                throw createRequestError(
                    {
                        status: 'TECHNICAL_ERROR',
                        message: 'The roadmap response did not include a valid steps collection.'
                    },
                    { status: 200 }
                );
            }
            
            // Guardar en caché
            nodeInfo.data = data;
            
            // Renderizar hijos en sidebar
            appendChildrenToSidebar(domId, data.steps);
            
            // Renderizar gráfico
            renderGraph(data);
            generateBreadcrumbs(domId);
            updateContextBanner(
                nodeInfo.label,
                nodeInfo.description,
                nodeInfo.evidenceIds
            );
        })
        .catch(err => {
            console.error('Roadmap drill-down request failed', {
                status: err.status || 'TECHNICAL_ERROR',
                detail: err.message,
                error: err
            });
            showUserError(err);
            if(icon) icon.innerText = "▶";
        })
        .finally(() => {
            stopLoadingState();
        });
    }
}

function setActiveNode(domId) {
    // Quitar activo anterior
    document.querySelectorAll('.node-item').forEach(el => el.classList.remove('active-node'));
    // Poner nuevo activo
    const el = document.getElementById(domId)?.querySelector('.node-item');
    if(el) el.classList.add('active-node');
    activeDomId = domId;
}

// --- 6. BREADCRUMBS DINÁMICOS (Rastrear padres) ---
function generateBreadcrumbs(currentDomId) {
    const path = [];
    let curr = currentDomId;
    
    // Subimos por el árbol buscando los padres en la caché
    while(curr && nodeDataCache[curr]) {
        path.unshift({ 
            id: curr, 
            label: nodeDataCache[curr].label 
        });
        curr = nodeDataCache[curr].parentId;
    }

    // Agregamos Inicio al principio si no está
    if(path.length === 0 || path[0].label !== "Inicio") {
        path.unshift({ id: 'root', label: "🏠 Inicio" });
    }

    updateBreadcrumbs(path);
}

function updateBreadcrumbs(pathArray) {
    const container = document.getElementById('breadcrumbs');
    container.innerHTML = '';

    pathArray.forEach((item, index) => {
        const span = document.createElement('span');
        span.className = item.id === activeDomId ? 'crumb active' : 'crumb';
        span.innerText = item.label;
        
        // Clic en el breadcrumb = Clic en el nodo del árbol correspondiente
        span.onclick = () => {
            if (item.id === 'root') {
                // Caso especial Inicio (reset visual del grafo, no del árbol)
                // Recuperamos la data inicial que guardamos en root-node? 
                // Simplificación: Recargar la primera búsqueda es complejo si no la guardamos.
                // Truco: Si hacen clic en Inicio, solo mostramos mensaje o la primera data si la guardamos.
                alert("Usa la barra lateral para volver a la raíz.");
            } else {
                handleNodeClick(item.id);
            }
        };

        container.appendChild(span);
        
        if (index < pathArray.length - 1) {
            const sep = document.createElement('span');
            sep.style.color = '#ccc';
            sep.innerText = '/';
            container.appendChild(sep);
        }
    });
}

// --- 7. UTILS Y EVENTOS ---

function renderGraph(data) {
    cy.elements().remove();
    updateSourceMetrics(data.sources);
    let elements = [];
    data.steps.forEach((step, i) => {
        elements.push({ data: { id: step.id, label: step.label, description: step.description, type: 'main', source: step.source || 'corpus' } });
        (step.key_points || []).forEach((p, j) => {
            const subId = `${step.id}_sub_${j}`;
            elements.push({ data: { id: subId, label: p, type: 'sub' } });
            elements.push({ data: { source: step.id, target: subId, type: 'sub-link' } });
        });
        if (i < data.steps.length - 1) elements.push({ data: { source: step.id, target: data.steps[i+1].id, type: 'main-flow' } });
    });
    cy.add(elements);
    cy.resize();
    cy.layout({ name: 'dagre', rankDir: 'LR', align: 'UL', rankSep: 80, nodeSep: 20, fit: true, padding: 30, animate: true }).run();
}

function updateSourceMetrics(sources) {
    const panel = document.getElementById('sourceMetrics');
    if (!sources) { panel.classList.add('hidden'); return; }

    const corpus = sources.corpus_pct ?? 0;
    const web    = sources.web_pct ?? 0;
    panel.classList.remove('hidden');

    document.getElementById('smCorpus').innerText = corpus + '%';
    document.getElementById('smWeb').innerText    = web + '%';
    document.getElementById('smBarCorpus').style.width = corpus + '%';
    document.getElementById('smBarWeb').style.width    = web + '%';

    const modeLabels = {
        corpus: 'Solo base de conocimiento',
        hybrid: 'Híbrido (corpus + web)',
        web:    'Solo fuentes web',
        error:  'Error'
    };
    const mode = sources.mode || 'corpus';
    document.getElementById('smMode').innerText = modeLabels[mode] || mode;

    const sourceList = document.getElementById('smSources');
    sourceList.replaceChildren();
    (sources.items || []).forEach(source => {
        const row = document.createElement('div');
        row.className = 'sm-source';
        const label = `${formatSourceId(source.id)}: ${source.title || formatSourceId(source.source_type)}`;
        if (source.url) {
            const link = document.createElement('a');
            link.href = source.url;
            link.target = '_blank';
            link.rel = 'noopener noreferrer';
            link.textContent = label;
            row.appendChild(link);
        } else {
            row.textContent = label;
        }
        sourceList.appendChild(row);
    });
}

function formatSourceId(sourceId) {
    return String(sourceId || 'Fuente')
        .replace(/[_-]+/g, ' ')
        .replace(/\b\w/g, character => character.toUpperCase());
}

function updateContextBanner(title, desc, evidenceIds = []) {
    const banner = document.getElementById('activeContext');
    banner.classList.remove('hidden');
    document.getElementById('contextTitle').innerText = title;
    const evidence = evidenceIds.length
        ? `\nFuentes: ${evidenceIds.map(formatSourceId).join(', ')}`
        : '';
    document.getElementById('contextDesc').innerText =
        (desc || "Detalle técnico") + evidence;
}

// Doble Clic en el Gráfico -> Busca el nodo en el árbol y lo clickea
cy.on('dblclick', 'node[type="main"]', function(evt){
    const label = evt.target.data('label');
    // Buscamos en el DOM de la sidebar algún nodo con ese texto
    const sidebarNodes = document.querySelectorAll('.node-text');
    for (let span of sidebarNodes) {
        if (span.innerText === label) {
            // Encontramos el nodo en el árbol, hacemos clic en él
            span.parentElement.click(); 
            return;
        }
    }
    // Si no lo encuentra (raro), fallback manual
    alert("No se encontró este nodo en el árbol lateral.");
});

function handleSearch() {
    if (generationInProgress) return;
    const q = document.getElementById('queryInput').value;
    activeSourceMode = document.getElementById('sourceMode').value;
    if(q) fetchRoadmap(q);
}

const sourceModeHelp = {
    corpus: 'Usa únicamente la base interna y rechaza contenido sin respaldo.',
    web: 'Usa únicamente páginas web recuperadas y citadas.',
    auto: 'Combina las fuentes disponibles y exige evidencia para cada paso.'
};

document.getElementById('sourceMode').addEventListener('change', (event) => {
    document.getElementById('sourceModeHelp').innerText =
        sourceModeHelp[event.target.value];
});
document.getElementById('queryInput').addEventListener("keypress", (e) => { if(e.key==="Enter") handleSearch() });
