// --- 1. CONFIGURACIÓN CYTOSCAPE ---
var cy = cytoscape({
    container: document.getElementById('cy'),
    style: [
        { selector: 'node[type="main"]', style: { 'label': 'data(label)', 'shape': 'round-rectangle', 'width': '220px', 'height': '60px', 'background-color': 'white', 'border-width': 2, 'border-color': '#333', 'text-valign': 'center', 'text-halign': 'center', 'text-wrap': 'wrap', 'text-max-width': '180px', 'font-size': '12px', 'font-weight': 'bold' } },
        { selector: 'node[type="sub"]', style: { 'label': 'data(label)', 'shape': 'tag', 'width': '140px', 'height': '40px', 'background-color': '#fffbe6', 'border-color': '#ffe58f', 'border-width': 1, 'text-valign': 'center', 'text-halign': 'center', 'text-wrap': 'wrap', 'text-max-width': '120px', 'font-size': '10px', 'color': '#555' } },
        { selector: 'edge', style: { 'width': 2, 'line-color': '#ccc', 'target-arrow-shape': 'triangle', 'curve-style': 'bezier' } },
        { selector: ':selected', style: { 'border-color': '#1890ff', 'border-width': 3, 'background-color': '#e6f7ff' } }
    ]
});

const API_URL = "/generate-roadmap";

// --- 2. MEMORIA GLOBAL (CACHÉ) ---
// Aquí guardamos los datos de cada nodo para no tener que pedirlos de nuevo
// Clave: ID del DOM del nodo en la barra lateral. Valor: Objeto JSON con steps.
const nodeDataCache = {}; 
let activeDomId = null; // Cuál nodo de la barra lateral estamos viendo actualmente

// --- 3. FUNCIÓN PRINCIPAL DE CARGA ---
async function fetchRoadmap(question, parentDomId = null, nodeLabel = "Inicio", nodeDesc = "") {
    const loading = document.getElementById('loading');
    loading.classList.remove('hidden');

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });
        const data = await response.json();

        if (!data.steps && data.title !== "Error") throw new Error("Datos incorrectos");

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
        alert("Error: " + e.message);
        console.error(e);
    } finally {
        loading.classList.add('hidden');
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
        updateContextBanner(nodeInfo.label, nodeInfo.description);
        
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
        console.log("Descargando datos para:", nodeInfo.label);
        const icon = document.getElementById(domId).querySelector('.node-icon');
        if(icon) icon.innerText = "⏳";

        // Llamada a la API
        fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                question: `Detalles técnicos paso a paso de: "${nodeInfo.label}". Contexto: ${nodeInfo.description || ""}` 
            })
        })
        .then(res => res.json())
        .then(data => {
            if (!data.steps) throw new Error("Error API");
            
            // Guardar en caché
            nodeInfo.data = data;
            
            // Renderizar hijos en sidebar
            appendChildrenToSidebar(domId, data.steps);
            
            // Renderizar gráfico
            renderGraph(data);
            generateBreadcrumbs(domId);
            updateContextBanner(nodeInfo.label, nodeInfo.description);
        })
        .catch(err => {
            alert("Error: " + err.message);
            if(icon) icon.innerText = "▶";
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
    let elements = [];
    data.steps.forEach((step, i) => {
        elements.push({ data: { id: step.id, label: step.label, description: step.description, type: 'main' } });
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

function updateContextBanner(title, desc) {
    const banner = document.getElementById('activeContext');
    banner.classList.remove('hidden');
    document.getElementById('contextTitle').innerText = title;
    document.getElementById('contextDesc').innerText = desc || "Detalle técnico";
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
    const q = document.getElementById('queryInput').value;
    if(q) fetchRoadmap(q);
}

document.getElementById('queryInput').addEventListener("keypress", (e) => { if(e.key==="Enter") handleSearch() });