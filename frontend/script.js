// --- CONFIGURACIÓN VISUAL (ESTILOS NOTION) ---
var cy = cytoscape({
    container: document.getElementById('cy'),
    style: [
        // NODO PRINCIPAL
        {
            selector: 'node[type="main"]',
            style: {
                'label': 'data(label)',
                'shape': 'round-rectangle',
                'width': '200px', 'height': '60px',
                'background-color': '#ffffff',
                'border-width': 2, 'border-color': '#2c3e50',
                'text-valign': 'center', 'text-halign': 'center', 'text-wrap': 'wrap', 'text-max-width': '180px',
                'font-size': '13px', 'font-weight': 'bold', 'color': '#2c3e50',
                'shadow-blur': 5, 'shadow-color': '#000', 'shadow-opacity': 0.1
            }
        },
        // NODO DETALLE (HOJA)
        {
            selector: 'node[type="sub"]',
            style: {
                'label': 'data(label)',
                'shape': 'tag',
                'width': '140px', 'height': '40px',
                'background-color': '#fff9c4', // Amarillo Post-it
                'border-width': 1, 'border-color': '#fbc02d',
                'text-valign': 'center', 'text-halign': 'center', 'text-wrap': 'wrap', 'text-max-width': '120px',
                'font-size': '11px', 'color': '#5d4037',
                'shadow-blur': 2, 'shadow-offset-y': 2
            }
        },
        // FLECHA PRINCIPAL
        {
            selector: 'edge[type="main-flow"]',
            style: {
                'width': 3,
                'line-color': '#2c3e50',
                'target-arrow-color': '#2c3e50',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier'
            }
        },
        // FLECHA PUNTEADA
        {
            selector: 'edge[type="sub-link"]',
            style: {
                'width': 1, 'line-style': 'dashed', 'line-color': '#999',
                'target-arrow-shape': 'circle', 'target-arrow-color': '#999'
            }
        },
        // SELECCIÓN
        {
            selector: ':selected',
            style: { 'border-color': '#2eaadc', 'border-width': 3, 'background-color': '#e6f7ff' }
        }
    ]
});

const API_URL = "/generate-roadmap";

// --- FUNCIÓN PRINCIPAL ---
async function getRoadmap(question) {
    const loading = document.getElementById('loading');
    loading.classList.remove('hidden');

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });

        if (!response.ok) throw new Error("Error en servidor");
        const data = await response.json();

        // Validación de seguridad
        if (!data.steps) {
             if(data.title === "Error") { 
                 renderGraph(data); 
                 return; 
             }
             throw new Error("Respuesta sin pasos");
        }
        
        // 1. DIBUJAR EL MAPA
        renderGraph(data);
        
        // 2. RELLENAR LA LISTA LATERAL (¡Esta es la parte importante!)
        updateSidebarList(data.steps); 

    } catch (error) {
        alert("⚠️ " + error.message);
        console.error(error);
    } finally {
        loading.classList.add('hidden');
    }
}

// --- DIBUJAR GRAFO ---
function renderGraph(data) {
    cy.elements().remove();
    let elements = [];
    let steps = data.steps;

    steps.forEach((step, index) => {
        // Nodo Principal
        elements.push({
            data: { 
                id: step.id, 
                label: step.label, 
                description: step.description,
                type: 'main'
            }
        });

        // Nodos Hijos (Detalles)
        const points = step.key_points || [];
        points.forEach((point, subIndex) => {
            const subId = `${step.id}_sub_${subIndex}`;
            elements.push({ data: { id: subId, label: point, type: 'sub' } });
            elements.push({ data: { source: step.id, target: subId, type: 'sub-link' } });
        });

        // Conexión Secuencial
        if (index < steps.length - 1) {
            elements.push({
                data: { 
                    source: steps[index].id, 
                    target: steps[index+1].id,
                    type: 'main-flow'
                }
            });
        }
    });

    cy.add(elements);
    
    cy.layout({
        name: 'dagre',
        rankDir: 'LR',
        align: 'UL',
        rankSep: 100,
        nodeSep: 20,
        ranker: 'network-simplex'
    }).run();
}

// --- ACTUALIZAR BARRA LATERAL ---
function updateSidebarList(steps) {
    const listContainer = document.getElementById('nodeList');
    listContainer.innerHTML = ''; // Limpiar lo que había antes

    steps.forEach(step => {
        // Solo mostramos en la lista los pasos principales, no los detalles pequeños
        const item = document.createElement('div');
        item.className = 'node-item';
        item.innerText = step.label;
        
        // Al hacer clic en la lista -> Profundizar
        item.addEventListener('click', () => {
            // Seleccionar visualmente en el mapa
            cy.$(`#${step.id}`).select();
            // Ejecutar la acción
            performDrillDown(step.label, step.description);
        });

        listContainer.appendChild(item);
    });
}

// --- LÓGICA DE PROFUNDIZAR (Compartida) ---
function performDrillDown(label, description) {
    const context = description || "Sin contexto";
    if(confirm(`¿Profundizar en "${label}"?`)){
        document.getElementById('queryInput').value = `Profundizando: ${label}`;
        getRoadmap(`Detalles técnicos de: ${label}. Contexto: ${context}`);
    }
}

// --- EVENTOS ---
function handleSearch() {
    const q = document.getElementById('queryInput').value;
    if (q) getRoadmap(q);
}
document.getElementById('queryInput').addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleSearch();
});

// Doble Clic en el Mapa
cy.on('dblclick', 'node[type="main"]', function(evt){
    var node = evt.target;
    performDrillDown(node.data('label'), node.data('description'));
});