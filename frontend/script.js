// --- ESTILOS VISUALES (EFECTO RAMA/HOJA) ---
var cy = cytoscape({
    container: document.getElementById('cy'),
    style: [
        // 1. CAMINO PRINCIPAL (Fuerte y Claro)
        {
            selector: 'node[type="main"]',
            style: {
                'label': 'data(label)',
                'shape': 'round-rectangle',
                'width': '200px', 'height': '60px',
                'background-color': '#ffffff',
                'border-width': 2, 'border-color': '#2c3e50', // Azul oscuro
                'text-valign': 'center', 'text-halign': 'center', 'text-wrap': 'wrap', 'text-max-width': '180px',
                'font-size': '14px', 'font-weight': 'bold', 'color': '#2c3e50',
                'shadow-blur': 5, 'shadow-color': '#000', 'shadow-opacity': 0.1
            }
        },
        // 2. HOJAS / BIFURCACIONES (Notas técnicas)
        {
            selector: 'node[type="sub"]',
            style: {
                'label': 'data(label)',
                'shape': 'tag', // Forma de etiqueta
                'width': '140px', 'height': '40px',
                'background-color': '#fff9c4', // Amarillo Post-it
                'border-width': 1, 'border-color': '#fbc02d',
                'text-valign': 'center', 'text-halign': 'center', 'text-wrap': 'wrap', 'text-max-width': '120px',
                'font-size': '11px', 'color': '#5d4037',
                'shadow-blur': 2, 'shadow-offset-y': 2
            }
        },
        // 3. CONEXIÓN PRINCIPAL (Sólida y con Flecha)
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
        // 4. CONEXIÓN A HOJA (Punteada y sutil)
        {
            selector: 'edge[type="sub-link"]',
            style: {
                'width': 1,
                'line-style': 'dashed', // Punteada
                'line-color': '#999',
                'target-arrow-shape': 'circle', // Un puntito al final
                'target-arrow-color': '#999'
            }
        }
    ]
});

const API_URL = "/generate-roadmap";

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

        if (!data.steps) {
             if(data.title === "Error") { renderGraph(data); return; }
             throw new Error("Respuesta sin pasos");
        }
        
        renderGraph(data);

    } catch (error) {
        alert("⚠️ " + error.message);
    } finally {
        loading.classList.add('hidden');
    }
}

function renderGraph(data) {
    cy.elements().remove();
    let elements = [];
    let steps = data.steps;

    steps.forEach((step, index) => {
        // --- NODO PRINCIPAL ---
        elements.push({
            data: { 
                id: step.id, 
                label: step.label, 
                description: step.description,
                type: 'main'
            }
        });

        // --- NODOS HOJA (Si existen key_points) ---
        const points = step.key_points || [];
        points.forEach((point, subIndex) => {
            const subId = `${step.id}_sub_${subIndex}`;
            
            // Hoja
            elements.push({ 
                data: { id: subId, label: point, type: 'sub' } 
            });
            
            // Rama (Conexión Padre -> Hoja)
            elements.push({ 
                data: { source: step.id, target: subId, type: 'sub-link' } 
            });
        });

        // --- CONEXIÓN AL SIGUIENTE PASO (Camino Principal) ---
        if (index < steps.length - 1) {
            elements.push({
                data: { 
                    source: steps[index].id, 
                    target: steps[index+1].id,
                    type: 'main-flow' // Importante para que salga gruesa
                }
            });
        }
    });

    cy.add(elements);
    
    // Configuración del Layout para que parezcan ramas
    cy.layout({
        name: 'dagre',
        rankDir: 'LR',     // Dirección Izquierda a Derecha
        align: 'UL',       // Alinear arriba-izquierda
        rankSep: 100,      // Separación entre pasos principales
        nodeSep: 20,       // Separación entre notas y pasos
        ranker: 'network-simplex' // Algoritmo que suele ordenar mejor las ramas
    }).run();
}

// Eventos
function handleSearch() {
    const q = document.getElementById('queryInput').value;
    if (q) getRoadmap(q);
}
document.getElementById('queryInput').addEventListener("keypress", (e) => {
    if (e.key === "Enter") handleSearch();
});

// Doble Clic (Drill-down)
cy.on('dblclick', 'node[type="main"]', function(evt){
    var node = evt.target;
    if(confirm(`¿Profundizar en "${node.data('label')}"?`)){
        document.getElementById('queryInput').value = `Profundizando: ${node.data('label')}`;
        getRoadmap(`Detalles técnicos de: ${node.data('label')}. Contexto: ${node.data('description')}`);
    }
});