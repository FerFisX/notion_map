// Configuración inicial de Cytoscape
var cy = cytoscape({
    container: document.getElementById('cy'),
    style: [
        {
            selector: 'node',
            style: {
                'label': 'data(label)',
                'shape': 'round-rectangle',
                'width': '200px',
                'height': '60px',
                'background-color': 'white',
                'border-width': 2,
                'border-color': '#37352f',
                'text-valign': 'center',
                'text-halign': 'center',
                'text-wrap': 'wrap',
                'text-max-width': '180px',
                'font-size': '12px',
                'color': '#37352f',
                'shadow-blur': 10,
                'shadow-color': 'rgba(0,0,0,0.1)',
                'shadow-opacity': 0.5
            }
        },
        {
            selector: 'edge',
            style: {
                'width': 2,
                'line-color': '#ccc',
                'target-arrow-color': '#ccc',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier'
            }
        },
        {
            selector: ':selected', // Al hacer clic
            style: {
                'border-color': '#2eaadc',
                'border-width': 3,
                'background-color': '#f0f9ff'
            }
        }
    ]
});

// URL de tu API (Backend)
const API_URL = "http://127.0.0.1:8000/generate-roadmap";

async function getRoadmap(question) {
    const loading = document.getElementById('loading');
    loading.classList.remove('hidden');
    
    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });
        
        const data = await response.json();
        renderGraph(data);
    } catch (error) {
        alert("Error conectando con la API: " + error);
    } finally {
        loading.classList.add('hidden');
    }
}

function renderGraph(data) {
    cy.elements().remove(); // Limpiar mapa anterior
    
    let elements = [];
    let steps = data.steps;

    // 1. Crear Nodos
    steps.forEach((step) => {
        elements.push({
            data: { 
                id: step.id, 
                label: step.label, 
                description: step.description 
            }
        });
    });

    // 2. Crear Flechas (Secuencial: 1->2->3...)
    for (let i = 0; i < steps.length - 1; i++) {
        elements.push({
            data: { 
                source: steps[i].id, 
                target: steps[i+1].id 
            }
        });
    }

    cy.add(elements);

    // 3. Organizar layout (DAGRE es especial para jerarquías/roadmaps)
    cy.layout({
        name: 'dagre',
        rankDir: 'LR', // De izquierda a derecha (Left to Right)
        padding: 50
    }).run();
}

function handleSearch() {
    const query = document.getElementById('queryInput').value;
    if (query) getRoadmap(query);
}

// EVENTO DE DOBLE CLIC (Drill-down)
cy.on('dblclick', 'node', function(evt){
    var node = evt.target;
    var label = node.data('label');
    var description = node.data('description');
    
    // Preguntar confirmación
    if(confirm(`¿Quieres profundizar en: "${label}"?`)){
        // Usamos el label del nodo como nueva pregunta
        document.getElementById('queryInput').value = `Detalles de: ${label}. ${description}`;
        getRoadmap(`Dame pasos detallados para: ${label}. Contexto: ${description}`);
    }
});