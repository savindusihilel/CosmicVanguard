const TARGET_CLASSES = ["SN Ia", "SN II", "AGN", "QSO", "RRLyrae", "Cepheid", "EB", "LPV", "CV", "Blazar"];
let currentLightCurve = null;
let probsChartContrastive = null;
let probsChartAe = null;
let modelInfoData = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initTabs();
    initInputMethods();
    initPredictionActions();
    populateHomeTab();
    fetchModelInfo();
    fetchComparisonData();

    // Set initial status
    updateAppStatus('Models loaded successfully!');
});

// Tab Logic - Radio based
function initTabs() {
    const radios = document.querySelectorAll('input[name="transient-tab"]');
    const contents = document.querySelectorAll('.tab-content');

    radios.forEach(radio => {
        radio.addEventListener('change', () => {
            if (radio.checked) {
                const target = radio.value;
                contents.forEach(c => c.classList.remove('active'));
                const targetEl = document.getElementById(`tab-${target}`);
                if (targetEl) targetEl.classList.add('active');

                if (target === 'flow') updateFlowStatus();
            }
        });
    });

    // Model Info Sub-tabs
    const infoTabBtns = document.querySelectorAll('.info-tab-btn');
    infoTabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            infoTabBtns.forEach(b => {
                b.classList.remove('active');
                b.style.color = 'var(--text-3)';
            });
            btn.classList.add('active');
            btn.style.color = '#38bdf8';
            updateModelInfoDisplay(btn.dataset.model);
        });
    });
}

function updateAppStatus(title) {
    document.getElementById('appStatusTitle').textContent = title;
}

function populateHomeTab() {
    const list = document.getElementById('classList');
    if (!list) return;
    list.innerHTML = TARGET_CLASSES.map(cls =>
        `<span style="background: rgba(14, 165, 233, 0.1); padding: 0.4rem 1rem; border-radius: 20px; font-size: 0.85rem; border: 1px solid rgba(14, 165, 233, 0.2); color: #bae6fd;">${cls}</span>`
    ).join('');
}

// Input Method Logic
function initInputMethods() {
    const radios = document.querySelectorAll('input[name="input-method"]');
    const sections = {
        json: document.getElementById('json-input-section'),
        manual: document.getElementById('manual-input-section'),
        csv: document.getElementById('csv-input-section')
    };

    radios.forEach(radio => {
        radio.addEventListener('change', () => {
            Object.values(sections).forEach(s => s.classList.add('hidden'));
            sections[radio.value].classList.remove('hidden');
            if (radio.value === 'manual' && document.getElementById('observations-container').children.length === 0) {
                generateObservationRows(10);
            }
        });
    });

    // CSV Input specifically
    const csvInput = document.getElementById('batchCsvInput');
    const fileName = document.getElementById('fileNameDisplay');
    if (csvInput) {
        csvInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                fileName.textContent = file.name;
                parseCSV(file);
            }
        });
    }
}

function generateObservationRows(count) {
    const container = document.getElementById('observations-container');
    if (!container) return;

    container.innerHTML = '';
    const defaultData = [
        { t: 58000.0, m: 19.5, e: 0.05, b: 'g' },
        { t: 58001.0, m: 19.2, e: 0.04, b: 'g' },
        { t: 58002.0, m: 18.8, e: 0.03, b: 'g' },
        { t: 58003.0, m: 18.2, e: 0.03, b: 'g' },
        { t: 58004.0, m: 17.8, e: 0.03, b: 'g' },
        { t: 58005.0, m: 17.5, e: 0.03, b: 'g' },
        { t: 58006.0, m: 17.3, e: 0.03, b: 'g' },
        { t: 58007.0, m: 17.2, e: 0.03, b: 'g' },
        { t: 58008.0, m: 17.1, e: 0.03, b: 'g' },
        { t: 58009.0, m: 17.2, e: 0.03, b: 'g' }
    ];

    for (let i = 0; i < count; i++) {
        const rowData = defaultData[i] || { t: (58000 + i).toFixed(1), m: 20.0, e: 0.1, b: i % 2 === 0 ? 'g' : 'r' };
        const rowHtml = `
            <div class="observation-row" style="background: rgba(15, 23, 42, 0.3); padding: 1rem; border-radius: 8px; border: 1px solid var(--border-subtle);">
                <div style="font-size: 0.75rem; color: #38bdf8; font-weight: 700; margin-bottom: 0.75rem; text-transform: uppercase;">Observation ${i + 1}</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                    <div>
                        <label style="font-size: 0.7rem; color: var(--text-3); display: block; margin-bottom: 0.25rem;">Time (MJD)</label>
                        <input type="number" step="0.1" class="manual-input-field" data-index="${i}" data-field="time" value="${rowData.t}" style="width: 100%; padding: 0.5rem; background: rgba(0,0,0,0.2); border: 1px solid var(--border-subtle); border-radius: 4px; color: white; font-size: 0.85rem;">
                    </div>
                    <div>
                        <label style="font-size: 0.7rem; color: var(--text-3); display: block; margin-bottom: 0.25rem;">Magnitude</label>
                        <input type="number" step="0.01" class="manual-input-field" data-index="${i}" data-field="magnitude" value="${rowData.m}" style="width: 100%; padding: 0.5rem; background: rgba(0,0,0,0.2); border: 1px solid var(--border-subtle); border-radius: 4px; color: white; font-size: 0.85rem;">
                    </div>
                    <div>
                        <label style="font-size: 0.7rem; color: var(--text-3); display: block; margin-bottom: 0.25rem;">Error</label>
                        <input type="number" step="0.01" class="manual-input-field" data-index="${i}" data-field="error" value="${rowData.e}" style="width: 100%; padding: 0.5rem; background: rgba(0,0,0,0.2); border: 1px solid var(--border-subtle); border-radius: 4px; color: white; font-size: 0.85rem;">
                    </div>
                    <div>
                        <label style="font-size: 0.7rem; color: var(--text-3); display: block; margin-bottom: 0.25rem;">Band</label>
                        <select class="manual-input-field" data-index="${i}" data-field="band" style="width: 100%; padding: 0.5rem; background: rgba(0,0,0,0.2); border: 1px solid var(--border-subtle); border-radius: 4px; color: white; font-size: 0.85rem;">
                            <option value="g" ${rowData.b === 'g' ? 'selected' : ''}>g</option>
                            <option value="r" ${rowData.b === 'r' ? 'selected' : ''}>r</option>
                        </select>
                    </div>
                </div>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', rowHtml);
    }

    // Add listeners to new inputs
    document.querySelectorAll('.manual-input-field').forEach(input => {
        input.addEventListener('input', updateManualPreview);
    });

    updateManualPreview();
}

function updateManualPreview() {
    const tableBody = document.querySelector('#preview-table tbody');
    if (!tableBody) return;

    tableBody.innerHTML = '';
    const rows = document.querySelectorAll('.observation-row');

    rows.forEach((row, i) => {
        const time = row.querySelector('[data-field="time"]').value;
        const mag = row.querySelector('[data-field="magnitude"]').value;
        const err = row.querySelector('[data-field="error"]').value;
        const band = row.querySelector('[data-field="band"]').value;

        tableBody.insertAdjacentHTML('beforeend', `
            <tr>
                <td style="padding: 0.5rem;">${time}</td>
                <td style="padding: 0.5rem;">${mag}</td>
                <td style="padding: 0.5rem;">${err}</td>
                <td style="padding: 0.5rem;">${band}</td>
            </tr>
        `);
    });
}

// Prediction Actions
function initPredictionActions() {
    const contrastiveBtn = document.getElementById('predictContrastiveBtn');
    const aeBtn = document.getElementById('predictAeBtn');

    if (contrastiveBtn) contrastiveBtn.addEventListener('click', () => runPredictions('contrastive'));
    if (aeBtn) aeBtn.addEventListener('click', () => runPredictions('autoencoder'));
}

async function getSelectedData() {
    const methodEl = document.querySelector('input[name="input-method"]:checked');
    if (!methodEl) return null;
    const method = methodEl.value;

    if (method === 'json') {
        try {
            const val = document.getElementById('jsonPayload').value;
            return JSON.parse(val);
        } catch (e) {
            alert('Invalid JSON format');
            return null;
        }
    } else if (method === 'manual') {
        const rows = document.querySelectorAll('.observation-row');
        if (rows.length !== 10) {
            alert('10 observations are required for prediction.');
            return null;
        }

        const observations = Array.from(rows).map(row => ({
            time: parseFloat(row.querySelector('[data-field="time"]').value),
            magnitude: parseFloat(row.querySelector('[data-field="magnitude"]').value),
            error: parseFloat(row.querySelector('[data-field="error"]').value),
            band: row.querySelector('[data-field="band"]').value
        }));

        return { object_id: 'manual_input', observations };
    } else {
        if (!currentLightCurve) {
            alert('Please upload a CSV file first');
            return null;
        }
        return currentLightCurve;
    }
}

async function runPredictions(type) {
    const data = await getSelectedData();
    if (!data) return;

    currentLightCurve = data;
    updateAppStatus(`State: Analyzing with ${type === 'contrastive' ? 'Contrastive' : 'Autoencoder'}...`);

    document.getElementById('singleResults').classList.remove('hidden');

    if (type === 'contrastive') {
        const res = await callPredictApi('/api/transient/predict', data);
        updateResultColumn('contrastive', res);
    } else if (type === 'autoencoder') {
        const res = await callPredictApi('/api/transient/autoencoder/predict', data);
        updateResultColumn('ae', res);
    }

    updateAppStatus('Models loaded successfully!');
}

async function callPredictApi(url, body) {
    try {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        });
        return await res.json();
    } catch (e) {
        console.error('API Error:', e);
        return null;
    }
}

function updateResultColumn(prefix, data) {
    if (!data) return;

    document.getElementById(`${prefix}TopClass`).textContent = data.predicted_class || 'Unknown';
    document.getElementById(`${prefix}Conf`).textContent = `${((data.confidence || 0) * 100).toFixed(2)}%`;
    document.getElementById(`${prefix}Oid`).textContent = data.object_id || 'N/A';

    // Alerts logic
    const alertEl = document.getElementById(`${prefix}Alert`);
    alertEl.classList.remove('hidden');

    if (data.confidence > 0.75) {
        alertEl.style.background = 'rgba(16, 185, 129, 0.1)';
        alertEl.style.border = '1px solid rgba(16, 185, 129, 0.2)';
        alertEl.style.color = '#e2e8f0';
        alertEl.innerHTML = `
            <span style="color: #10b981; font-size: 1.2rem;">🚀</span>
            <div style="font-size: 0.85rem; font-weight: 500;">
                High confidence prediction! The model is very sure about this classification.
            </div>`;
    } else {
        alertEl.style.background = 'rgba(234, 179, 8, 0.1)';
        alertEl.style.border = '1px solid rgba(234, 179, 8, 0.2)';
        alertEl.style.color = '#fde68a';
        alertEl.innerHTML = `
            <span style="color: #eab308; font-size: 1.2rem;">⚠️</span>
            <div style="font-size: 0.85rem; font-weight: 500;">
                Low confidence prediction. This light curve may be ambiguous or require more data.
            </div>`;
    }

    // Update Probs Chart
    const chart = prefix === 'contrastive' ? probsChartContrastive : probsChartAe;
    if (chart && data.probabilities) {
        // We want to show all 10 classes in the correct order for the comparison
        chart.data.labels = TARGET_CLASSES;
        chart.data.datasets[0].data = TARGET_CLASSES.map(cls => data.probabilities[cls] || 0);
        chart.update();
    }
}

// Charts Initialization
function initCharts() {
    probsChartContrastive = createProbsChart('probsChartContrastive');
    probsChartAe = createProbsChart('probsChartAe');
}

function createProbsChart(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    const ctx = el.getContext('2d');

    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: TARGET_CLASSES,
            datasets: [{
                data: Array(10).fill(0),
                backgroundColor: 'rgba(125, 211, 252, 0.8)',
                borderColor: '#38bdf8',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: { backgroundColor: 'rgba(15, 23, 42, 0.9)' }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: '#94a3b8', font: { size: 10 } }
                },
                y: {
                    min: 0,
                    max: 1,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#64748b' }
                }
            }
        }
    });
}

async function parseCSV(file) {
    const text = await file.text();
    const rows = text.split('\n').filter(r => r.trim());
    const headers = rows[0].split(',').map(h => h.trim().toLowerCase());

    const oidIdx = headers.indexOf('object_id');
    const timeIdx = headers.indexOf('time');
    const magIdx = headers.indexOf('magnitude');
    const errIdx = headers.indexOf('error');
    const bandIdx = headers.indexOf('band');

    if (oidIdx === -1) {
        alert('CSV must contain object_id column');
        return;
    }

    const grouped = {};
    for (let i = 1; i < rows.length; i++) {
        const cols = rows[i].split(',');
        if (cols.length < headers.length) continue;
        const oid = cols[oidIdx].trim();
        if (!grouped[oid]) grouped[oid] = [];
        grouped[oid].push({
            time: parseFloat(cols[timeIdx]),
            magnitude: parseFloat(cols[magIdx]),
            error: errIdx !== -1 ? parseFloat(cols[errIdx]) : 0.05,
            band: bandIdx !== -1 ? cols[bandIdx].trim() : 'g'
        });
    }

    const entries = Object.entries(grouped);
    if (entries.length === 0) return;

    currentLightCurve = {
        object_id: entries[0][0],
        observations: entries[0][1]
    };

    updateAppStatus(`State: Data Loaded. Parsed CSV with ${entries.length} objects.`);
}

async function fetchModelInfo() {
    try {
        const res = await fetch('/api/transient/model-info');
        modelInfoData = await res.json();
        updateModelInfoDisplay('contrastive');
    } catch (e) {
        console.error("Info Fetch Error:", e);
    }
}

function updateModelInfoDisplay(modelType) {
    if (!modelInfoData) return;
    const key = modelType === 'contrastive' ? 'contrastive_model' : 'autoencoder_model';
    const displayEl = document.getElementById('infoJsonDisplay');
    if (displayEl) {
        const data = { ...modelInfoData[key], device: "cpu", num_classes: 10 };
        displayEl.textContent = JSON.stringify(data, null, 4);
    }
}

async function fetchComparisonData() {
    try {
        const res = await fetch('/api/transient/model-comparison');
        const d = await res.json();
        const compEl = document.getElementById('compContrastiveArch');
        if (compEl) {
            document.getElementById('compContrastiveArch').textContent = d.contrastive_model.architecture;
            document.getElementById('compContrastiveAdv').innerHTML = d.contrastive_model.advantages.map(a => `<li>• ${a}</li>`).join('');
            document.getElementById('compAeArch').textContent = d.autoencoder_model.architecture;
            document.getElementById('compAeDisadv').innerHTML = d.autoencoder_model.disadvantages.map(a => `<li>• ${a}</li>`).join('');
        }
    } catch (e) { }
}

function updateFlowStatus() {
    console.log("System flow status updated");
}


