/* ═══════════════════════════════════════════
   StarForge — Frontend Logic
   ═══════════════════════════════════════════ */

Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;

let comparisonChart = null, radarChart = null, robustnessChart = null, latentChart = null, cmdChart = null;
let diagnosticsLoaded = false;
let currentPredictionData = null;

document.getElementById('starforgeForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    document.getElementById('loadingOverlay').style.display = 'block';
    document.getElementById('resultsContainer').style.display = 'none';

    const data = {
        u: parseFloat(document.getElementById('sf_u').value),
        g: parseFloat(document.getElementById('sf_g').value),
        r: parseFloat(document.getElementById('sf_r').value),
        i: parseFloat(document.getElementById('sf_i').value),
        z: parseFloat(document.getElementById('sf_z').value),
        redshift: parseFloat(document.getElementById('sf_redshift').value)
    };

    currentPredictionData = data;

    try {
        const res = await fetch('/api/starforge/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (!res.ok) throw new Error(`Prediction failed: ${res.statusText}`);
        const result = await res.json();

        displayPredictionResults(result);

        // Load diagnostics asynchronously if they haven't been loaded yet
        await loadValidationDiagnostics(data);

        document.getElementById('resultsContainer').style.display = 'flex';
    } catch (err) {
        alert('Analysis failed: ' + err.message);
    } finally {
        document.getElementById('loadingOverlay').style.display = 'none';

        // Need to update CMD chart with new target point
        if (diagnosticsLoaded && cmdChart) {
            const userGr = data.g - data.r;
            cmdChart.data.datasets[1].data = [{ x: userGr, y: data.r }];
            cmdChart.update();
        }
    }
});

function displayPredictionResults(r) {
    // Render population bars
    renderPopBars('baseBars', r.labels, r.baseline, false);
    renderPopBars('researchBars', r.labels, r.research, true);

    // Charts
    renderComparisonChart(r);
    renderRadarChart(r);
}

function renderPopBars(containerId, labels, values, isResearch) {
    const container = document.getElementById(containerId);
    const classNames = ['fill-young', 'fill-inter', 'fill-old'];

    container.innerHTML = '';

    labels.forEach((label, i) => {
        const val = values[i];
        const div = document.createElement('div');
        div.className = `sf-progress-item`;
        div.innerHTML = `
            <div class="sf-progress-label">
                <strong>${label}</strong>
                <span style="font-family: 'JetBrains Mono', monospace;">${val.toFixed(2)}%</span>
            </div>
            <div class="sf-progress-bar-bg">
                <div class="sf-progress-bar-fill ${classNames[i]}" style="width:0%"></div>
            </div>
        `;
        container.appendChild(div);

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                div.querySelector('.sf-progress-bar-fill').style.width = `${Math.min(val, 100)}%`;
            });
        });
    });
}

function renderComparisonChart(r) {
    const ctx = document.getElementById('comparisonChart');
    if (comparisonChart) comparisonChart.destroy();

    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: r.labels,
            datasets: [
                {
                    label: 'Baseline Model',
                    data: r.baseline,
                    backgroundColor: 'rgba(59, 130, 246, 0.5)',
                    borderColor: '#3b82f6',
                    borderWidth: 1,
                    borderRadius: 4,
                    barPercentage: 0.6,
                    categoryPercentage: 0.7
                },
                {
                    label: 'Research Model',
                    data: r.research,
                    backgroundColor: 'rgba(139, 92, 246, 0.7)',
                    borderColor: '#8b5cf6',
                    borderWidth: 1,
                    borderRadius: 4,
                    barPercentage: 0.6,
                    categoryPercentage: 0.7
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 800, easing: 'easeOutQuart' },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
                y: {
                    title: { display: true, text: 'Fraction (%)', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    min: 0, max: 100
                }
            },
            plugins: {
                legend: { labels: { color: '#cbd5e1', usePointStyle: true, padding: 12 } }
            }
        }
    });
}

function renderRadarChart(r) {
    const ctx = document.getElementById('radarChart');
    if (radarChart) radarChart.destroy();

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: r.labels,
            datasets: [
                {
                    label: 'Baseline',
                    data: r.baseline,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    pointBackgroundColor: '#3b82f6',
                    pointRadius: 4
                },
                {
                    label: 'Research',
                    data: r.research,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.2)',
                    borderWidth: 2,
                    pointBackgroundColor: '#8b5cf6',
                    pointRadius: 4
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 800 },
            scales: {
                r: {
                    beginAtZero: true, max: 100,
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    angleLines: { color: 'rgba(255,255,255,0.06)' },
                    pointLabels: { color: '#94a3b8', font: { size: 12 } },
                    ticks: { display: false }
                }
            },
            plugins: {
                legend: { labels: { color: '#cbd5e1', usePointStyle: true, padding: 12 } }
            }
        }
    });
}

async function loadValidationDiagnostics(userData) {
    if (diagnosticsLoaded) return;

    try {
        const [resRob, resTrans, resLat, resCmd] = await Promise.all([
            fetch('/api/starforge/validation/robustness').then(r => r.json()),
            fetch('/api/starforge/validation/transition').then(r => r.json()),
            fetch('/api/starforge/diagnostics/latent').then(r => r.json()),
            fetch('/api/starforge/diagnostics/cmd').then(r => r.json())
        ]);

        renderRobustnessChart(resRob);
        renderTransitionCards(resTrans);
        renderLatentChart(resLat);
        renderCmdChart(resCmd, userData);

        diagnosticsLoaded = true;
    } catch (e) {
        console.error("Failed to load diagnostics:", e);
    }
}

function renderRobustnessChart(data) {
    const ctx = document.getElementById('robustnessChart');
    if (robustnessChart) robustnessChart.destroy();

    const points = data.redshifts.map((rs, i) => ({ x: rs, y: data.maes[i] }));

    robustnessChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Prediction Error',
                data: points,
                backgroundColor: 'rgba(20, 184, 166, 0.4)',
                borderColor: 'rgba(20, 184, 166, 0.8)',
                pointRadius: 4,
                borderWidth: 1
            }, {
                type: 'line',
                label: 'Global Research MAE',
                data: [{ x: 0, y: data.global_mae }, { x: Math.max(...data.redshifts), y: data.global_mae }],
                borderColor: '#ef4444',
                borderDash: [5, 5],
                borderWidth: 2,
                pointRadius: 0,
                fill: false
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Redshift', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                y: { title: { display: true, text: 'Prediction Error (MAE per galaxy)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.04)' } }
            },
            plugins: { legend: { labels: { color: '#cbd5e1' } } }
        }
    });
}

function renderTransitionCards(data) {
    const container = document.getElementById('greenValleyContainer');
    container.innerHTML = '';

    data.candidates.forEach(c => {
        container.innerHTML += `
            <div class="gv-card">
                <h5>Galaxy ID: ${c.galaxy_id}</h5>
                <div class="gv-entropy">Entropy: ${c.entropy.toFixed(3)}</div>
                <p><strong>Young:</strong> ${c.young.toFixed(1)}%</p>
                <p><strong>Inter:</strong> ${c.inter.toFixed(1)}%</p>
                <p><strong>Old:</strong> ${c.old.toFixed(1)}%</p>
            </div>
        `;
    });
}

function renderLatentChart(data) {
    const ctx = document.getElementById('latentChart');
    if (latentChart) latentChart.destroy();

    const bgColors = data.points.map(p => {
        const v = Math.min(Math.max(p.young, 0), 1);
        return `rgba(${Math.floor(255 * (1 - v))}, ${Math.floor(100 + 100 * v)}, ${Math.floor(255 * v)}, 0.6)`;
    });

    latentChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Galaxies',
                data: data.points.map(p => ({ x: p.x, y: p.y })),
                backgroundColor: bgColors,
                pointRadius: 3,
                borderWidth: 0
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Latent Component 1', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                y: { title: { display: true, text: 'Latent Component 2', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.04)' } }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (ctx) {
                            const p = data.points[ctx.dataIndex];
                            return [
                                `Galaxy ID: ${p.id}`,
                                `Young: ${p.young.toFixed(3)}`,
                                `Inter: ${p.inter.toFixed(3)}`,
                                `Old: ${p.old.toFixed(3)}`,
                                `g-r: ${p.gr.toFixed(3)}`
                            ];
                        }
                    }
                }
            }
        }
    });
}

function renderCmdChart(data, userData) {
    const ctx = document.getElementById('cmdChart');
    if (cmdChart) cmdChart.destroy();

    const bgPoints = data.gr.map((gr, i) => ({ x: gr, y: data.r[i] }));
    const userGr = userData.g - userData.r;

    cmdChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Survey Galaxies',
                data: bgPoints,
                backgroundColor: 'rgba(148, 163, 184, 0.2)',
                pointRadius: 2,
                borderWidth: 0
            }, {
                label: 'User-defined Galaxy',
                data: [{ x: userGr, y: userData.r }],
                backgroundColor: '#ef4444',
                borderColor: '#ffffff',
                borderWidth: 1,
                pointRadius: 8,
                pointStyle: 'star'
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'g - r Color Index', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.04)' } },
                y: { title: { display: true, text: 'r-band Magnitude', color: '#94a3b8' }, reverse: true, grid: { color: 'rgba(255,255,255,0.04)' } }
            },
            plugins: { legend: { labels: { color: '#cbd5e1' } } }
        }
    });
}

