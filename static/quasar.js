/* ═══════════════════════════════════════════
   QuasarWatch — Frontend Logic
   ═══════════════════════════════════════════ */

Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;

const COLORS = {
    'UAT-CTGRU': '#2ea043',
    'Transformer': '#388bfd',
    'Basic RNN': '#f0883e'
};

// Tab logic for QuasarWatch
function switchQwTab(btn) {
    // 1. Remove active class from all buttons
    const nav = btn.closest('.sf-tabs-nav');
    nav.querySelectorAll('.sf-tab-btn').forEach(b => b.classList.remove('active'));

    // 2. Add active to clicked
    btn.classList.add('active');

    // 3. Hide all tab contents
    const container = document.getElementById('quasarResults');
    container.querySelectorAll('.sf-tab-content').forEach(tc => tc.classList.remove('active'));

    // 4. Show the target
    const targetId = btn.getAttribute('data-target');
    document.getElementById(targetId).classList.add('active');
}

let obsChart = null, forecastChart = null, uncertaintyChart = null, attentionChart = null;

// Load sample list
async function loadSamples() {
    try {
        const res = await fetch('/api/quasar/samples');
        const data = await res.json();
        const sel = document.getElementById('sampleSelector');
        data.samples.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s;
            opt.textContent = s;
            sel.appendChild(opt);
        });
    } catch (e) {
        console.warn('Failed to load samples:', e);
    }
}

// Run forecast
document.getElementById('runQuasarBtn').addEventListener('click', async () => {
    const sample = document.getElementById('sampleSelector').value;
    if (!sample) return alert('Please select a quasar sample first.');

    document.getElementById('quasarLoading').classList.remove('hidden');
    document.getElementById('quasarResults').classList.add('hidden');

    try {
        const res = await fetch(`/api/quasar/predict?sample=${encodeURIComponent(sample)}`);
        if (!res.ok) throw new Error('Prediction failed');
        const data = await res.json();
        displayQuasarResults(data);
    } catch (e) {
        alert('Forecast failed: ' + e.message);
    } finally {
        document.getElementById('quasarLoading').classList.add('hidden');
    }
});

function displayQuasarResults(data) {
    document.getElementById('quasarResults').classList.remove('hidden');

    const obs = data.observations;
    const results = data.results;

    // Metrics row
    const timeGaps = obs.time_gaps;
    const metricsHtml = `
        <div class="metric-card-sm">
            <div class="metric-label">Observations</div>
            <div class="metric-value">${obs.times.length}</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">Time Span</div>
            <div class="metric-value">${(obs.times[obs.times.length - 1] - obs.times[0]).toFixed(0)}</div>
            <div class="metric-sub">days</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">Bands</div>
            <div class="metric-value">${[...new Set(obs.bands)].sort().join(', ')}</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">Mag Range</div>
            <div class="metric-value">${Math.min(...obs.mags).toFixed(2)} – ${Math.max(...obs.mags).toFixed(2)}</div>
        </div>
    `;
    document.getElementById('obsMetrics').innerHTML = metricsHtml;

    // Observations chart
    renderObsChart(obs);

    // Forecast chart
    renderForecastChart(results);

    // Uncertainty chart
    renderUncertaintyChart(results);

    // Attention heatmap
    if (results['UAT-CTGRU'] && results['UAT-CTGRU'].attn) {
        document.getElementById('attentionSection').classList.remove('hidden');
        renderAttentionChart(results['UAT-CTGRU'].attn);
    }
}

function renderObsChart(obs) {
    const ctx = document.getElementById('obsChart');
    if (obsChart) obsChart.destroy();

    const filterColors = { g: '#4CAF50', r: '#F44336', i: '#FF9800', z: '#9C27B0', y: '#00BCD4' };
    const uniqueBands = [...new Set(obs.bands)].sort();

    const datasets = uniqueBands.map(band => {
        const points = [];
        for (let i = 0; i < obs.times.length; i++) {
            if (obs.bands[i] === band) {
                points.push({ x: obs.times[i], y: obs.mags[i] });
            }
        }
        return {
            label: `${band}-band`,
            data: points,
            backgroundColor: filterColors[band] || '#888',
            borderColor: filterColors[band] || '#888',
            pointRadius: 4,
            pointHoverRadius: 6,
            showLine: false
        };
    });

    obsChart = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            animation: { duration: 600 },
            scales: {
                x: {
                    title: { display: true, text: 'MJD (Modified Julian Date)', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                },
                y: {
                    title: { display: true, text: 'Magnitude (brighter ↑)', color: '#94a3b8' },
                    reverse: true,
                    grid: { color: 'rgba(255,255,255,0.04)' }
                }
            },
            plugins: {
                legend: { labels: { color: '#cbd5e1', usePointStyle: true, padding: 12 } },
                tooltip: {
                    backgroundColor: 'rgba(15,23,42,0.95)', titleColor: '#f1f5f9',
                    bodyColor: '#cbd5e1', borderColor: 'rgba(255,255,255,0.08)',
                    borderWidth: 1, cornerRadius: 8, padding: 10
                }
            }
        }
    });
}

function renderForecastChart(results) {
    const ctx = document.getElementById('forecastChart');
    if (forecastChart) forecastChart.destroy();

    const days = Array.from({ length: 365 }, (_, i) => i);
    const datasets = [];

    const modelOrder = ['UAT-CTGRU', 'Transformer', 'Basic RNN'];
    const lineStyles = { 'UAT-CTGRU': [], 'Transformer': [8, 4], 'Basic RNN': [4, 4] };

    modelOrder.forEach(name => {
        if (!results[name]) return;
        const mu = results[name].mu;
        const sigma = results[name].sigma;
        const color = COLORS[name];

        datasets.push({
            label: name,
            data: mu,
            borderColor: color,
            borderWidth: name === 'UAT-CTGRU' ? 2.5 : 1.8,
            borderDash: lineStyles[name],
            pointRadius: 0,
            tension: 0.3,
            fill: false
        });

        // Confidence interval for UAT-CTGRU
        if (name === 'UAT-CTGRU') {
            datasets.push({
                label: 'UAT 95% CI',
                data: mu.map((m, i) => m + 1.96 * sigma[i]),
                borderColor: 'transparent',
                backgroundColor: color + '30',
                pointRadius: 0,
                fill: '+1'
            });
            datasets.push({
                label: '_lower',
                data: mu.map((m, i) => m - 1.96 * sigma[i]),
                borderColor: 'transparent',
                backgroundColor: 'transparent',
                pointRadius: 0,
                fill: false
            });
        }
    });

    forecastChart = new Chart(ctx, {
        type: 'line',
        data: { labels: days, datasets },
        options: {
            responsive: true,
            animation: { duration: 800, easing: 'easeOutQuart' },
            scales: {
                x: {
                    title: { display: true, text: 'Days into Future', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { maxTicksLimit: 12 }
                },
                y: {
                    title: { display: true, text: 'Predicted Magnitude', color: '#94a3b8' },
                    reverse: true,
                    grid: { color: 'rgba(255,255,255,0.04)' }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#cbd5e1', usePointStyle: true, padding: 12,
                        filter: item => !item.text.startsWith('_')
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(15,23,42,0.95)', titleColor: '#f1f5f9',
                    bodyColor: '#cbd5e1', borderColor: 'rgba(255,255,255,0.08)',
                    borderWidth: 1, cornerRadius: 8, padding: 10,
                    filter: item => !item.dataset.label.startsWith('_')
                }
            }
        }
    });
}

function renderUncertaintyChart(results) {
    const ctx = document.getElementById('uncertaintyChart');
    if (uncertaintyChart) uncertaintyChart.destroy();

    const days = Array.from({ length: 365 }, (_, i) => i);
    const datasets = [];

    ['UAT-CTGRU', 'Transformer', 'Basic RNN'].forEach(name => {
        if (!results[name]) return;
        datasets.push({
            label: `${name} σ`,
            data: results[name].sigma,
            borderColor: COLORS[name],
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.3,
            fill: false
        });
    });

    uncertaintyChart = new Chart(ctx, {
        type: 'line',
        data: { labels: days, datasets },
        options: {
            responsive: true,
            animation: { duration: 600 },
            scales: {
                x: {
                    title: { display: true, text: 'Days into Future', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { maxTicksLimit: 12 }
                },
                y: {
                    title: { display: true, text: 'Predicted Uncertainty (σ)', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                }
            },
            plugins: {
                legend: { labels: { color: '#cbd5e1', usePointStyle: true, padding: 12 } },
                tooltip: {
                    backgroundColor: 'rgba(15,23,42,0.95)', titleColor: '#f1f5f9',
                    bodyColor: '#cbd5e1', cornerRadius: 8, padding: 10
                }
            }
        }
    });
}

function renderAttentionChart(attn) {
    const ctx = document.getElementById('attentionChart');
    if (attentionChart) attentionChart.destroy();

    const nForecast = attn.length;
    const nHistory = attn[0].length;

    // Convert to scatter data with color intensity
    const points = [];
    let maxW = 0;
    for (let f = 0; f < nForecast; f++) {
        for (let h = 0; h < nHistory; h++) {
            const w = attn[f][h];
            if (w > maxW) maxW = w;
            points.push({ x: h, y: f, w });
        }
    }

    attentionChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Attention Weight',
                data: points.map(p => ({ x: p.x, y: p.y })),
                backgroundColor: points.map(p => {
                    const intensity = maxW > 0 ? p.w / maxW : 0;
                    const viridis = [
                        [68, 1, 84], [72, 33, 115], [67, 62, 133], [56, 88, 140],
                        [45, 112, 142], [37, 133, 142], [30, 155, 138], [42, 176, 127],
                        [82, 197, 105], [134, 213, 73], [194, 223, 35], [253, 231, 37]
                    ];
                    let val = Math.pow(intensity, 0.7); // slightly boost lower values for visibility
                    val = Math.max(0, Math.min(1, val));
                    if (val === 0) return 'rgba(255,255,255,0.02)';
                    const idx = val * (viridis.length - 1);
                    const i = Math.floor(idx);
                    const f = idx - i;
                    if (i >= viridis.length - 1) return `rgba(${viridis[i].join(',')}, 0.95)`;
                    const [r1, g1, b1] = viridis[i];
                    const [r2, g2, b2] = viridis[i + 1];
                    const r = Math.round(r1 + f * (r2 - r1));
                    const g = Math.round(g1 + f * (g2 - g1));
                    const b = Math.round(b1 + f * (b2 - b1));
                    return `rgba(${r}, ${g}, ${b}, 0.95)`;
                }),
                pointRadius: Math.max(2, Math.min(6, 300 / nHistory)),
                pointHoverRadius: 4
            }]
        },
        options: {
            responsive: true,
            animation: false,
            scales: {
                x: {
                    title: { display: true, text: 'History Timestep', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    min: 0, max: nHistory - 1
                },
                y: {
                    title: { display: true, text: 'Forecast Step', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    min: 0, max: nForecast - 1,
                    reverse: false
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(15,23,42,0.95)',
                    callbacks: {
                        label: ctx => {
                            const idx = ctx.dataIndex;
                            return `History: ${points[idx].x}, Forecast: ${points[idx].y}, Weight: ${points[idx].w.toFixed(4)}`;
                        }
                    }
                }
            }
        }
    });
}

// Init
loadSamples();
