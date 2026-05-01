/* ═══════════════════════════════════════════
   QuasarWatch — Frontend Logic
   Search-based: resolve name → fetch Pan-STARRS → forecast
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
    const nav = btn.closest('.sf-tabs-nav');
    nav.querySelectorAll('.sf-tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    const container = document.getElementById('quasarResults');
    container.querySelectorAll('.sf-tab-content').forEach(tc => tc.classList.remove('active'));

    const targetId = btn.getAttribute('data-target');
    document.getElementById(targetId).classList.add('active');

    // Resize all Chart.js instances so hidden-tab charts render at correct size
    const allCharts = [
        obsChart, forecastChart, uncertaintyChart, attentionChart,
        uatForecastChart, uatUncertaintyChart, structureFunctionChart, magHistChart
    ];
    // Use a small delay to let the tab become visible before measuring dimensions
    requestAnimationFrame(() => {
        allCharts.forEach(c => { if (c) c.resize(); });
    });
}

let obsChart = null, forecastChart = null, uncertaintyChart = null, attentionChart = null;
let uatForecastChart = null, uatUncertaintyChart = null, structureFunctionChart = null, magHistChart = null;

// ── UI helpers ──

function showLoading(msg) {
    document.getElementById('quasarLoadingText').textContent = msg;
    document.getElementById('quasarLoading').classList.remove('hidden');
    document.getElementById('quasarResults').classList.add('hidden');
    document.getElementById('quasarError').classList.add('hidden');
}

function hideLoading() {
    document.getElementById('quasarLoading').classList.add('hidden');
}

function showError(msg) {
    hideLoading();
    const el = document.getElementById('quasarError');
    document.getElementById('quasarErrorText').textContent = msg;
    el.classList.remove('hidden');
}

function hideError() {
    document.getElementById('quasarError').classList.add('hidden');
}

// ── Quasar info panel ──

function showQuasarInfo(info) {
    const grid = document.getElementById('quasarInfoGrid');
    grid.innerHTML = `
        <div class="metric-card-sm">
            <div class="metric-label">Object</div>
            <div class="metric-value" style="font-size:1rem;">${info.name || info.query}</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">RA</div>
            <div class="metric-value">${info.ra.toFixed(5)}°</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">DEC</div>
            <div class="metric-value">${info.dec >= 0 ? '+' : ''}${info.dec.toFixed(5)}°</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">Redshift</div>
            <div class="metric-value">${info.redshift != null ? info.redshift.toFixed(6) : '—'}</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">Type</div>
            <div class="metric-value" style="font-size:0.85rem;">${info.object_type || 'Unknown'}</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">Data Source</div>
            <div class="metric-value" style="font-size:0.85rem;">Pan-STARRS DR2</div>
        </div>
    `;
    document.getElementById('quasarInfoPanel').classList.remove('hidden');
}

// ── Main search & forecast flow ──

async function searchAndForecast(queryName) {
    if (!queryName || !queryName.trim()) {
        return alert('Please enter a quasar name or ID.');
    }

    hideError();
    document.getElementById('quasarInfoPanel').classList.add('hidden');
    document.getElementById('quasarResults').classList.add('hidden');

    // Step 1: Resolve name → coordinates
    showLoading(`Resolving "${queryName}" via CDS/Sesame…`);

    let info;
    try {
        const res = await fetch(`/api/quasar/resolve?name=${encodeURIComponent(queryName)}`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Resolution failed (${res.status})`);
        }
        info = await res.json();
    } catch (e) {
        return showError(`Could not resolve "${queryName}": ${e.message}`);
    }

    showQuasarInfo(info);

    // Step 2: Fetch light curve from Pan-STARRS & run forecast
    showLoading(`Fetching Pan-STARRS DR2 photometry for RA=${info.ra.toFixed(4)}, DEC=${info.dec.toFixed(4)}…`);

    let data;
    try {
        const params = new URLSearchParams({
            ra: info.ra,
            dec: info.dec,
            name: info.name || queryName,
        });
        if (info.redshift != null) params.set('redshift', info.redshift);

        const res = await fetch(`/api/quasar/predict?${params}`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Prediction failed (${res.status})`);
        }
        data = await res.json();
    } catch (e) {
        return showError(`Forecast failed: ${e.message}`);
    }

    hideLoading();
    displayQuasarResults(data);
}

// ── Event listeners ──

document.getElementById('runQuasarBtn').addEventListener('click', () => {
    const name = document.getElementById('quasarSearchInput').value.trim();
    searchAndForecast(name);
});

document.getElementById('quasarSearchInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        e.preventDefault();
        const name = e.target.value.trim();
        searchAndForecast(name);
    }
});

// Suggestion chips
document.querySelectorAll('.quasar-chip').forEach(chip => {
    chip.addEventListener('click', () => {
        const name = chip.getAttribute('data-name');
        document.getElementById('quasarSearchInput').value = name;
        searchAndForecast(name);
    });
});

// ── File browse / CSV upload ──

document.getElementById('browseFileBtn').addEventListener('click', () => {
    document.getElementById('quasarFileInput').click();
});

document.getElementById('quasarFileInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    e.target.value = ''; // reset so same file can be re-selected

    hideError();
    document.getElementById('quasarInfoPanel').classList.add('hidden');
    document.getElementById('quasarResults').classList.add('hidden');
    showLoading(`Parsing "${file.name}" and running forecast…`);

    const formData = new FormData();
    formData.append('file', file);

    let data;
    try {
        const res = await fetch(`/api/quasar/upload?name=${encodeURIComponent(file.name.replace(/\.csv$/i, ''))}`, {
            method: 'POST',
            body: formData
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Upload failed (${res.status})`);
        }
        data = await res.json();
    } catch (err) {
        return showError(`File processing failed: ${err.message}`);
    }

    // Show file info panel
    const grid = document.getElementById('quasarInfoGrid');
    grid.innerHTML = `
        <div class="metric-card-sm">
            <div class="metric-label">File</div>
            <div class="metric-value" style="font-size:0.88rem;">${file.name}</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">Source</div>
            <div class="metric-value" style="font-size:0.88rem;">Local CSV</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">File Size</div>
            <div class="metric-value">${(file.size / 1024).toFixed(1)}</div>
            <div class="metric-sub">KB</div>
        </div>
    `;
    document.getElementById('quasarInfoPanel').classList.remove('hidden');
    hideLoading();
    displayQuasarResults(data);
});

// ── Display results (unchanged logic) ──

function displayQuasarResults(data) {
    document.getElementById('quasarResults').classList.remove('hidden');

    // Reveal UAT content div and hide placeholder
    const uatEmpty   = document.getElementById('uatEmptyState');
    const uatContent = document.getElementById('uatContent');
    if (uatEmpty)   uatEmpty.style.display   = 'none';
    if (uatContent) uatContent.style.display = 'block';

    const obs = data.observations;
    const results = data.results;

    // ── Observations tab metrics ──
    const meanMagObs = obs.mags.reduce((a, b) => a + b, 0) / obs.mags.length;
    const gaps = obs.time_gaps || [];
    const meanGap = gaps.length ? (gaps.reduce((a, b) => a + b, 0) / gaps.length).toFixed(1) : '—';
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
        <div class="metric-card-sm">
            <div class="metric-label">Mean Mag</div>
            <div class="metric-value">${meanMagObs.toFixed(3)}</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">Mean Gap</div>
            <div class="metric-value">${meanGap}</div>
            <div class="metric-sub">days</div>
        </div>
    `;
    document.getElementById('obsMetrics').innerHTML = metricsHtml;

    renderObsChart(obs);
    renderForecastChart(results);      // Forecast Comparison tab
    renderUncertaintyChart(results);   // Forecast Comparison tab

    // ── UAT-CTGRU Forecast tab ──
    renderUATForecastChart(results);
    renderUATUncertaintyChart(results);
    renderUATForecastStats(results, obs);
    renderStructureFunction(obs);
    renderMagHistogram(obs);

    // ── Attention tab ──
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

// ══════════════════════════════════════════════════
// UAT-CTGRU Forecast Tab — new functions
// ══════════════════════════════════════════════════

function renderUATForecastChart(results) {
    const ctx = document.getElementById('uatForecastChart');
    if (uatForecastChart) uatForecastChart.destroy();
    const uat = results['UAT-CTGRU'];
    if (!uat) return;

    const days = Array.from({ length: 365 }, (_, i) => i + 1);
    const mu = uat.mu;
    const sigma = uat.sigma;
    const color = '#2ea043';

    uatForecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: days,
            datasets: [
                {
                    label: 'UAT-CTGRU Forecast',
                    data: mu,
                    borderColor: color,
                    borderWidth: 2.5,
                    pointRadius: 0,
                    tension: 0.3,
                    fill: false,
                    order: 0
                },
                {
                    label: '95% CI',
                    data: mu.map((m, i) => m + 1.96 * sigma[i]),
                    borderColor: 'transparent',
                    backgroundColor: color + '28',
                    pointRadius: 0,
                    fill: '+1',
                    order: 1
                },
                {
                    label: '_lower',
                    data: mu.map((m, i) => m - 1.96 * sigma[i]),
                    borderColor: 'transparent',
                    backgroundColor: 'transparent',
                    pointRadius: 0,
                    fill: false,
                    order: 2
                }
            ]
        },
        options: {
            responsive: true,
            animation: { duration: 900, easing: 'easeOutQuart' },
            scales: {
                x: {
                    title: { display: true, text: 'Days into Future', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { maxTicksLimit: 13 }
                },
                y: {
                    title: { display: true, text: 'Predicted Magnitude (brighter ↑)', color: '#94a3b8' },
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
                    filter: item => item.datasetIndex === 0,
                    callbacks: {
                        label: ctx => {
                            const i = ctx.dataIndex;
                            const s = sigma[i];
                            return [
                                `Magnitude: ${mu[i].toFixed(3)}`,
                                `±1σ: [${(mu[i]-s).toFixed(3)}, ${(mu[i]+s).toFixed(3)}]`
                            ];
                        }
                    }
                }
            }
        }
    });
}

function renderUATUncertaintyChart(results) {
    const ctx = document.getElementById('uatUncertaintyChart');
    if (uatUncertaintyChart) uatUncertaintyChart.destroy();
    const uat = results['UAT-CTGRU'];
    if (!uat) return;

    const days = Array.from({ length: 365 }, (_, i) => i + 1);

    uatUncertaintyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: days,
            datasets: [{
                label: 'UAT-CTGRU σ',
                data: uat.sigma,
                borderColor: '#2ea043',
                backgroundColor: 'rgba(46,160,67,0.10)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            animation: { duration: 600 },
            scales: {
                x: {
                    title: { display: true, text: 'Days into Future', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { maxTicksLimit: 13 }
                },
                y: {
                    title: { display: true, text: 'Uncertainty σ (mag)', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                }
            },
            plugins: {
                legend: { labels: { color: '#cbd5e1', usePointStyle: true } },
                tooltip: {
                    backgroundColor: 'rgba(15,23,42,0.95)', titleColor: '#f1f5f9',
                    bodyColor: '#cbd5e1', cornerRadius: 8, padding: 10,
                    callbacks: { label: c => `σ = ${c.parsed.y.toFixed(4)} mag` }
                }
            }
        }
    });
}

function renderUATForecastStats(results, obs) {
    const uat = results['UAT-CTGRU'];
    if (!uat) return;
    const mu = uat.mu;
    const sigma = uat.sigma;

    const minMag = Math.min(...mu);
    const maxMag = Math.max(...mu);
    const meanMag = mu.reduce((a, b) => a + b, 0) / mu.length;
    const amplitude = maxMag - minMag;
    const brightestDay = mu.indexOf(minMag) + 1;
    const faintestDay  = mu.indexOf(maxMag) + 1;
    const finalSigma   = sigma[sigma.length - 1];

    const obsM = obs.mags;
    const obsMean = obsM.reduce((a, b) => a + b, 0) / obsM.length;
    const obsStd  = Math.sqrt(obsM.reduce((s, m) => s + (m - obsMean) ** 2, 0) / obsM.length);
    // Variability index: excess variance relative to mean (Eddington bias corrected proxy)
    const varIdx  = (obsStd / obsMean * 100).toFixed(2);

    // Confidence interval half-width at day 90 / 180 / 365
    const ci90  = (1.96 * sigma[89]).toFixed(3);
    const ci180 = (1.96 * sigma[179]).toFixed(3);
    const ci365 = (1.96 * sigma[364]).toFixed(3);

    document.getElementById('uatForecastStats').innerHTML = `
        <div class="metric-card-sm">
            <div class="metric-label">Brightest Predicted</div>
            <div class="metric-value">Day ${brightestDay}</div>
            <div class="metric-sub">${minMag.toFixed(3)} mag</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">Faintest Predicted</div>
            <div class="metric-value">Day ${faintestDay}</div>
            <div class="metric-sub">${maxMag.toFixed(3)} mag</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">Forecast Amplitude</div>
            <div class="metric-value">${amplitude.toFixed(3)}</div>
            <div class="metric-sub">Δ mag over year</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">Mean Forecast Mag</div>
            <div class="metric-value">${meanMag.toFixed(3)}</div>
            <div class="metric-sub">averaged over 365 d</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">Obs. Variability Index</div>
            <div class="metric-value">${varIdx}%</div>
            <div class="metric-sub">σ/μ × 100</div>
        </div>
        <div class="metric-card-sm">
            <div class="metric-label">±95% CI Half-Width</div>
            <div class="metric-value">${ci90} / ${ci180} / ${ci365}</div>
            <div class="metric-sub">at day 90 / 180 / 365</div>
        </div>
    `;
}

function renderStructureFunction(obs) {
    const ctx = document.getElementById('structureFunctionChart');
    if (structureFunctionChart) structureFunctionChart.destroy();

    const times = obs.times;
    const mags  = obs.mags;
    const n     = times.length;

    // Compute all pairs
    const pairs = [];
    for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
            const dt  = Math.abs(times[j] - times[i]);
            const dm2 = (mags[j] - mags[i]) ** 2;
            if (dt > 0) pairs.push({ dt, dm2 });
        }
    }
    if (pairs.length < 3) return;

    // Log-bin
    const dts    = pairs.map(p => p.dt);
    const logMin = Math.log10(Math.min(...dts));
    const logMax = Math.log10(Math.max(...dts));
    if (logMin >= logMax) return;

    const nBins  = 14;
    const bw     = (logMax - logMin) / nBins;
    const bins   = Array.from({ length: nBins }, (_, i) => ({ center: logMin + (i + 0.5) * bw, vals: [] }));

    pairs.forEach(({ dt, dm2 }) => {
        const idx = Math.min(Math.floor((Math.log10(dt) - logMin) / bw), nBins - 1);
        if (idx >= 0) bins[idx].vals.push(dm2);
    });

    const sfPts = bins
        .filter(b => b.vals.length >= 2)
        .map(b => ({
            x: parseFloat(Math.pow(10, b.center).toFixed(4)),
            y: parseFloat((b.vals.reduce((a, c) => a + c, 0) / b.vals.length).toFixed(6))
        }));

    structureFunctionChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'SF(τ)',
                data: sfPts,
                borderColor: '#6366f1',
                backgroundColor: 'rgba(99,102,241,0.65)',
                pointRadius: 5,
                pointHoverRadius: 7,
                showLine: true,
                tension: 0.2,
                borderWidth: 1.5
            }]
        },
        options: {
            responsive: true,
            animation: { duration: 600 },
            scales: {
                x: {
                    type: 'logarithmic',
                    title: { display: true, text: 'Time Lag τ (days)', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                },
                y: {
                    type: 'logarithmic',
                    title: { display: true, text: 'SF(τ) = ⟨Δm²⟩', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                }
            },
            plugins: {
                legend: { labels: { color: '#cbd5e1', usePointStyle: true } },
                tooltip: {
                    backgroundColor: 'rgba(15,23,42,0.95)', titleColor: '#f1f5f9',
                    bodyColor: '#cbd5e1', cornerRadius: 8, padding: 10,
                    callbacks: {
                        label: c => `τ = ${c.parsed.x.toFixed(1)} d, SF = ${c.parsed.y.toFixed(5)}`
                    }
                }
            }
        }
    });
}

function renderMagHistogram(obs) {
    const ctx = document.getElementById('magHistChart');
    if (magHistChart) magHistChart.destroy();

    const mags  = obs.mags;
    const bands = obs.bands;
    const min   = Math.min(...mags);
    const max   = Math.max(...mags);
    const nBins = 20;
    const bw    = (max - min) / nBins || 0.1;

    const labels   = Array.from({ length: nBins }, (_, i) => (min + (i + 0.5) * bw).toFixed(2));
    const counts   = new Array(nBins).fill(0);
    mags.forEach(m => {
        const idx = Math.min(Math.floor((m - min) / bw), nBins - 1);
        if (idx >= 0) counts[idx]++;
    });

    // Color by brightness (bright=green, faint=purple)
    const barColors = counts.map((_, i) => {
        const t = i / (nBins - 1);
        const r = Math.round(46  + t * (139 - 46));
        const g = Math.round(160 + t * (92  - 160));
        const b = Math.round(67  + t * (246 - 67));
        return `rgba(${r},${g},${b},0.75)`;
    });

    magHistChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Observation Count',
                data: counts,
                backgroundColor: barColors,
                borderColor: barColors.map(c => c.replace('0.75', '1')),
                borderWidth: 1,
                borderRadius: 3
            }]
        },
        options: {
            responsive: true,
            animation: { duration: 600 },
            scales: {
                x: {
                    title: { display: true, text: 'Magnitude (brighter → left)', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    reverse: true
                },
                y: {
                    title: { display: true, text: 'Number of Observations', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.04)' }
                }
            },
            plugins: {
                legend: { labels: { color: '#cbd5e1', usePointStyle: true } },
                tooltip: {
                    backgroundColor: 'rgba(15,23,42,0.95)', titleColor: '#f1f5f9',
                    bodyColor: '#cbd5e1', cornerRadius: 8, padding: 10,
                    callbacks: {
                        title: items => `Mag ≈ ${items[0].label}`,
                        label: c => `${c.parsed.y} observations`
                    }
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
                    let val = Math.pow(intensity, 0.7);
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
