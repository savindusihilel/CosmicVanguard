/* ═══════════════════════════════════════════
   StarCharacterizer — Frontend Logic
   ═══════════════════════════════════════════ */

// ── Chart.js Global Theme (identical to galaxy.js) ──
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;

let comparisonChart = null, radarChart = null;

// ══════════════════════
//  LIVE DERIVED CHIPS
// ══════════════════════
function updateDerivedChips() {
    const u = parseFloat(document.getElementById('sc_u').value) || 0;
    const g = parseFloat(document.getElementById('sc_g').value) || 0;
    const r = parseFloat(document.getElementById('sc_r').value) || 0;
    const i = parseFloat(document.getElementById('sc_i').value) || 0;
    const z = parseFloat(document.getElementById('sc_z').value) || 0;

    document.getElementById('chip_ug').textContent = (u - g).toFixed(2);
    document.getElementById('chip_ri').textContent = (r - i).toFixed(2);
    document.getElementById('chip_iz').textContent = (i - z).toFixed(2);
}

['sc_u', 'sc_g', 'sc_r', 'sc_i', 'sc_z', 'sc_redshift'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateDerivedChips);
});

// ══════════════════════
//  PRESET LOADERS
// ══════════════════════
function fillPreset(u, g, r, i, z, redshift) {
    document.getElementById('sc_u').value       = u;
    document.getElementById('sc_g').value       = g;
    document.getElementById('sc_r').value       = r;
    document.getElementById('sc_i').value       = i;
    document.getElementById('sc_z').value       = z;
    document.getElementById('sc_redshift').value = redshift;
    updateDerivedChips();
}

document.getElementById('preset-starburst').addEventListener('click', () =>
    fillPreset(20.1, 20.5, 20.3, 20.1, 20.0, 0.08));

document.getElementById('preset-elliptical').addEventListener('click', () =>
    fillPreset(21.5, 20.2, 19.5, 19.1, 18.9, 0.10));

document.getElementById('preset-greenvalley').addEventListener('click', () =>
    fillPreset(20.8, 20.0, 19.6, 19.3, 19.1, 0.12));

document.getElementById('preset-disk').addEventListener('click', () =>
    fillPreset(20.3, 19.9, 19.5, 19.2, 19.0, 0.09));

// ══════════════════════
//  LOADING STATE
// ══════════════════════
function setLoading(isLoading) {
    const loader   = document.getElementById('loading');
    const results  = document.getElementById('resultsContent');
    const empty    = document.getElementById('emptyState');

    if (isLoading) {
        loader.classList.remove('hidden');
        empty.classList.add('hidden');
        if (!results.classList.contains('hidden')) results.classList.add('hidden');
    } else {
        loader.classList.add('hidden');
    }
}

// ══════════════════════
//  FORM SUBMIT
// ══════════════════════
document.getElementById('scForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    setLoading(true);

    const data = {
        u:        parseFloat(document.getElementById('sc_u').value),
        g:        parseFloat(document.getElementById('sc_g').value),
        r:        parseFloat(document.getElementById('sc_r').value),
        i:        parseFloat(document.getElementById('sc_i').value),
        z:        parseFloat(document.getElementById('sc_z').value),
        redshift: parseFloat(document.getElementById('sc_redshift').value)
    };

    try {
        const res = await fetch('/api/starcharacterizer/predict', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify(data)
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || res.statusText);
        }
        const result = await res.json();
        displayResults(result);
    } catch (err) {
        alert('Characterization failed: ' + err.message);
    } finally {
        setLoading(false);
    }
});

// ══════════════════════
//  DISPLAY RESULTS
// ══════════════════════
function displayResults(r) {
    document.getElementById('emptyState').classList.add('hidden');
    document.getElementById('resultsContent').classList.remove('hidden');

    renderPopBars('gmmBars', r.labels, r.gmm_fractions);
    renderPopBars('mlpBars', r.labels, r.mlp_fractions);
    renderComparisonChart(r);
    renderRadarChart(r);
    renderDominantCard(r);
    renderEntropyBar(r);
    renderMetrics(r);
}

// ══════════════════════
//  POPULATION BARS
//  (identical pattern to starforge.js renderPopBars)
// ══════════════════════
function renderPopBars(containerId, labels, values) {
    const container  = document.getElementById(containerId);
    const classNames = ['fill-young', 'fill-inter', 'fill-old'];

    container.innerHTML = '';

    labels.forEach((label, i) => {
        const val = values[i];
        const div = document.createElement('div');
        div.className = 'sf-progress-item';
        div.innerHTML = `
            <div class="sf-progress-label">
                <strong>${label}</strong>
                <span style="font-family:'JetBrains Mono',monospace;">${val.toFixed(2)}%</span>
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

// ══════════════════════
//  COMPARISON BAR CHART
// ══════════════════════
function renderComparisonChart(r) {
    const ctx = document.getElementById('comparisonChart');
    if (comparisonChart) comparisonChart.destroy();

    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: r.labels,
            datasets: [
                {
                    label: 'GMM Soft Assignment',
                    data: r.gmm_fractions,
                    backgroundColor: 'rgba(59, 130, 246, 0.5)',
                    borderColor: '#3b82f6',
                    borderWidth: 1,
                    borderRadius: 4,
                    barPercentage: 0.6,
                    categoryPercentage: 0.7
                },
                {
                    label: 'MLP / Split-VAE',
                    data: r.mlp_fractions,
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
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 800, easing: 'easeOutQuart' },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
                y: {
                    title: { display: true, text: 'Fraction (%)', color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    min: 0, max: 100,
                    ticks: { color: '#64748b' }
                }
            },
            plugins: {
                legend: { position: 'top', labels: { color: '#cbd5e1', usePointStyle: true, padding: 12 } },
                tooltip: {
                    backgroundColor: 'rgba(15,23,42,0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255,255,255,0.08)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 10,
                    callbacks: { label: c => `${c.dataset.label}: ${c.raw.toFixed(1)}%` }
                }
            }
        }
    });
}

// ══════════════════════
//  RADAR CHART
// ══════════════════════
function renderRadarChart(r) {
    const ctx = document.getElementById('radarChart');
    if (radarChart) radarChart.destroy();

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: r.labels,
            datasets: [
                {
                    label: 'GMM',
                    data: r.gmm_fractions,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.08)',
                    borderWidth: 2,
                    pointBackgroundColor: '#3b82f6',
                    pointRadius: 3
                },
                {
                    label: 'Split-VAE (MLP)',
                    data: r.mlp_fractions,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.12)',
                    borderWidth: 2,
                    pointBackgroundColor: '#8b5cf6',
                    pointRadius: 3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 800 },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    grid: { color: 'rgba(255,255,255,0.06)' },
                    angleLines: { color: 'rgba(255,255,255,0.06)' },
                    pointLabels: { color: '#94a3b8', font: { size: 11 } },
                    ticks: { display: false }
                }
            },
            plugins: {
                legend: { labels: { color: '#cbd5e1', usePointStyle: true, padding: 12 } }
            }
        }
    });
}

// ══════════════════════
//  DOMINANT CARD
// ══════════════════════
function renderDominantCard(r) {
    const interpretations = {
        'Young stars':        'Actively star-forming galaxy with significant recent star formation activity.',
        'Intermediate stars': 'Mixed-age stellar population suggesting a transitional evolutionary phase.',
        'Old stars':          'Passively evolving galaxy dominated by old, metal-rich stellar populations.'
    };

    const dominantPct  = Math.max(...r.mlp_fractions);
    const interpretation = interpretations[r.dominant] || '';

    document.getElementById('sc-dominant-name').textContent = r.dominant;
    document.getElementById('sc-dominant-pct').textContent  = `${dominantPct.toFixed(1)}%`;
    document.getElementById('sc-dominant-interp').textContent = interpretation;

    const badge = document.getElementById('sc-transitional-badge');
    if (r.entropy > 1.0) {
        badge.classList.remove('hidden');
    } else {
        badge.classList.add('hidden');
    }
}

// ══════════════════════
//  ENTROPY BAR
// ══════════════════════
function renderEntropyBar(r) {
    const fill   = document.getElementById('sc-entropy-fill');
    const target = Math.min((r.entropy / Math.log(3)) * 100, 100);

    // Color coding
    let color;
    if (r.entropy < 0.5) {
        color = 'var(--emerald)';
    } else if (r.entropy <= 1.0) {
        color = 'var(--amber)';
    } else {
        color = 'var(--red)';
    }

    fill.style.backgroundColor = color;

    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            fill.style.width = `${target}%`;
        });
    });

    document.getElementById('sc-entropy-val').textContent    = `H = ${r.entropy.toFixed(3)} nats`;
    document.getElementById('sc-certainty-val').textContent  = `${r.certainty_pct.toFixed(1)}% certain`;
}

// ══════════════════════
//  METRICS
// ══════════════════════
function renderMetrics(r) {
    document.getElementById('sc-agreement-val').textContent = `${r.agreement_pct.toFixed(1)}%`;
    document.getElementById('sc-certainty-chip').textContent = `${r.certainty_pct.toFixed(1)}%`;
}

// ── Init derived chips on load ──
updateDerivedChips();
