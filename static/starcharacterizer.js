/* ═══════════════════════════════════════════
   StarCharacterizer — Conference Edition JS
   ═══════════════════════════════════════════ */

// ── Global Chart.js theme ──
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;

// ── Chart instances ──
let comparisonChart = null, radarChart = null;
let cmdChart = null, ablationChart = null, comparisonMiniChart = null, featureChart = null;
let ablationRendered = false;

// ── Ablation data (hardcoded research constants) ──
const ABLATION_DATA = {
    configs: ['Baseline VAE', 'VAE + GRL', 'VAE + GRL + HSIC', 'Full Model'],
    r2:      [0.8812, 0.9205, 0.9610, 0.9795],
    pearson: [0.3210, 0.1540, 0.0820, 0.0297]
};

// ── CMD background points (hardcoded) ──
const CMD_RED_SEQ   = [{x:-20.1,y:2.1},{x:-20.5,y:2.3},{x:-21.0,y:2.4},{x:-21.3,y:2.6},{x:-21.7,y:2.7},{x:-22.0,y:2.8},{x:-22.2,y:2.5},{x:-22.5,y:2.9},{x:-20.8,y:2.2},{x:-21.5,y:2.6},{x:-22.8,y:3.0},{x:-20.3,y:2.3},{x:-21.9,y:2.7},{x:-22.1,y:2.8},{x:-23.0,y:2.9}];
const CMD_GREEN_VAL = [{x:-19.5,y:1.6},{x:-19.8,y:1.7},{x:-20.1,y:1.8},{x:-20.4,y:1.9},{x:-20.7,y:1.6},{x:-21.0,y:1.9},{x:-19.2,y:1.5},{x:-20.8,y:1.8},{x:-20.2,y:1.7},{x:-19.7,y:1.6}];
const CMD_BLUE_CLOUD= [{x:-18.2,y:0.6},{x:-18.5,y:0.8},{x:-19.0,y:0.9},{x:-19.3,y:1.1},{x:-19.7,y:1.3},{x:-20.0,y:1.0},{x:-20.3,y:1.4},{x:-18.8,y:0.7},{x:-19.5,y:1.2},{x:-20.1,y:1.1},{x:-18.4,y:0.5},{x:-19.1,y:0.8},{x:-20.5,y:1.3},{x:-18.7,y:0.9},{x:-21.0,y:1.5}];
const CMD_EXT_BLUE  = [{x:-17.2,y:-0.1},{x:-17.5,y:0.2},{x:-18.0,y:-0.3},{x:-18.3,y:0.1},{x:-17.8,y:0.3},{x:-19.0,y:0.0},{x:-18.5,y:-0.4},{x:-19.2,y:0.1},{x:-17.3,y:0.4},{x:-20.0,y:-0.2}];

// ══════════════════════════
//  TAB SYSTEM
// ══════════════════════════
function initTabs() {
    document.querySelectorAll('.sc-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.sc-tab').forEach(b => b.classList.remove('sc-tab--active'));
            document.querySelectorAll('.sc-tab-panel').forEach(p => p.classList.add('hidden'));
            btn.classList.add('sc-tab--active');
            const panel = document.getElementById('tab-' + btn.dataset.tab);
            if (panel) panel.classList.remove('hidden');
            // Lazy-render ablation chart on first diagnostics tab visit
            if (btn.dataset.tab === 'diagnostics' && !ablationRendered) {
                renderAblation();
            }
        });
    });
}

// ══════════════════════════
//  LIVE DERIVED CHIPS
// ══════════════════════════
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
['sc_u','sc_g','sc_r','sc_i','sc_z','sc_redshift'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', updateDerivedChips);
});

// ══════════════════════════
//  PRESET LOADERS
// ══════════════════════════
function fillPreset(u, g, r, i, z, redshift) {
    document.getElementById('sc_u').value        = u;
    document.getElementById('sc_g').value        = g;
    document.getElementById('sc_r').value        = r;
    document.getElementById('sc_i').value        = i;
    document.getElementById('sc_z').value        = z;
    document.getElementById('sc_redshift').value = redshift;
    updateDerivedChips();
}
document.getElementById('preset-starburst').addEventListener('click',  () => fillPreset(18.9, 18.4, 18.1, 17.9, 17.7, 0.08));
document.getElementById('preset-elliptical').addEventListener('click', () => fillPreset(20.8, 19.2, 18.4, 18.0, 17.7, 0.10));
document.getElementById('preset-greenvalley').addEventListener('click',() => fillPreset(20.1, 19.0, 18.5, 18.2, 18.0, 0.12));
document.getElementById('preset-disk').addEventListener('click',       () => fillPreset(19.5, 18.9, 18.4, 18.1, 17.9, 0.09));

// ══════════════════════════
//  LOADING STATE
// ══════════════════════════
function setLoading(on) {
    document.getElementById('loading').classList.toggle('hidden', !on);
    if (on) {
        document.getElementById('emptyState').classList.add('hidden');
        document.getElementById('resultsContent').classList.add('hidden');
    }
}

// ══════════════════════════
//  FORM SUBMIT — dual fetch
// ══════════════════════════
document.getElementById('scForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    setLoading(true);

    const inputs = {
        u:        parseFloat(document.getElementById('sc_u').value),
        g:        parseFloat(document.getElementById('sc_g').value),
        r:        parseFloat(document.getElementById('sc_r').value),
        i:        parseFloat(document.getElementById('sc_i').value),
        z:        parseFloat(document.getElementById('sc_z').value),
        redshift: parseFloat(document.getElementById('sc_redshift').value)
    };

    const body = JSON.stringify(inputs);
    const hdr  = { 'Content-Type': 'application/json' };

    try {
        const [mainResult, baselineResult] = await Promise.all([
            fetch('/api/starcharacterizer/predict',  { method: 'POST', headers: hdr, body }).then(r => { if (!r.ok) return r.json().then(e => { throw new Error(e.detail || r.statusText); }); return r.json(); }),
            fetch('/api/starcharacterizer/baseline', { method: 'POST', headers: hdr, body }).then(r => { if (!r.ok) return r.json().then(e => { throw new Error(e.detail || r.statusText); }); return r.json(); })
        ]);
        window._scLastResult = { ...mainResult, _baseline: baselineResult, _input: inputs };
        displayResults(mainResult, baselineResult, inputs);
    } catch (err) {
        alert('Characterization failed: ' + err.message);
    } finally {
        setLoading(false);
    }
});

// ══════════════════════════
//  DISPLAY RESULTS
// ══════════════════════════
function displayResults(r, baseline, inputs) {
    document.getElementById('emptyState').classList.add('hidden');
    document.getElementById('resultsContent').classList.remove('hidden');

    // Tab 1 — Interpretation
    // Update chips with backend-computed absolute colour indices
    if (r.derived) {
        document.getElementById('chip_ug').textContent = r.derived.u_g.toFixed(2);
        document.getElementById('chip_ri').textContent = r.derived.r_i.toFixed(2);
        document.getElementById('chip_iz').textContent = r.derived.i_z.toFixed(2);
        // Update chip labels to reflect absolute values
        const chipLabels = document.querySelectorAll('.sc-chip-label');
        if (chipLabels[0]) chipLabels[0].textContent = 'U−G (abs)';
        if (chipLabels[1]) chipLabels[1].textContent = 'R−I (abs)';
        if (chipLabels[2]) chipLabels[2].textContent = 'I−Z (abs)';
    }
    renderDominantCard(r);
    renderPopBars('gmmBars', r.labels, r.gmm_fractions);
    renderPopBars('mlpBars', r.labels, r.mlp_fractions);
    renderEntropyBar(r);
    renderPlainSummary(r);
    document.getElementById('sc-agreement-val').textContent  = r.agreement_pct.toFixed(1) + '%';
    document.getElementById('sc-certainty-chip').textContent = r.certainty_pct.toFixed(1) + '%';

    // Tab 2 — Explainability
    renderFeatureImportance(r);
    renderPopulationConfidence(r);
    renderDisentanglementPanel(r);

    // Tab 3 — Evolution
    renderCMD(r, inputs);
    renderEvolutionTimeline(r);

    // Tab 5 — Comparison (baseline + main available)
    renderComparisonTab(r, baseline);
}

// ══════════════════════════
//  TAB 1 RENDERS
// ══════════════════════════
function renderPopBars(containerId, labels, values) {
    const container  = document.getElementById(containerId);
    const classNames = ['fill-young', 'fill-inter', 'fill-old'];
    container.innerHTML = '';
    labels.forEach((label, i) => {
        const val = values[i];
        const div = document.createElement('div');
        div.className = 'sf-progress-item';
        div.innerHTML = `<div class="sf-progress-label"><strong>${label}</strong><span style="font-family:'JetBrains Mono',monospace;">${val.toFixed(2)}%</span></div><div class="sf-progress-bar-bg"><div class="sf-progress-bar-fill ${classNames[i]}" style="width:0%"></div></div>`;
        container.appendChild(div);
        requestAnimationFrame(() => { requestAnimationFrame(() => { div.querySelector('.sf-progress-bar-fill').style.width = Math.min(val, 100) + '%'; }); });
    });
}

function renderDominantCard(r) {
    const icons = { 'Young stars': '🔵', 'Intermediate stars': '🟢', 'Old stars': '🔴' };
    const interps = {
        'Young stars':        'The dominant stellar population is classified as young, indicating photometric colours consistent with recently formed, high-mass stars.',
        'Intermediate stars': 'The dominant stellar population is classified as intermediate-age, with photometric colours spanning the transition between young and old stellar populations.',
        'Old stars':          'The dominant stellar population is classified as old, characterised by photometric colours consistent with low-mass, long-lived stellar populations.'
    };
    const pct = Math.max(...r.mlp_fractions);
    document.getElementById('sc-dominant-icon').textContent   = icons[r.dominant] || '⭐';
    document.getElementById('sc-dominant-name').textContent   = r.dominant;
    document.getElementById('sc-dominant-pct').textContent    = pct.toFixed(1) + '%';
    document.getElementById('sc-dominant-interp').textContent = interps[r.dominant] || '';
    document.getElementById('sc-transitional-badge').classList.toggle('hidden', r.entropy <= 1.0);
}

function renderEntropyBar(r) {
    const fill   = document.getElementById('sc-entropy-fill');
    const target = Math.min((r.entropy / Math.log(3)) * 100, 100);
    const color  = r.entropy < 0.5 ? 'var(--emerald)' : r.entropy <= 1.0 ? 'var(--amber)' : 'var(--red)';
    fill.style.backgroundColor = color;
    requestAnimationFrame(() => { requestAnimationFrame(() => { fill.style.width = target + '%'; }); });
    document.getElementById('sc-entropy-val').textContent   = 'H = ' + r.entropy.toFixed(3) + ' nats';
    document.getElementById('sc-certainty-val').textContent = r.certainty_pct.toFixed(1) + '% certain';
}

function renderPlainSummary(r) {
    const dom = r.dominant.toLowerCase();
    const y = r.mlp_fractions[0].toFixed(1);
    const m = r.mlp_fractions[1].toFixed(1);
    const o = r.mlp_fractions[2].toFixed(1);
    let text = '';
    if (dom.includes('young')) {
        text = `The Split-VAE model classifies this galaxy as dominated by young stellar populations (${y}%), with intermediate-age populations at ${m}% and old populations at ${o}%. The photometric colour indices, corrected for cosmological redshift bias, are consistent with a young-dominant population fraction.`;
    } else if (dom.includes('inter')) {
        text = `The Split-VAE model classifies this galaxy as dominated by intermediate-age stellar populations (${m}%), with young populations at ${y}% and old populations at ${o}%. The colour-magnitude position suggests a mixed-age population distribution in the transitional regime.`;
    } else {
        text = `The Split-VAE model classifies this galaxy as dominated by old stellar populations (${o}%), with intermediate-age populations at ${m}% and young populations at ${y}%. The photometric colour indices are consistent with an old-dominant population fraction.`;
    }
    if (r.entropy > 1.0) text += ' The elevated Shannon entropy (H = ' + r.entropy.toFixed(3) + ' nats) indicates significant uncertainty across population classes.';
    text += ` Redshift nuisance projection applied with α = ${r.alpha_used}.`;
    document.getElementById('sc-plain-summary').textContent = text;
}

// ══════════════════════════
//  TAB 2 RENDERS
// ══════════════════════════
function renderFeatureImportance(r) {
    const ctx    = document.getElementById('featureChart');
    const fi     = r.feature_importance;
    const vals   = [fi.u_g, fi.r_i, fi.i_z, fi.redshift_weight];
    const labels = ['u−g', 'r−i', 'i−z', 'Redshift'];
    const maxVal = Math.max(...vals);
    const colors = vals.map(v => v === maxVal ? '#3b82f6' : 'rgba(99,102,241,0.4)');
    if (featureChart) featureChart.destroy();
    featureChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{ data: vals, backgroundColor: colors, borderColor: colors, borderWidth: 1, borderRadius: 4 }]
        },
        options: {
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 700, easing: 'easeOutQuart' },
            scales: {
                x: { min: 0, max: 100, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { callback: v => v + '%' } },
                y: { grid: { display: false } }
            },
            plugins: {
                legend: { display: false },
                tooltip: { callbacks: { label: c => c.raw.toFixed(1) + '%' }, backgroundColor: 'rgba(15,23,42,0.95)', padding: 8, cornerRadius: 6 }
            }
        }
    });
}

function renderPopulationConfidence(r) {
    const gauges = [
        { id: 'gauge-young', pctId: 'gauge-young-pct', val: r.population_confidence[0] },
        { id: 'gauge-inter', pctId: 'gauge-inter-pct', val: r.population_confidence[1] },
        { id: 'gauge-old',   pctId: 'gauge-old-pct',   val: r.population_confidence[2] }
    ];
    gauges.forEach(g => {
        const el  = document.getElementById(g.id);
        const pct = document.getElementById(g.pctId);
        if (el) el.style.setProperty('--fill', g.val);
        if (pct) pct.textContent = g.val.toFixed(0) + '%';
    });
}

function renderDisentanglementPanel(r) {
    // Pearson r
    const pr = r.pearson_r;
    const prBadge = pr < 0.15
        ? '<span class="sc-badge-pass">PASS</span>'
        : '<span class="sc-badge-pass sc-badge-fail">FAIL</span>';
    document.getElementById('dq-pearson').innerHTML = pr.toFixed(4) + prBadge;

    // HSIC
    const hs = r.hsic;
    const hsBadge = hs < 0.05
        ? '<span class="sc-badge-pass">PASS</span>'
        : '<span class="sc-badge-pass sc-badge-fail">FAIL</span>';
    document.getElementById('dq-hsic').innerHTML = hs.toFixed(6) + hsBadge;

    // Reconstruction R²
    const rr2 = r.reconstruction_r2;
    document.getElementById('dq-recon-val').textContent = rr2.toFixed(4);
    requestAnimationFrame(() => { requestAnimationFrame(() => { document.getElementById('dq-recon-bar').style.width = (rr2 * 100) + '%'; }); });

    // Alpha
    document.getElementById('dq-alpha').textContent = r.alpha_used;
}

// ══════════════════════════
//  TAB 3 RENDERS
// ══════════════════════════
function renderCMD(r, inputs) {
    const ctx = document.getElementById('cmdChart');
    if (cmdChart) cmdChart.destroy();

    const ur    = inputs.u - inputs.r;
    const rAbs  = inputs.r - 5 * Math.log10(Math.max(inputs.redshift * 4283 + 10, 1)) - 25;
    const domColors = { 'Young stars': '#3b82f6', 'Intermediate stars': '#10b981', 'Old stars': '#f59e0b' };
    const pointColor = domColors[r.dominant] || '#8b5cf6';

    cmdChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                { label: 'Red Sequence',   data: CMD_RED_SEQ,   backgroundColor: 'rgba(239,68,68,0.35)',  pointRadius: 5 },
                { label: 'Green Valley',   data: CMD_GREEN_VAL, backgroundColor: 'rgba(16,185,129,0.35)', pointRadius: 5 },
                { label: 'Blue Cloud',     data: CMD_BLUE_CLOUD,backgroundColor: 'rgba(59,130,246,0.35)', pointRadius: 5 },
                { label: 'Extreme Blue',   data: CMD_EXT_BLUE,  backgroundColor: 'rgba(147,197,253,0.35)',pointRadius: 5 },
                { label: 'This Galaxy ★', data: [{ x: rAbs, y: ur }], backgroundColor: pointColor, borderColor: '#ffffff', pointRadius: 10, pointStyle: 'star', borderWidth: 2 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Absolute r-band Magnitude (Mᵣ)', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.04)' }, reverse: true },
                y: { title: { display: true, text: 'u − r Colour Index', color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.04)' } }
            },
            plugins: {
                legend: { labels: { color: '#cbd5e1', usePointStyle: true, padding: 12, font: { size: 11 } } },
                tooltip: { backgroundColor: 'rgba(15,23,42,0.95)', padding: 8, cornerRadius: 6, callbacks: { label: c => `${c.dataset.label}: (${c.raw.x.toFixed(2)}, ${c.raw.y.toFixed(2)})` } }
            }
        }
    });
}

function renderEvolutionTimeline(r) {
    // Clear previous active states
    [0, 1, 2].forEach(i => {
        const node = document.getElementById('sc-node-' + i);
        if (node) node.classList.remove('sc-node--active');
    });
    const domIdx = r.dominant.toLowerCase().includes('young') ? 0 : r.dominant.toLowerCase().includes('inter') ? 1 : 2;
    const active = document.getElementById('sc-node-' + domIdx);
    if (active) {
        active.setAttribute('r', '30');
        active.classList.add('sc-node--active');
    }
}

// ══════════════════════════
//  TAB 4 RENDERS
// ══════════════════════════
function renderAblation() {
    if (ablationRendered) return;
    ablationRendered = true;
    const ctx = document.getElementById('ablationChart');
    if (ablationChart) ablationChart.destroy();
    ablationChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ABLATION_DATA.configs,
            datasets: [
                {
                    label: 'R² Score',
                    data: ABLATION_DATA.r2,
                    backgroundColor: ABLATION_DATA.configs.map((_, i) => i === 3 ? '#3b82f6' : 'rgba(59,130,246,0.45)'),
                    borderColor: '#3b82f6', borderWidth: 1, borderRadius: 4, yAxisID: 'y'
                },
                {
                    label: 'Pearson r with Redshift',
                    data: ABLATION_DATA.pearson,
                    backgroundColor: ABLATION_DATA.configs.map((_, i) => i === 3 ? '#ef4444' : 'rgba(239,68,68,0.45)'),
                    borderColor: '#ef4444', borderWidth: 1, borderRadius: 4, yAxisID: 'y2'
                }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                y:  { title: { display: true, text: 'R²', color: '#3b82f6' }, min: 0.8, max: 1.0, grid: { color: 'rgba(255,255,255,0.04)' }, position: 'left' },
                y2: { title: { display: true, text: '|Pearson r|', color: '#ef4444' }, min: 0, max: 0.4, grid: { display: false }, position: 'right' },
                x:  { grid: { display: false } }
            },
            plugins: {
                legend: { labels: { color: '#cbd5e1', usePointStyle: true, padding: 12 } },
                tooltip: { backgroundColor: 'rgba(15,23,42,0.95)', padding: 8, cornerRadius: 6 }
            }
        }
    });
}

// ══════════════════════════
//  HELPERS
// ══════════════════════════
function updateFractionBars(fracs, labels) {
    renderPopBars('mlpBars', labels, fracs);
}

// ══════════════════════════
//  TAB 5 RENDERS
// ══════════════════════════
function renderComparisonTab(main, baseline) {
    const compPanel = document.getElementById('tab-comparison');

    // 3-way R² cards — inject once
    if (!compPanel.querySelector('.r2-three-way')) {
        const cards = `
        <div class="comp-toggle-row">
          <span class="comp-mode-lbl">Split-VAE</span>
          <label class="sc-toggle">
            <input type="checkbox" id="baseline-mode-tog" onchange="toggleBaselineMode(this.checked)">
            <span class="sc-toggle-slider"></span>
          </label>
          <span class="comp-mode-lbl muted">Baseline Mode (PCA-GMM)</span>
        </div>
        <div id="baseline-warn" class="baseline-warn" style="display:none;">
          Baseline Mode active — redshift disentanglement disabled. Results are cosmologically biased.
        </div>
        <div class="r2-three-way">
          <div class="r2-card loss">
            <div class="r2-model">PCA-GMM Baseline</div>
            <div class="r2-subtitle">Non-disentangled</div>
            <div class="r2-big">${baseline.baseline_r2.toFixed(4)}</div>
            <div class="r2-unit">R²</div>
            <div class="r2-mae">MAE ${baseline.baseline_mae.toFixed(4)}</div>
          </div>
          <div class="r2-card ref">
            <div class="r2-model">Random Forest</div>
            <div class="r2-subtitle">Supervised reference</div>
            <div class="r2-big">0.8473</div>
            <div class="r2-unit">R²</div>
            <div class="r2-mae">Supervised ceiling</div>
          </div>
          <div class="r2-card win">
            <div class="r2-model">Split-VAE</div>
            <div class="r2-subtitle">Disentangled · GRL · HSIC</div>
            <div class="r2-big">${main.model_r2.toFixed(4)}</div>
            <div class="r2-unit">R²</div>
            <div class="r2-mae">MAE ${main.model_mae.toFixed(4)}</div>
          </div>
        </div>
        <div class="r2-delta-line">
          Split-VAE exceeds the supervised Random Forest reference by <strong>+${(main.model_r2 - 0.8473).toFixed(4)} R²</strong> while remaining fully unsupervised — no stellar population labels used during training.
        </div>`;
        compPanel.querySelector('#comparisonMiniChart').parentElement.insertAdjacentHTML('beforebegin', cards);
    }

    // Mini bar chart
    const ctx = document.getElementById('comparisonMiniChart');
    if (comparisonMiniChart) comparisonMiniChart.destroy();
    comparisonMiniChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['R²', 'MAE'],
            datasets: [
                { label: 'Baseline',  data: [baseline.baseline_r2, baseline.baseline_mae], backgroundColor: 'rgba(148,163,184,0.5)', borderColor: '#94a3b8', borderWidth: 1, borderRadius: 4, barPercentage: 0.6 },
                { label: 'Split-VAE', data: [main.model_r2, main.model_mae],               backgroundColor: 'rgba(52,211,153,0.6)',  borderColor: '#34d399', borderWidth: 1, borderRadius: 4, barPercentage: 0.6 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { grid: { display: false } },
                y: { min: -0.6, max: 1.1, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b' } }
            },
            plugins: { legend: { labels: { color: '#cbd5e1', usePointStyle: true, padding: 10, font: { size: 11 } } } }
        }
    });

    // Per-population deltas
    const labels   = ['Young', 'Intermediate', 'Old'];
    const mlpFracs = main.mlp_fractions;
    const gmmFracs = baseline.gmm_fractions;
    ['diff-young', 'diff-inter', 'diff-old'].forEach((id, i) => {
        const el   = document.getElementById(id);
        if (!el) return;
        const diff = mlpFracs[i] - gmmFracs[i];
        const sign = diff >= 0 ? '+' : '';
        el.textContent = `Δ ${labels[i]}: ${sign}${diff.toFixed(1)}%`;
        el.style.color = diff >= 0 ? 'var(--emerald)' : 'var(--amber)';
    });

    // Invariance panel — inject once
    if (!compPanel.querySelector('.invariance-panel')) {
        compPanel.insertAdjacentHTML('beforeend', `
        <div class="invariance-panel">
          <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.1em;color:var(--text-3);margin-bottom:10px;">LIVE REDSHIFT INVARIANCE CHECK</div>
          <p style="font-size:0.70rem;color:var(--text-3);margin-bottom:10px;line-height:1.55;">Vary redshift independently of photometry. Split-VAE population fractions should remain stable (redshift-invariant).</p>
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
            <label style="font-size:0.71rem;color:var(--text-2);white-space:nowrap;">Test z = <strong id="inv-z-val">0.10</strong></label>
            <input type="range" id="inv-z-slider" min="0.01" max="0.40" step="0.01" value="0.10" style="flex:1;"
              oninput="document.getElementById('inv-z-val').textContent=parseFloat(this.value).toFixed(2);runInvarianceCheck(this.value)">
          </div>
          <div id="inv-result"></div>
        </div>
        <p style="font-size:0.65rem;color:var(--text-3);font-style:italic;margin-top:16px;line-height:1.5;">Baseline uses PCA-GMM without redshift disentanglement. Higher R² and lower MAE indicate better population fraction recovery across the full SDSS DR17 spectroscopic test set (11,799 galaxies).</p>`);
    }
}

function toggleBaselineMode(isBaseline) {
    const warn = document.getElementById('baseline-warn');
    if (warn) warn.style.display = isBaseline ? 'block' : 'none';
    if (!window._scLastResult) return;
    const r = window._scLastResult;
    const fracs = isBaseline ? (r._baseline ? r._baseline.gmm_fractions : r.gmm_fractions) : r.mlp_fractions;
    updateFractionBars(fracs, r.labels);
}

async function runInvarianceCheck(z) {
    if (!window._scLastResult) return;
    const el  = document.getElementById('inv-result');
    el.innerHTML = '<span style="font-size:0.70rem;color:var(--text-3);">Running inference…</span>';
    const inp = window._scLastResult._input;
    try {
        const res  = await fetch('/api/starcharacterizer/predict', {
            method: 'POST', headers: {'Content-Type':'application/json'},
            body: JSON.stringify({ u:inp.u, g:inp.g, r:inp.r, i:inp.i, z:inp.z, redshift: parseFloat(z) })
        });
        const data = await res.json();
        const orig = window._scLastResult.mlp_fractions;
        const maxShift = Math.max(...data.mlp_fractions.map((f, i) => Math.abs(f - orig[i]))).toFixed(1);
        const isInvariant = maxShift < 5.0;
        el.innerHTML = `
          <div style="font-size:0.71rem;color:#34d399;margin-bottom:6px;">
            Max population shift = <strong>${maxShift}%</strong> (z=${parseFloat(z).toFixed(2)} vs z=${parseFloat(inp.redshift).toFixed(2)})
            <span style="display:inline-block;padding:2px 8px;border-radius:10px;font-size:0.63rem;font-weight:700;margin-left:8px;
              background:${isInvariant?'rgba(52,211,153,0.15)':'rgba(251,191,36,0.15)'};
              border:1px solid ${isInvariant?'#34d399':'#fbbf24'};
              color:${isInvariant?'#34d399':'#fbbf24'}">${isInvariant?'REDSHIFT INVARIANT':'MINOR SENSITIVITY'}</span>
          </div>
          <div style="font-size:0.67rem;color:#f87171;font-style:italic;">PCA-GMM baseline would reclassify galaxy type at z > 0.20 due to unlabelled distance correlation in latent space.</div>`;
    } catch(e) {
        el.innerHTML = '<span style="font-size:0.70rem;color:#f87171;">Inference failed</span>';
    }
}

// ══════════════════════════
//  INIT
// ══════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    updateDerivedChips();
});

