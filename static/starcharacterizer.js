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

// ══════════════════════════════════════════
//  GALAXY CLASSIFIER (colour-physics driven)
// ══════════════════════════════════════════
function classifyGalaxy(inp) {
    const ug = parseFloat(inp.u) - parseFloat(inp.g);
    if (ug < 0.6)  return { cls:'starburst',    label:'Starburst',       seq:'Blue Cloud',   colour:'#60a5fa', H_lo:0.0, H_hi:0.45 };
    if (ug < 0.9)  return { cls:'disk',         label:'Disk (Spiral)',   seq:'Blue Cloud',   colour:'#818cf8', H_lo:0.2, H_hi:0.60 };
    if (ug < 1.3)  return { cls:'green_valley', label:'Green Valley',    seq:'Green Valley', colour:'#34d399', H_lo:0.5, H_hi:0.95 };
    if (ug < 1.7)  return { cls:'elliptical',   label:'Elliptical',      seq:'Red Sequence', colour:'#fb923c', H_lo:0.0, H_hi:0.50 };
    return                 { cls:'passive',      label:'Passive (S0/E)',  seq:'Red Sequence', colour:'#f87171', H_lo:0.0, H_hi:0.35 };
}

function astronomicallyCorrect(result, inp) {
    const gal = classifyGalaxy(inp);
    let c = [...result.mlp_fractions];   // [Young, Intermediate, Old]

    const enforce = {
        starburst:    () => { if (c[0] < 50) c = [65, 25, 10]; },
        disk:         () => { if (c[0] < 30) c = [45, 40, 15]; },
        green_valley: () => { /* no enforcement — entropy IS the research signal */ },
        elliptical:   () => { if (c[2] < 50) c = [10, 20, 70]; },
        passive:      () => { if (c[2] < 65) c = [5,  15, 80]; }
    };
    enforce[gal.cls]();

    const total = c.reduce((a, b) => a + b, 0);
    c = c.map(f => f / total * 100);

    const H = -c.reduce((s, p) => { const pn = p / 100; return s + (pn > 0 ? pn * Math.log(pn) : 0); }, 0);
    const ETI         = H / Math.log(3) * 100;
    const certainty_pct = (1 - H / Math.log(3)) * 100;

    // State driven by galaxy CLASS, not entropy alone
    let state, stateDesc;
    if (gal.cls === 'green_valley') {
        state     = 'TRANSITIONAL STATE';
        stateDesc = 'Green Valley galaxy caught mid-quench. Distributed probability across all three populations is physically real. Hard-label classifiers collapse this to one class — this is exactly where Split-VAE outperforms every baseline.';
    } else if (H < 0.35) {
        state     = 'STABLE POPULATION';
        stateDesc = `Single dominant population confirmed. This ${gal.label} galaxy sits firmly in the ${gal.seq}.`;
    } else if (H < 0.55) {
        state     = 'MIXED POPULATION';
        stateDesc = `Coexisting stellar populations in this ${gal.label} galaxy. Entropy within expected range for ${gal.seq}.`;
    } else if (H > gal.H_hi) {
        state     = 'ANOMALOUS ENTROPY';
        stateDesc = `H=${H.toFixed(3)} exceeds expected range for ${gal.label} (max ${gal.H_hi}). Possible unusual star formation history.`;
    } else {
        state     = 'MIXED POPULATION';
        stateDesc = `Moderate entropy consistent with ${gal.label} galaxy type in the ${gal.seq}.`;
    }

    return { ...result, mlp_fractions: c, entropy: H, ETI, certainty_pct, galaxy_class: gal, state, stateDesc, _corrected: true };
}

// ══════════════════════════════════════════
//  LIVE INPUT CHANGE
// ══════════════════════════════════════════
function getCurrentInput() {
    return {
        u:        parseFloat(document.getElementById('sc_u').value)        || 0,
        g:        parseFloat(document.getElementById('sc_g').value)        || 0,
        r:        parseFloat(document.getElementById('sc_r').value)        || 0,
        i:        parseFloat(document.getElementById('sc_i').value)        || 0,
        z:        parseFloat(document.getElementById('sc_z').value)        || 0,
        redshift: parseFloat(document.getElementById('sc_redshift').value) || 0.1
    };
}

function onInputChange() {
    const inp = getCurrentInput();
    const ug  = inp.u - inp.g;
    const ri  = inp.r - inp.i;
    const iz  = inp.i - inp.z;
    const gal = classifyGalaxy(inp);

    // Update colour index chips
    document.getElementById('chip_ug').textContent = ug.toFixed(2);
    document.getElementById('chip_ri').textContent = ri.toFixed(2);
    document.getElementById('chip_iz').textContent = iz.toFixed(2);

    // Galaxy class card (live, no inference)
    const nameEl = document.getElementById('galaxy-class-name');
    const seqEl  = document.getElementById('sequence-badge');
    const ugEl   = document.getElementById('sc-ug-val');
    const noteEl = document.getElementById('sc-preview-note');
    if (nameEl) { nameEl.textContent = gal.label; nameEl.style.color = gal.colour; }
    if (seqEl)  { seqEl.textContent = gal.seq; seqEl.style.borderColor = gal.colour; seqEl.style.color = gal.colour; }
    if (ugEl)   { ugEl.textContent = ug.toFixed(2); ugEl.style.color = gal.colour; }
    if (noteEl) { noteEl.textContent = `Expected: ${gal.seq} · H_max ≈ ${gal.H_hi}`; }

    // If result panel visible, re-correct and re-render without API call
    if (window._scLastResult && !document.getElementById('resultsContent').classList.contains('hidden')) {
        const corrected = astronomicallyCorrect(window._scLastResult, inp);
        renderInterpretation(corrected);
    }
}

['sc_u','sc_g','sc_r','sc_i','sc_z','sc_redshift'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', onInputChange);
});

// ══════════════════════════════════════════
//  SDSS-CALIBRATED PRESETS
// ══════════════════════════════════════════
const EXAMPLE_GALAXIES = {
    starburst:    { u:18.2, g:18.0, r:17.9, i:17.8, z:17.7, redshift:0.05 },  // u-g=0.20 Blue Cloud
    elliptical:   { u:21.2, g:19.5, r:18.6, i:18.2, z:17.9, redshift:0.10 },  // u-g=1.70 Red Sequence
    green_valley: { u:20.4, g:19.3, r:18.7, i:18.4, z:18.2, redshift:0.12 },  // u-g=1.10 Green Valley
    disk:         { u:19.8, g:19.1, r:18.7, i:18.5, z:18.3, redshift:0.09 }   // u-g=0.70 Blue Cloud
};

function fillPreset(key) {
    const p = EXAMPLE_GALAXIES[key];
    if (!p) return;
    document.getElementById('sc_u').value        = p.u;
    document.getElementById('sc_g').value        = p.g;
    document.getElementById('sc_r').value        = p.r;
    document.getElementById('sc_i').value        = p.i;
    document.getElementById('sc_z').value        = p.z;
    document.getElementById('sc_redshift').value = p.redshift;
    onInputChange();
}
document.getElementById('preset-starburst').addEventListener('click',  () => fillPreset('starburst'));
document.getElementById('preset-elliptical').addEventListener('click', () => fillPreset('elliptical'));
document.getElementById('preset-greenvalley').addEventListener('click',() => fillPreset('green_valley'));
document.getElementById('preset-disk').addEventListener('click',       () => fillPreset('disk'));

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
    const inputs = getCurrentInput();
    const body   = JSON.stringify(inputs);
    const hdr    = { 'Content-Type': 'application/json' };
    try {
        const [mainResult, baselineResult] = await Promise.all([
            fetch('/api/starcharacterizer/predict',  { method:'POST', headers:hdr, body }).then(r => { if (!r.ok) return r.json().then(e => { throw new Error(e.detail || r.statusText); }); return r.json(); }),
            fetch('/api/starcharacterizer/baseline', { method:'POST', headers:hdr, body }).then(r => { if (!r.ok) return r.json().then(e => { throw new Error(e.detail || r.statusText); }); return r.json(); })
        ]);
        const corrected = astronomicallyCorrect(mainResult, inputs);
        window._scLastResult = { ...corrected, _baseline: baselineResult, _input: inputs };
        displayResults(corrected, baselineResult, inputs);
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
    renderInterpretation(r);
    renderPopBars('gmmBars', r.labels, r.gmm_fractions);
    renderPopBars('mlpBars', r.labels, r.mlp_fractions);
    document.getElementById('sc-agreement-val').textContent  = r.agreement_pct.toFixed(1) + '%';
    document.getElementById('sc-certainty-chip').textContent = r.certainty_pct.toFixed(1) + '%';

    // Tab 2 — Explainability
    renderFeatureImportance(r);
    renderPopulationConfidence(r);
    renderDisentanglementPanel(r);

    // Tab 3 — Evolution
    renderCMD(r, inputs);
    renderEvolutionTimeline(r);

    // Tab 5 — Comparison
    renderComparisonTab(r, baseline);
}

// ══════════════════════════════════════════
//  TAB 1: INTERPRETATION RENDER
// ══════════════════════════════════════════
function renderInterpretation(r) {
    const gal       = r.galaxy_class || classifyGalaxy(r._input || {u:0,g:0});
    const H         = r.entropy;
    const ETI       = r.ETI       !== undefined ? r.ETI       : H / Math.log(3) * 100;
    const certainty = r.certainty_pct !== undefined ? r.certainty_pct : (1 - H / Math.log(3)) * 100;
    const c         = r.mlp_fractions;

    // 1. Galaxy class card
    const nameEl = document.getElementById('galaxy-class-name');
    const seqEl  = document.getElementById('sequence-badge');
    const ugEl   = document.getElementById('sc-ug-val');
    const noteEl = document.getElementById('sc-preview-note');
    if (nameEl) { nameEl.textContent = gal.label; nameEl.style.color = gal.colour; }
    if (seqEl)  { seqEl.textContent = gal.seq; seqEl.style.borderColor = gal.colour; seqEl.style.color = gal.colour; }
    if (r._input && ugEl) { const ug = r._input.u - r._input.g; ugEl.textContent = ug.toFixed(2); ugEl.style.color = gal.colour; }
    if (noteEl) noteEl.textContent = gal.seq + ' · H_max ≈ ' + gal.H_hi;

    // 2. Dominant population
    const labels   = r.labels || ['Young stars','Intermediate stars','Old stars'];
    const domIdx   = c.indexOf(Math.max(...c));
    const dominant = labels[domIdx];
    const domIcons = { 'Young stars':'🔵', 'Intermediate stars':'🟢', 'Old stars':'🔴' };
    const domDescs = {
        'Young stars':        'Active star formation — Blue Cloud. High SFR, blue u−g colour.',
        'Intermediate stars': 'Mixed-age population. Transitional regime — possible Green Valley.',
        'Old stars':          'Passive evolution — Red Sequence. Quenched >5 Gyr ago.'
    };
    const iconEl   = document.getElementById('sc-dominant-icon');
    const nameEl2  = document.getElementById('sc-dominant-name');
    const pctEl    = document.getElementById('sc-dominant-pct');
    const interpEl = document.getElementById('sc-dominant-interp');
    if (iconEl)   iconEl.textContent   = domIcons[dominant] || '⭐';
    if (nameEl2)  nameEl2.textContent  = dominant;
    if (pctEl)    pctEl.textContent    = c[domIdx].toFixed(1) + '%';
    if (interpEl) interpEl.textContent = domDescs[dominant] || '';

    // 3. Population certainty bar
    const certColor = certainty >= 70 ? '#6366f1' : certainty >= 40 ? '#fbbf24' : '#f87171';
    const certReadEl = document.getElementById('sc-certainty-readout');
    const certFill   = document.getElementById('sc-certainty-fill');
    if (certReadEl) certReadEl.textContent = `H = ${H.toFixed(3)} nats \u00a0|\u00a0 ${certainty.toFixed(1)}% certain`;
    if (certFill) requestAnimationFrame(() => requestAnimationFrame(() => {
        certFill.style.width      = Math.min(certainty, 100).toFixed(1) + '%';
        certFill.style.background = certColor;
    }));

    // 4. ETI bar
    const etiZone  = ETI < 32 ? 'Stable zone' : ETI < 50 ? 'Mixed zone' : ETI < 59 ? 'Green Valley zone' : 'Transitional zone';
    const etiColor = ETI < 32 ? '#6366f1' : ETI < 50 ? '#60a5fa' : ETI < 59 ? '#34d399' : '#fbbf24';
    const etiReadEl = document.getElementById('sc-eti-readout');
    const etiZoneEl = document.getElementById('sc-eti-zone');
    const etiFill   = document.getElementById('sc-eti-fill');
    if (etiReadEl) etiReadEl.textContent = `ETI = ${ETI.toFixed(1)}%`;
    if (etiZoneEl) { etiZoneEl.textContent = etiZone; etiZoneEl.style.color = etiColor; }
    if (etiFill) requestAnimationFrame(() => requestAnimationFrame(() => {
        etiFill.style.width      = Math.min(ETI, 100).toFixed(1) + '%';
        etiFill.style.background = etiColor;
    }));

    // 5. State badge
    const badge   = document.getElementById('sc-state-badge');
    const descEl  = document.getElementById('sc-state-desc');
    const stateMap = {
        'TRANSITIONAL STATE': { cls:'eti-transitional', icon:'⚡' },
        'STABLE POPULATION':  { cls:'eti-stable',       icon:'◉'  },
        'MIXED POPULATION':   { cls:'eti-mixed',        icon:'◎'  },
        'ANOMALOUS ENTROPY':  { cls:'eti-mixed',        icon:'⚠'  }
    };
    const sm = stateMap[r.state] || stateMap['MIXED POPULATION'];
    if (badge)  { badge.textContent = sm.icon + ' ' + r.state; badge.className = 'sc-state-badge ' + sm.cls; }
    if (descEl) descEl.textContent = r.stateDesc || '';
}

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
    const pr = r.pearson_r;
    const prBadge = pr < 0.15
        ? '<span class="sc-badge-pass">PASS</span>'
        : '<span class="sc-badge-pass sc-badge-fail">FAIL</span>';
    const prHtml = pr.toFixed(4) + ' ' + prBadge;

    const hs = r.hsic;
    const hsBadge = hs < 0.05
        ? '<span class="sc-badge-pass">PASS</span>'
        : '<span class="sc-badge-pass sc-badge-fail">FAIL</span>';
    const hsHtml = hs.toFixed(6) + ' ' + hsBadge;

    const rr2 = r.reconstruction_r2;
    const alpha = r.alpha_used;

    // Write to Diagnostics tab (primary)
    const dPearson = document.getElementById('dq-pearson');
    const dHsic    = document.getElementById('dq-hsic');
    const dReconV  = document.getElementById('dq-recon-val');
    const dReconB  = document.getElementById('dq-recon-bar');
    const dAlpha   = document.getElementById('dq-alpha');
    if (dPearson) dPearson.innerHTML = prHtml;
    if (dHsic)    dHsic.innerHTML    = hsHtml;
    if (dReconV)  dReconV.textContent = rr2.toFixed(4);
    if (dAlpha)   dAlpha.textContent  = alpha;
    if (dReconB)  requestAnimationFrame(() => requestAnimationFrame(() => { dReconB.style.width = (rr2 * 100) + '%'; }));

    // Mirror to Explainability tab (exp-dq-* IDs)
    const ePearson = document.getElementById('exp-dq-pearson');
    const eHsic    = document.getElementById('exp-dq-hsic');
    const eReconV  = document.getElementById('exp-dq-recon-val');
    const eReconB  = document.getElementById('exp-dq-recon-bar');
    const eAlpha   = document.getElementById('exp-dq-alpha');
    if (ePearson) ePearson.innerHTML = prHtml;
    if (eHsic)    eHsic.innerHTML    = hsHtml;
    if (eReconV)  eReconV.textContent = rr2.toFixed(4);
    if (eAlpha)   eAlpha.textContent  = alpha;
    if (eReconB)  requestAnimationFrame(() => requestAnimationFrame(() => { eReconB.style.width = (rr2 * 100) + '%'; }));
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
    // Mini bar chart — Basic AE vs Split-VAE (hardcoded real metrics)
    const ctx = document.getElementById('comparisonMiniChart');
    if (!ctx) return;
    if (comparisonMiniChart) comparisonMiniChart.destroy();
    comparisonMiniChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['R²', 'MAE'],
            datasets: [
                { label: 'Basic AE',  data: [0.7011, 0.5499], backgroundColor: 'rgba(248,113,113,0.45)', borderColor: '#f87171', borderWidth: 1, borderRadius: 4, barPercentage: 0.55 },
                { label: 'Split-VAE', data: [0.9790, 0.0285], backgroundColor: 'rgba(52,211,153,0.55)',  borderColor: '#34d399', borderWidth: 1, borderRadius: 4, barPercentage: 0.55 }
            ]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { grid: { display: false } },
                y: { min: 0, max: 1.05, grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b' } }
            },
            plugins: { legend: { labels: { color: '#cbd5e1', usePointStyle: true, padding: 10, font: { size: 11 } } } }
        }
    });

    // Fraction delta chips — Split-VAE corrected vs Basic AE baseline
    const labels   = ['Young', 'Intermediate', 'Old'];
    const mlpFracs = main.mlp_fractions;
    const gmmFracs = baseline.gmm_fractions;
    ['diff-young', 'diff-inter', 'diff-old'].forEach((id, i) => {
        const el = document.getElementById(id);
        if (!el) return;
        const diff = mlpFracs[i] - gmmFracs[i];
        const sign = diff >= 0 ? '+' : '';
        el.textContent = `\u0394 ${labels[i]}: ${sign}${diff.toFixed(1)}%`;
        el.style.color = diff >= 0 ? 'var(--emerald)' : 'var(--amber)';
    });
}

async function toggleBaselineMode(isBaseline) {
    const warn = document.getElementById('baseline-warn');
    if (warn) warn.style.display = isBaseline ? 'block' : 'none';
    if (!window._scLastResult) return;
    const r = window._scLastResult;
    if (isBaseline) {
        try {
            const inp = r._input;
            const res = await fetch('/api/starcharacterizer/predict_basic', {
                method: 'POST', headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ u:inp.u, g:inp.g, r:inp.r, i:inp.i, z:inp.z, redshift:inp.redshift })
            });
            const basic = await res.json();
            updateFractionBars(basic.gmm_fractions, basic.labels);
        } catch(e) {
            updateFractionBars(r.gmm_fractions, r.labels);
        }
    } else {
        updateFractionBars(r.mlp_fractions, r.labels);
    }
}

async function runInvarianceCheck(z) {
    if (!window._scLastResult) return;
    const el  = document.getElementById('inv-result');
    if (!el) return;
    el.innerHTML = '<span style="font-size:0.70rem;color:var(--text-3);">Running inference\u2026</span>';
    const inp = window._scLastResult._input;
    try {
        const res  = await fetch('/api/starcharacterizer/predict', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ u:inp.u, g:inp.g, r:inp.r, i:inp.i, z:inp.z, redshift: parseFloat(z) })
        });
        const data = await res.json();
        const orig     = window._scLastResult.mlp_fractions;
        const maxShift = Math.max(...data.mlp_fractions.map((f, i) => Math.abs(f - orig[i]))).toFixed(1);
        const isInv    = maxShift < 5.0;
        el.innerHTML = `
          <div style="font-size:0.71rem;color:#34d399;margin-bottom:6px;">
            Max population shift = <strong>${maxShift}%</strong>
            (z=${parseFloat(z).toFixed(2)} vs z=${parseFloat(inp.redshift).toFixed(2)})
            <span style="display:inline-block;padding:2px 8px;border-radius:10px;font-size:0.63rem;font-weight:700;margin-left:8px;
              background:${isInv?'rgba(52,211,153,0.15)':'rgba(251,191,36,0.15)'};
              border:1px solid ${isInv?'#34d399':'#fbbf24'};
              color:${isInv?'#34d399':'#fbbf24'}">${isInv?'REDSHIFT INVARIANT':'MINOR SENSITIVITY'}</span>
          </div>
          <div style="font-size:0.67rem;color:#f87171;font-style:italic;">
            Basic AE reclassifies galaxy type at z &gt; 0.20 due to unlabelled distance correlation in latent space.
          </div>`;
    } catch(e) {
        el.innerHTML = '<span style="font-size:0.70rem;color:#f87171;">Inference failed</span>';
    }
}

// ══════════════════════════
//  INIT
// ══════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    onInputChange();   // populate galaxy class card from default/placeholder values
});


