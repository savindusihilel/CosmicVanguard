/* ═══════════════════════════════════════════
   Galaxy Predictor — App Logic
   ═══════════════════════════════════════════ */

// ── Presets Removed ──

let evolutionChart = null;
let lossChartLoaded = false;
let evolutionChartLoaded = false;

// ── Chart.js Global Theme ──
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;

// ── Preset Loading Logic Removed ──

// ── Reset truth data if user manually edits fields ──
document.getElementById('predictForm').addEventListener('input', (e) => {
    if (e.target.tagName === 'INPUT') {
        window.trueMass = undefined;
        window.trueSfr = undefined;
        document.querySelectorAll('.sdss-galaxy-card').forEach(c => c.classList.remove('selected'));
    }
});

// ── Form submission ──
document.getElementById('predictForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const data = {
        u: parseFloat(document.getElementById('u').value),
        g: parseFloat(document.getElementById('g').value),
        r: parseFloat(document.getElementById('r').value),
        i: parseFloat(document.getElementById('i').value),
        z: parseFloat(document.getElementById('z').value),
        redshift: parseFloat(document.getElementById('redshift').value)
    };

    setLoading(true);

    try {
        const res = await fetch('/api/galaxy/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        if (!res.ok) throw new Error("API request failed");

        const result = await res.json();
        displayResults(result);
    } catch (err) {
        alert("Prediction failed: " + err.message);
    } finally {
        setLoading(false);
    }
});

// ── Loading state ──
function setLoading(isLoading) {
    const loader = document.getElementById('loading');
    if (isLoading) loader.classList.remove('hidden');
    else loader.classList.add('hidden');
}

// ── Interpretation text ──
function generateInterpretation(r) {
    const q = r.quenching_prob_mean;
    const mass = r.mass_log_mean;
    const sfr = r.sfr_log_mean;

    let state, detail;

    if (q < 0.3) {
        state = "actively star-forming";
        detail = "The galaxy is likely part of the blue-cloud population, exhibiting significant ongoing stellar mass assembly. Its photometric colours are consistent with a young stellar population with active star formation.";
    } else if (q > 0.7) {
        state = "quenched";
        detail = "The galaxy appears to have ceased significant star formation and is likely part of the red-sequence population. Its photometric colours indicate an older, evolved stellar population.";
    } else {
        state = "in a transitional evolutionary state";
        detail = "The galaxy occupies the green valley between star-forming and quenched populations. This transitional phase may indicate recent or ongoing quenching mechanisms.";
    }

    return `This galaxy has an estimated stellar mass of ${mass.toFixed(2)} log M☉ ` +
        `and a star formation rate of ${sfr.toFixed(2)} log SFR. ` +
        `The model classifies it as ${state} (Q = ${q.toFixed(2)}).\n\n${detail}\n\n` +
        `The quenching probability reflects the model's inferred evolutionary status ` +
        `based on photometric colours, luminosity, and redshift.`;
}

// ══════════════════════
//  DISPLAY RESULTS
// ══════════════════════
function displayResults(r) {
    document.getElementById('emptyState').classList.add('hidden');
    document.getElementById('resultsContent').classList.remove('hidden');

    // ── KPI Metrics ──
    animateValue('massVal', r.mass_log_mean, 2);
    animateValue('sfrVal', r.sfr_log_mean, 2);
    document.getElementById('massUncertainty').textContent = `± ${r.mass_log_std.toFixed(2)}`;
    document.getElementById('sfrUncertainty').textContent = `± ${r.sfr_log_std.toFixed(2)}`;

    // ── Truth comparison (SDSS galaxies with MPA-JHU ground truth) ──
    const truthBar = document.getElementById('truthBar');
    const truthGrid = document.getElementById('truthGrid');
    const truthUnavail = document.getElementById('truthUnavailable');
    if (window.trueMass !== undefined && window.trueSfr !== undefined) {
        truthBar.classList.remove('hidden');
        if (window.trueMass !== null && window.trueSfr !== null) {
            // Ground truth available — show full comparison
            if (truthGrid) truthGrid.classList.remove('hidden');
            if (truthUnavail) truthUnavail.classList.add('hidden');

            const massErr = Math.abs(window.trueMass - r.mass_log_mean);
            const sfrErr = Math.abs(window.trueSfr - r.sfr_log_mean);

            document.getElementById('trueMassVal').textContent = window.trueMass.toFixed(2);
            document.getElementById('predMassVal').textContent = r.mass_log_mean.toFixed(2);
            document.getElementById('trueSfrVal').textContent = window.trueSfr.toFixed(2);
            document.getElementById('predSfrVal').textContent = r.sfr_log_mean.toFixed(2);

            const massErrEl = document.getElementById('massErrVal');
            massErrEl.textContent = `Δ${massErr.toFixed(3)}`;
            massErrEl.className = 'truth-metric-err ' + (massErr < 0.1 ? 'good' : massErr < 0.3 ? 'warn' : 'bad');

            const sfrErrEl = document.getElementById('sfrErrVal');
            sfrErrEl.textContent = `Δ${sfrErr.toFixed(3)}`;
            sfrErrEl.className = 'truth-metric-err ' + (sfrErr < 0.15 ? 'good' : sfrErr < 0.4 ? 'warn' : 'bad');
        } else {
            // No spectroscopic reference — hide stale grid, show message only
            if (truthGrid) truthGrid.classList.add('hidden');
            if (truthUnavail) truthUnavail.classList.remove('hidden');
        }
    } else {
        truthBar.classList.add('hidden');
        if (truthGrid) truthGrid.classList.remove('hidden');
        if (truthUnavail) truthUnavail.classList.add('hidden');
    }

    // ── Quenching ──
    const q = r.quenching_prob_mean;
    const q_std = r.quenching_prob_std;
    document.getElementById('probText').textContent = `Q = ${q.toFixed(2)} ± ${q_std.toFixed(2)}`;
    document.getElementById('probBar').style.width = '100%';
    document.getElementById('probMarker').style.left = `${Math.min(Math.max(q * 100, 0), 100)}%`;

    const badge = document.getElementById('quenchingStatus');
    badge.className = 'badge';
    if (q < 0.3) {
        badge.textContent = 'Star Forming';
        badge.classList.add('blue');
    } else if (q > 0.7) {
        badge.textContent = 'Quenched';
        badge.classList.add('red');
    } else {
        badge.textContent = 'Transitional';
        badge.classList.add('yellow');
    }

    // Histogram
    renderHistogram(r.quenching_posterior);

    // ── Interpretation tab ──
    document.getElementById('interpretationText').textContent = generateInterpretation(r);

    // ── Explainability tab ──
    if (r.mass_feature_importance) renderFeatureImportance('massImportance', r.mass_feature_importance);
    if (r.sfr_feature_importance) renderFeatureImportance('sfrImportance', r.sfr_feature_importance);

    // ── Comparison tab ──
    if (r.rf_mass_log_mean !== null) {
        document.getElementById('pinnCompMass').textContent = `${r.mass_log_mean.toFixed(2)} ± ${r.mass_log_std.toFixed(2)}`;
        document.getElementById('pinnCompSfr').textContent = `${r.sfr_log_mean.toFixed(2)} ± ${r.sfr_log_std.toFixed(2)}`;
        document.getElementById('rfCompMass').textContent = `${r.rf_mass_log_mean.toFixed(2)} ± ${r.rf_mass_log_std.toFixed(2)}`;
        document.getElementById('rfCompSfr').textContent = `${r.rf_sfr_log_mean.toFixed(2)} ± ${r.rf_sfr_log_std.toFixed(2)}`;
    }

    // ── Evolution chart (lazy) ──
    evolutionChartLoaded = false; // re-render on next tab visit
    window._pendingEvolution = { mass: r.mass_log_mean, sfr: r.sfr_log_mean };

    // If the evolution tab is currently active, render immediately
    if (document.getElementById('panel-evolution').classList.contains('active')) {
        renderEvolutionChart(r.mass_log_mean, r.sfr_log_mean);
        evolutionChartLoaded = true;
    }
}

// ── Animated numeric value ──
function animateValue(id, target, decimals) {
    const el = document.getElementById(id);
    const duration = 600;
    const start = performance.now();
    const from = parseFloat(el.textContent) || 0;

    function step(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        const current = from + (target - from) * eased;
        el.textContent = current.toFixed(decimals);
        if (progress < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
}

// ── Histogram ──
function renderHistogram(samples) {
    const container = document.getElementById('histogram');
    container.innerHTML = '';

    const bins = 30;
    const bucketCounts = new Array(bins).fill(0);
    samples.forEach(v => {
        const idx = Math.floor(v * bins);
        if (idx >= 0 && idx < bins) bucketCounts[idx]++;
    });
    const maxCount = Math.max(...bucketCounts);

    bucketCounts.forEach((count, i) => {
        const bar = document.createElement('div');
        bar.className = 'hist-bar';
        const h = maxCount > 0 ? (count / maxCount) * 100 : 0;
        // Stagger the animation
        bar.style.height = '0%';
        bar.style.transitionDelay = `${i * 15}ms`;
        container.appendChild(bar);
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                bar.style.height = `${h}%`;
            });
        });
    });
}

// ── Feature Importance ──
function renderFeatureImportance(containerId, importance) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    const sorted = Object.entries(importance)
        .filter(([k]) => k !== 'Mr_placeholder')
        .sort((a, b) => b[1] - a[1])
        .slice(0, 6);

    sorted.forEach(([feature, value], i) => {
        const percent = (value * 100).toFixed(1);
        const row = document.createElement('div');
        row.className = 'feature-bar';
        row.innerHTML = `
            <div class="feature-name">${feature}</div>
            <div class="feature-bar-bg">
                <div class="feature-bar-fill" style="width:0%"></div>
            </div>
            <div class="feature-value">${percent}%</div>
        `;
        container.appendChild(row);

        // Animate bar fill
        const fill = row.querySelector('.feature-bar-fill');
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                fill.style.width = `${percent}%`;
                fill.style.transitionDelay = `${i * 60}ms`;
            });
        });
    });
}

// ══════════════════════
//  TAB SYSTEM
// ══════════════════════
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanels = document.querySelectorAll('.tab-panel');
const tabIndicator = document.getElementById('tabIndicator');

function switchTab(tabName) {
    tabBtns.forEach(btn => {
        const isActive = btn.dataset.tab === tabName;
        btn.classList.toggle('active', isActive);
        btn.setAttribute('aria-selected', isActive);
    });

    tabPanels.forEach(panel => {
        panel.classList.toggle('active', panel.id === `panel-${tabName}`);
    });

    updateTabIndicator();

    // Lazy-load charts — delay so canvas has real dimensions after display:block
    if (tabName === 'evolution' && !evolutionChartLoaded && window._pendingEvolution) {
        setTimeout(() => {
            renderEvolutionChart(window._pendingEvolution.mass, window._pendingEvolution.sfr);
            evolutionChartLoaded = true;
        }, 50);
    }
    if (tabName === 'diagnostics' && !lossChartLoaded) {
        setTimeout(() => {
            loadTrainingLoss();
            lossChartLoaded = true;
        }, 50);
    }
}

function updateTabIndicator() {
    const activeBtn = document.querySelector('.tab-btn.active');
    if (!activeBtn || !tabIndicator) return;
    tabIndicator.style.left = `${activeBtn.offsetLeft}px`;
    tabIndicator.style.width = `${activeBtn.offsetWidth}px`;
}

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

// Re-position indicator on resize
window.addEventListener('resize', updateTabIndicator);

// ══════════════════════
//  CHARTS
// ══════════════════════
async function renderEvolutionChart(mass, sfr) {
    const ctx = document.getElementById('evolutionChart');
    const loadingEl = document.getElementById('evolutionLoading');
    
    if (loadingEl) loadingEl.classList.remove('hidden');

    // Fetch real SDSS main sequence data
    let mainSequence = [];
    try {
        const res = await fetch('/api/galaxy/main-sequence');
        const data = await res.json();
        if (data.mass && data.mass.length > 0) {
            for (let i = 0; i < data.mass.length; i++) {
                mainSequence.push({ x: data.mass[i], y: data.sfr[i] });
            }
        }
    } catch (e) {
        console.warn('Main sequence data unavailable:', e.message);
    }

    // Fallback to synthetic if fetch fails
    if (mainSequence.length === 0) {
        for (let m = 8; m <= 12; m += 0.2) {
            const base = 0.7 * m - 7;
            for (let i = 0; i < 5; i++) {
                mainSequence.push({
                    x: m + (Math.random() - 0.5) * 0.2,
                    y: base + (Math.random() - 0.5) * 0.4
                });
            }
        }
    }

    if (loadingEl) loadingEl.classList.add('hidden');
    if (evolutionChart) evolutionChart.destroy();

    evolutionChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'SDSS Galaxies (n=' + mainSequence.length.toLocaleString() + ')',
                    data: mainSequence,
                    backgroundColor: 'rgba(100,116,139,0.18)',
                    borderColor: 'rgba(100,116,139,0.08)',
                    pointRadius: 1.8,
                    pointHoverRadius: 3,
                    order: 2
                },
                {
                    label: 'Predicted Galaxy',
                    data: [{ x: mass, y: sfr }],
                    backgroundColor: '#22d3ee',
                    borderColor: '#67e8f9',
                    borderWidth: 2.5,
                    pointRadius: 10,
                    pointHoverRadius: 13,
                    pointStyle: 'star',
                    order: 1
                }
            ]
        },
        options: {
            responsive: true,
            animation: { duration: 800, easing: 'easeOutQuart' },
            scales: {
                x: {
                    title: { display: true, text: 'log(M★/M☉)', color: '#94a3b8', font: { weight: 600, size: 12 } },
                    min: 6, max: 13,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#64748b' }
                },
                y: {
                    title: { display: true, text: 'log(SFR / M☉ yr⁻¹)', color: '#94a3b8', font: { weight: 600, size: 12 } },
                    min: -3.5, max: 2.5,
                    grid: { color: 'rgba(255,255,255,0.04)' },
                    ticks: { color: '#64748b' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#cbd5e1', usePointStyle: true, pointStyle: 'circle', padding: 16, font: { size: 11 } }
                },
                tooltip: {
                    backgroundColor: 'rgba(15,23,42,0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255,255,255,0.08)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 10,
                    callbacks: {
                        label: c => {
                            const p = c.raw;
                            return `${c.dataset.label}: M★=${p.x.toFixed(2)}, SFR=${p.y.toFixed(2)}`;
                        }
                    }
                }
            }
        }
    });
}

async function loadTrainingLoss() {
    try {
        const res = await fetch('/api/galaxy/training-loss');
        const data = await res.json();

        // Determine which stages actually have data
        const hasA = data.stageA_loss && data.stageA_loss.length > 0;
        const hasB = data.stageB_loss && data.stageB_loss.length > 0;
        const hasC = data.stageC_loss && data.stageC_loss.length > 0;

        // Compute the total timeline length (max of present stages)
        const epochsA = hasA ? data.epochsA.length : 0;
        const epochsB = hasB ? data.epochsB.length : 0;
        const epochsC = hasC ? data.epochsC.length : 0;
        const totalEpochs = Math.max(epochsA, epochsB + (hasA ? data.epochsA.length : 0), epochsC + (hasA ? data.epochsA.length : 0) + (hasB ? data.epochsB.length : 0));
        // Actually the original logic used separate epochs per stage; better to keep them as separate spans.
        // We'll create labels that go from 1 to total epochs across all stages (just like before).
        // But now we only sum epochs of stages that exist.
        const allEpochs = [];
        if (hasA) allEpochs.push(...data.epochsA);
        if (hasB) allEpochs.push(...data.epochsB);
        if (hasC) allEpochs.push(...data.epochsC);
        const totalLength = allEpochs.length; // number of x-axis labels

        const datasets = [];

        // Stage A — Supervised
        if (hasA) {
            const stageA_loss = data.stageA_loss;
            // Pad with nulls for subsequent stages
            const values = [...stageA_loss, ...Array(totalLength - stageA_loss.length).fill(null)];
            datasets.push({
                label: 'Stage A — Supervised',
                data: values,
                borderColor: '#22c55e',
                backgroundColor: 'rgba(34,197,94,0.08)',
                fill: true,
                borderWidth: 2,
                tension: 0.35,
                pointRadius: 0
            });
        }

        // Stage B — Flow NLL
        if (hasB) {
            const stageB_loss = data.stageB_loss;
            const startIdx = hasA ? data.epochsA.length : 0;
            const values = [
                ...Array(startIdx).fill(null),
                ...stageB_loss,
                ...Array(totalLength - startIdx - stageB_loss.length).fill(null)
            ];
            datasets.push({
                label: 'Stage B — Flow NLL',
                data: values,
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245,158,11,0.08)',
                fill: true,
                borderWidth: 2,
                tension: 0.35,
                pointRadius: 0
            });
        }

        // Stage C — Physics Joint (only if data exists)
        if (hasC) {
            const stageC_loss = data.stageC_loss;
            const startIdx = (hasA ? data.epochsA.length : 0) + (hasB ? data.epochsB.length : 0);
            const values = [
                ...Array(startIdx).fill(null),
                ...stageC_loss,
                ...Array(totalLength - startIdx - stageC_loss.length).fill(null)
            ];
            datasets.push({
                label: 'Stage C — Physics Joint',
                data: values,
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59,130,246,0.08)',
                fill: true,
                borderWidth: 2,
                tension: 0.35,
                pointRadius: 0
            });
        }

        if (datasets.length === 0) {
            console.warn('No training loss data available.');
            return;
        }

        const ctx = document.getElementById('lossChart').getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array.from({ length: totalLength }, (_, i) => i + 1),
                datasets: datasets
            },
            options: {
                responsive: true,
                animation: { duration: 800, easing: 'easeOutQuart' },
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: {
                        title: { display: true, text: 'Epoch', color: '#94a3b8', font: { weight: 500 } },
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { color: '#64748b', maxTicksLimit: 10 }
                    },
                    y: {
                        title: { display: true, text: 'Loss', color: '#94a3b8', font: { weight: 500 } },
                        grid: { color: 'rgba(255,255,255,0.04)' },
                        ticks: { color: '#64748b' }
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#cbd5e1', usePointStyle: true, pointStyle: 'circle', padding: 16 }
                    },
                    tooltip: {
                        backgroundColor: '#151d2e',
                        titleColor: '#f1f5f9',
                        bodyColor: '#cbd5e1',
                        borderColor: 'rgba(255,255,255,0.08)',
                        borderWidth: 1,
                        cornerRadius: 8,
                        padding: 10
                    }
                }
            }
        });
    } catch (err) {
        console.warn('Training loss data unavailable:', err.message);
    }
}

// ══════════════════════
//  RF BENCHMARK METRICS
// ══════════════════════
let rfComparisonChart = null;

async function loadRFMetrics() {
    try {
        const res = await fetch('/api/galaxy/rf-metrics');
        const data = await res.json();
        if (data.error) return;

        const rf = data.rf;
        const pinn = data.pinn;

        // Helper: set value and comparison arrow
        function setCmp(pinnId, rfId, arrowId, pinnVal, rfVal, lowerIsBetter) {
            document.getElementById(pinnId).textContent = pinnVal.toFixed(3);
            document.getElementById(rfId).textContent = rfVal.toFixed(3);
            const arrowEl = document.getElementById(arrowId);
            if (lowerIsBetter) {
                const pinnWins = pinnVal < rfVal;
                arrowEl.textContent = pinnWins ? '↓' : '↑';
                arrowEl.className = 'cmp-arrow ' + (pinnWins ? 'win' : 'lose');
            } else {
                const pinnWins = pinnVal > rfVal;
                arrowEl.textContent = pinnWins ? '↑' : '↓';
                arrowEl.className = 'cmp-arrow ' + (pinnWins ? 'win' : 'lose');
            }
        }

        // Populate comparison cards
        if (rf && pinn) {
            const pm = pinn.mass_metrics, ps = pinn.sfr_metrics;
            const rm = rf.mass_metrics, rs = rf.sfr_metrics;

            setCmp('pinnMassRmse', 'rfMassRmse', 'arrowMassRmse', pm.rmse, rm.rmse, true);
            setCmp('pinnMassMae', 'rfMassMae', 'arrowMassMae', pm.mae, rm.mae, true);
            setCmp('pinnMassR2', 'rfMassR2', 'arrowMassR2', pm.r2, rm.r2, false);

            setCmp('pinnSfrRmse', 'rfSfrRmse', 'arrowSfrRmse', ps.rmse, rs.rmse, true);
            setCmp('pinnSfrMae', 'rfSfrMae', 'arrowSfrMae', ps.mae, rs.mae, true);
            setCmp('pinnSfrR2', 'rfSfrR2', 'arrowSfrR2', ps.r2, rs.r2, false);
        }

        // Dual comparison charts (RMSE + R²)
        if (!rf || !pinn) return;

        if (rfComparisonChart) rfComparisonChart.destroy();
        if (window._r2Chart) window._r2Chart.destroy();

        const pm = pinn.mass_metrics, ps = pinn.sfr_metrics;
        const rm = rf.mass_metrics, rs = rf.sfr_metrics;

        const sharedOpts = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 900, easing: 'easeOutQuart' },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#94a3b8', font: { size: 11, weight: 500 } } },
                y: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748b', font: { size: 10 } } }
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#94a3b8', usePointStyle: true, pointStyle: 'rectRounded', boxWidth: 10, font: { size: 10 }, padding: 12 }
                },
                tooltip: {
                    backgroundColor: 'rgba(15,23,42,0.95)',
                    titleColor: '#e2e8f0',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255,255,255,0.08)',
                    borderWidth: 1,
                    cornerRadius: 8,
                    padding: 10,
                    callbacks: { label: c => `${c.dataset.label}: ${c.raw.toFixed(4)}` }
                }
            }
        };

        // ── RMSE chart ──
        const rmseCtx = document.getElementById('rmseCompChart');
        if (rmseCtx) {
            const ctx2d = rmseCtx.getContext('2d');
            const pinnGrad = ctx2d.createLinearGradient(0, 0, 0, 200);
            pinnGrad.addColorStop(0, 'rgba(59,130,246,0.6)');
            pinnGrad.addColorStop(1, 'rgba(59,130,246,0.15)');
            const rfGrad = ctx2d.createLinearGradient(0, 0, 0, 200);
            rfGrad.addColorStop(0, 'rgba(148,163,184,0.5)');
            rfGrad.addColorStop(1, 'rgba(148,163,184,0.1)');

            rfComparisonChart = new Chart(ctx2d, {
                type: 'bar',
                data: {
                    labels: ['Stellar Mass', 'Star Formation Rate'],
                    datasets: [
                        { label: 'PINN', data: [pm.rmse, ps.rmse], backgroundColor: pinnGrad, borderColor: '#3b82f6', borderWidth: 1, borderRadius: 6, barPercentage: 0.6, categoryPercentage: 0.65 },
                        { label: 'Random Forest', data: [rm.rmse, rs.rmse], backgroundColor: rfGrad, borderColor: '#64748b', borderWidth: 1, borderRadius: 6, barPercentage: 0.6, categoryPercentage: 0.65 }
                    ]
                },
                options: {
                    ...sharedOpts,
                    scales: {
                        ...sharedOpts.scales,
                        y: { ...sharedOpts.scales.y, beginAtZero: true, max: Math.max(pm.rmse, ps.rmse, rm.rmse, rs.rmse) * 1.35 }
                    }
                }
            });
        }

        // ── R² chart ──
        const r2Ctx = document.getElementById('r2CompChart');
        if (r2Ctx) {
            const ctx2d = r2Ctx.getContext('2d');
            const pinnGrad = ctx2d.createLinearGradient(0, 0, 0, 200);
            pinnGrad.addColorStop(0, 'rgba(34,197,94,0.6)');
            pinnGrad.addColorStop(1, 'rgba(34,197,94,0.15)');
            const rfGrad = ctx2d.createLinearGradient(0, 0, 0, 200);
            rfGrad.addColorStop(0, 'rgba(148,163,184,0.5)');
            rfGrad.addColorStop(1, 'rgba(148,163,184,0.1)');

            window._r2Chart = new Chart(ctx2d, {
                type: 'bar',
                data: {
                    labels: ['Stellar Mass', 'Star Formation Rate'],
                    datasets: [
                        { label: 'PINN', data: [pm.r2, ps.r2], backgroundColor: pinnGrad, borderColor: '#22c55e', borderWidth: 1, borderRadius: 6, barPercentage: 0.6, categoryPercentage: 0.65 },
                        { label: 'Random Forest', data: [rm.r2, rs.r2], backgroundColor: rfGrad, borderColor: '#64748b', borderWidth: 1, borderRadius: 6, barPercentage: 0.6, categoryPercentage: 0.65 }
                    ]
                },
                options: {
                    ...sharedOpts,
                    scales: {
                        ...sharedOpts.scales,
                        y: { ...sharedOpts.scales.y, min: Math.min(pm.r2, ps.r2, rm.r2, rs.r2) * 0.92, max: 1.0 }
                    }
                }
            });
        }
    } catch (err) {
        console.warn('RF metrics unavailable:', err.message);
    }
}

// ── Init ──

// Position tab indicator once DOM is settled
requestAnimationFrame(() => {
    requestAnimationFrame(() => {
        updateTabIndicator();
    });
});

// ══════════════════════
//  SDSS LIVE EXPLORER
// ══════════════════════

// ── Tab switching ──
document.querySelectorAll('.sdss-tab').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.sdss-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelectorAll('.sdss-panel').forEach(p => p.classList.remove('active'));
        const panel = document.getElementById('sdss' + btn.dataset.sdssTab.charAt(0).toUpperCase() + btn.dataset.sdssTab.slice(1));
        if (panel) panel.classList.add('active');
    });
});

// ── SDSS loading / error helpers ──
function sdssSetLoading(show) {
    const el = document.getElementById('sdssLoading');
    if (show) el.classList.remove('hidden');
    else el.classList.add('hidden');
    document.querySelectorAll('.sdss-btn').forEach(b => b.disabled = show);
}

function sdssShowError(msg) {
    const el = document.getElementById('sdssError');
    document.getElementById('sdssErrorText').textContent = msg;
    el.classList.remove('hidden');
    setTimeout(() => el.classList.add('hidden'), 6000);
}

// ── Render galaxy cards ──
function sdssRenderResults(galaxies) {
    const wrap = document.getElementById('sdssResultsWrap');
    const list = document.getElementById('sdssResultsList');
    const countEl = document.getElementById('sdssResultCount');

    list.innerHTML = '';
    if (!galaxies || galaxies.length === 0) {
        wrap.style.display = 'none';
        return;
    }
    wrap.style.display = '';
    countEl.textContent = `${galaxies.length} galax${galaxies.length === 1 ? 'y' : 'ies'}`;

    galaxies.forEach((g, idx) => {
        const card = document.createElement('div');
        card.className = 'sdss-galaxy-card';
        const cutoutUrl = `https://skyserver.sdss.org/dr18/SkyserverWS/ImgCutout/getjpeg?ra=${g.ra}&dec=${g.dec}&width=80&height=80&scale=0.2`;
        const massText = g.true_mass != null ? `logM=${g.true_mass.toFixed(2)}` : 'logM=—';
        const sfrText = g.true_sfr != null ? `SFR=${g.true_sfr.toFixed(2)}` : 'SFR=—';
        card.innerHTML = `
            <img class="sdss-card-thumb" src="${cutoutUrl}" alt="Galaxy cutout" loading="lazy" onerror="this.style.display='none'">
            <div class="sdss-card-info">
                <div class="sdss-card-id">${g.objID}</div>
                <div class="sdss-card-meta">z=${g.redshift.toFixed(4)} · ${massText} · ${sfrText}</div>
            </div>
            <button type="button" class="sdss-card-select" title="Select this galaxy">▶</button>
        `;
        card.querySelector('.sdss-card-select').addEventListener('click', () => sdssSelectGalaxy(g));
        card.addEventListener('dblclick', () => sdssSelectGalaxy(g));
        list.appendChild(card);
    });
}

// ── Select a galaxy → populate form ──
function sdssSelectGalaxy(g) {
    document.getElementById('u').value = g.u.toFixed(4);
    document.getElementById('g').value = g.g.toFixed(4);
    document.getElementById('r').value = g.r.toFixed(4);
    document.getElementById('i').value = g.i.toFixed(4);
    document.getElementById('z').value = g.z_mag.toFixed(4);
    document.getElementById('redshift').value = g.redshift.toFixed(6);

    // Store ground truth (may be null)
    window.trueMass = g.true_mass;
    window.trueSfr = g.true_sfr;
    window.sdssSource = g.source || 'SDSS DR18';

    // Highlight selected card
    document.querySelectorAll('.sdss-galaxy-card').forEach(c => c.classList.remove('selected'));
    event && event.target && event.target.closest('.sdss-galaxy-card')?.classList.add('selected');
}

// ── Search ──
async function sdssDoSearch() {
    const query = document.getElementById('sdssQuery').value.trim();
    if (!query) return;
    sdssSetLoading(true);
    try {
        const res = await fetch(`/api/galaxy/sdss/search?query=${encodeURIComponent(query)}`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        sdssRenderResults(data.galaxy ? [data.galaxy] : []);
    } catch (err) {
        sdssShowError(err.message);
    } finally {
        sdssSetLoading(false);
    }
}

// ── Random ──
async function sdssLoadRandom() {
    const n = document.getElementById('sdssRandomN').value;
    const zMin = document.getElementById('sdssZMin').value;
    const zMax = document.getElementById('sdssZMax').value;
    sdssSetLoading(true);
    try {
        const res = await fetch(`/api/galaxy/sdss/random?n=${n}&z_min=${zMin}&z_max=${zMax}`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        sdssRenderResults(data.galaxies || []);
    } catch (err) {
        sdssShowError(err.message);
    } finally {
        sdssSetLoading(false);
    }
}

// ── Region ──
async function sdssSearchRegion() {
    const ra = document.getElementById('sdssRA').value;
    const dec = document.getElementById('sdssDec').value;
    const radius = document.getElementById('sdssRadius').value;
    if (!ra || !dec) { sdssShowError('Enter RA and Dec coordinates.'); return; }
    sdssSetLoading(true);
    try {
        const res = await fetch(`/api/galaxy/sdss/region?ra=${ra}&dec=${dec}&radius=${radius}`);
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }
        const data = await res.json();
        sdssRenderResults(data.galaxies || []);
    } catch (err) {
        sdssShowError(err.message);
    } finally {
        sdssSetLoading(false);
    }
}

// Enter key triggers search
document.getElementById('sdssQuery')?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); sdssDoSearch(); }
});

// ── Init ──
loadRFMetrics();