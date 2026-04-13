/* ============================================================
   EdgeDistillDet Web UI - Charts & Metrics Visualization (Enhanced)
   ============================================================ */

let currentChartType = 'loss';
const charts = {};
window.charts = charts;
let autoRefreshIntervalId = null;
let refreshCountdownIntervalId = null;
let refreshSecondsLeft = 30;
let isDarkMode = false;

// ==================== Chart Color Themes ====================
function getChartColors(dark = false) {
    return dark ? {
        primary: '#D0BCFF',
        secondary: '#CCC2DC',
        tertiary: '#EFB8C8',
        success: '#81C784',
        warning: '#FFB74D',
        error: '#F2B8B5',
        info: '#90CAF9',
        gridColor: 'rgba(202, 196, 208, 0.08)',
        tickColor: '#CAC4D0',
        tooltipBg: '#2B2B36',
        tooltipTitle: '#E6E0E9',
        tooltipBody: '#E6E0E9',
        titleColor: '#E6E0E9',
        zeroLineColor: 'rgba(202, 196, 208, 0.15)',
    } : {
        primary: '#6750A4',
        secondary: '#625B71',
        tertiary: '#7D5260',
        success: '#2E7D32',
        warning: '#F57C00',
        error: '#B3261E',
        info: '#1565C0',
        gridColor: 'rgba(121, 116, 126, 0.08)',
        tickColor: '#49454F',
        tooltipBg: '#313133',
        tooltipTitle: '#FFFFFF',
        tooltipBody: '#E0E0E0',
        titleColor: '#49454F',
        zeroLineColor: 'rgba(121, 116, 126, 0.25)',
    };
}

let ChartColors = getChartColors(false);

// Common options builder
function buildOptions(isDark) {
    const c = getChartColors(isDark);
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    usePointStyle: true, pointStyle: 'circle', padding: 16,
                    font: { size: 12, family: "'Roboto', sans-serif" },
                    color: c.tickColor
                }
            },
            tooltip: {
                backgroundColor: c.tooltipBg,
                titleFont: { size: 13, weight: '500' },
                bodyFont: { size: 12 },
                padding: 12, cornerRadius: 8,
                borderColor: isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.1)', borderWidth: 1,
                titleColor: c.tooltipTitle, bodyColor: c.tooltipBody,
            }
        },
        scales: {
            x: {
                grid: { color: c.gridColor, drawBorder: false },
                ticks: { color: c.tickColor, font: { size: 11 }, maxTicksLimit: 15 }
            },
            y: {
                grid: { color: c.gridColor, drawBorder: false },
                ticks: { color: c.tickColor, font: { size: 11 } }
            }
        },
        elements: {
            line: { tension: 0.35, borderWidth: 2.5 },
            point: { radius: 0, hoverRadius: 5, hitRadius: 8 }
        },
        animation: { duration: 320, easing: 'easeOutCubic' }
    };
}

// Global theme update for all charts
function updateChartsTheme(dark) {
    isDarkMode = dark;
    ChartColors = getChartColors(dark);
    
    Object.values(charts).forEach(chart => {
        if (!chart || !chart.options) return;
        
        // Update colors
        if (chart.options.scales?.x?.grid) chart.options.scales.x.grid.color = ChartColors.gridColor;
        if (chart.options.scales?.x?.ticks) chart.options.scales.x.ticks.color = ChartColors.tickColor;
        if (chart.options.scales?.y?.grid) chart.options.scales.y.grid.color = ChartColors.gridColor;
        if (chart.options.scales?.y?.ticks) chart.options.scales.y.ticks.color = ChartColors.tickColor;
        
        // Update all y-axis sub-scales
        if (chart.options.scales) {
            Object.values(chart.options.scales).forEach(scale => {
                if (scale.ticks) scale.ticks.color = ChartColors.tickColor;
                if (scale.title) scale.title.color = ChartColors.titleColor;
            });
        }
        
        if (chart.options.plugins?.legend?.labels)
            chart.options.plugins.legend.labels.color = ChartColors.tickColor;
        if (chart.options.plugins?.tooltip) {
            chart.options.plugins.tooltip.backgroundColor = ChartColors.tooltipBg;
            chart.options.plugins.tooltip.titleColor = ChartColors.tooltipTitle;
            chart.options.plugins.tooltip.bodyColor = ChartColors.tooltipBody;
        }
        
        chart.update('none');
    });
}

// ==================== Initialize All Charts ====================
function initCharts() {
    const opts = buildOptions(isDarkMode);
    
    // Loss Chart
    charts.lossChart = new Chart(document.getElementById('loss-chart'), {
        type: 'line',
        data: {
            labels: [], datasets: [
                { label: 'Box Loss', data: [], borderColor: ChartColors.primary, backgroundColor: transparentize(ChartColors.primary, 0.12), fill: true },
                { label: 'CLS Loss', data: [], borderColor: ChartColors.warning, backgroundColor: transparentize(ChartColors.warning, 0.12), fill: true },
                { label: 'DFL Loss', data: [], borderColor: ChartColors.info, backgroundColor: transparentize(ChartColors.info, 0.12), fill: true }
            ]
        },
        options: { ...opts, scales: { ...opts.scales, y: { ...opts.scales.y, type: 'logarithmic', title: { display: true, text: 'Loss (log scale)', color: ChartColors.titleColor } } } }
    });

    // mAP Chart
    charts.mapChart = new Chart(document.getElementById('map-chart'), {
        type: 'line',
        data: {
            labels: [], datasets: [
                { label: 'mAP@50', data: [], borderColor: ChartColors.success, backgroundColor: transparentize(ChartColors.success, 0.12), fill: true },
                { label: 'mAP@50-95', data: [], borderColor: ChartColors.tertiary, backgroundColor: transparentize(ChartColors.tertiary, 0.12), fill: true }
            ]
        }, options: opts
    });

    // Learning Rate Chart
    charts.lrChart = new Chart(document.getElementById('lr-chart'), {
        type: 'line',
        data: {
            labels: [], datasets: [
                { label: 'LR pg0', data: [], borderColor: ChartColors.primary, backgroundColor: transparentize(ChartColors.primary, 0.12), fill: true },
                { label: 'LR pg1', data: [], borderColor: ChartColors.secondary, backgroundColor: transparentize(ChartColors.secondary, 0.12), fill: true },
                { label: 'LR pg2', data: [], borderColor: ChartColors.tertiary, backgroundColor: transparentize(ChartColors.tertiary, 0.12), fill: true }
            ]
        }, options: { ...opts, scales: { ...opts.scales, y: { ...opts.scales.y, type: 'logarithmic', title: { display: true, text: 'Learning Rate (log)', color: ChartColors.titleColor } } } }
    });

    // Distillation Dynamics Chart
    charts.distillChart = new Chart(document.getElementById('distill-chart'), {
        type: 'line',
        data: {
            labels: [], datasets: [
                { label: 'Alpha', data: [], yAxisID: 'y_alpha', borderColor: ChartColors.primary, backgroundColor: transparentize(ChartColors.primary, 0.12), fill: true },
                { label: 'Temperature T', data: [], yAxisID: 'y_temp', borderColor: ChartColors.error, backgroundColor: transparentize(ChartColors.error, 0.12), fill: true },
                { label: 'KD Loss', data: [], yAxisID: 'y_loss', borderColor: ChartColors.warning, backgroundColor: transparentize(ChartColors.warning, 0.12), fill: true }
            ]
        }, options: { ...opts, scales: {
            x: opts.scales.x,
            y_alpha: { position: 'left', min: 0, max: 1, title: { display: true, text: 'Alpha', color: ChartColors.primary }, grid: { drawOnChartArea: false }, ticks: { color: ChartColors.tickColor } },
            y_temp: { position: 'right', min: 0, max: 8, title: { display: true, text: 'Temperature', color: ChartColors.error }, grid: { color: transparentize(ChartColors.error, 0.06) }, ticks: { color: ChartColors.tickColor } },
            y_loss: { display: false, min: 0 }
        }}
    });

    // NEW: Precision-Recall Curve
    try {
        charts.prChart = new Chart(document.getElementById('pr-chart'), {
            type: 'line',
            data: {
                labels: [], datasets: [{
                    label: 'Precision-Recall', data: [],
                    borderColor: ChartColors.success, backgroundColor: transparentize(ChartColors.success, 0.08),
                    fill: true, tension: 0.1, pointRadius: 0, borderWidth: 2
                }]
            }, options: { ...opts, scales: { ...opts.scales, x: { ...opts.scales.x, title: { display: true, text: 'Recall', color: ChartColors.titleColor } }, y: { ...opts.scales.y, min: 0, max: 1.05, title: { display: true, text: 'Precision', color: ChartColors.titleColor } } } }
        });
    } catch(e) {}

    // NEW: Class Performance Bar Chart
    try {
        charts.classChart = new Chart(document.getElementById('class-chart'), {
            type: 'bar',
            data: {
                labels: ['drone', 'person', 'vehicle', 'boat', 'bird'],
                datasets: [
                    { label: 'mAP', data: [0.94, 0.89, 0.91, 0.87, 0.82], backgroundColor: transparentize(ChartColors.primary, 0.75), borderColor: ChartColors.primary, borderRadius: 6, borderSkipped: false },
                    { label: 'Recall', data: [0.92, 0.86, 0.89, 0.84, 0.78], backgroundColor: transparentize(ChartColors.success, 0.65), borderColor: ChartColors.success, borderRadius: 6, borderSkipped: false }
                ]
            }, options: {
                ...opts, scales: { x: { ...opts.scales.x, grid: { display: false } }, y: { ...opts.scales.y, beginAtZero: true, max: 1, title: { display: true, text: 'Score', color: ChartColors.titleColor } } }
            }
        });
    } catch(e) {}

    loadSampleData();
    updateOverviewStats();
}

// Helper to create rgba from hex
function transparentize(hex, alpha) {
    if (hex.startsWith('rgba')) return hex;
    const r = parseInt(hex.slice(1,3), 16), g = parseInt(hex.slice(3,5), 16), b = parseInt(hex.slice(5,7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ==================== Sample Data ====================
function loadSampleData() {
    const epochs = Array.from({ length: 100 }, (_, i) => i + 1);
    const genDecay = (s, r, n=0.05) => epochs.map((_,i) => s * Math.exp(-r*i) * (1+(Math.random()-0.5)*n));
    const genRise = (t, r, n=0.03) => epochs.map((_,i) => t*(1-Math.exp(-r*i))*(1+(Math.random()-0.5)*n));

    charts.lossChart.data.labels = epochs;
    charts.lossChart.data.datasets[0].data = genDecay(2.5, 0.04, 0.15);
    charts.lossChart.data.datasets[1].data = genDecay(2.8, 0.03, 0.18);
    charts.lossChart.data.datasets[2].data = genDecay(1.8, 0.035, 0.12);
    charts.lossChart.update('none');

    charts.mapChart.data.labels = epochs;
    charts.mapChart.data.datasets[0].data = genRise(0.92, 0.04, 0.05);
    charts.mapChart.data.datasets[1].data = genRise(0.68, 0.035, 0.06);
    charts.mapChart.update('none');

    charts.lrChart.data.labels = epochs;
    const baseLr = 0.01;
    charts.lrChart.data.datasets[0].data = epochs.map((_,i) => baseLr*0.5*(1+Math.cos(Math.PI*i/100)));
    charts.lrChart.data.datasets[1].data = epochs.map((_,i) => baseLr*0.1*0.5*(1+Math.cos(Math.PI*i/100)));
    charts.lrChart.data.datasets[2].data = epochs.map((_,i) => baseLr*0.01*0.5*(1+Math.cos(Math.PI*i/100)));
    charts.lrChart.update('none');

    charts.distillChart.data.labels = epochs;
    charts.distillChart.data.datasets[0].data = epochs.map(i => 0.5+0.25*Math.sin(i*0.1)*Math.exp(-i*0.008));
    charts.distillChart.data.datasets[1].data = epochs.map(i => 6-4.5*(1-Math.cos(Math.PI*i/100))/2);
    charts.distillChart.data.datasets[2].data = genDecay(3.0, 0.045, 0.2);
    charts.distillChart.update('none');

    // PR curve sample
    if (charts.prChart) {
        const recall = Array.from({length: 100}, (_, i) => i/99);
        const precision = recall.map(r => 0.95 * Math.pow(r, -0.05) * (1-Math.exp(-10*r)) + (Math.random()-0.5)*0.02);
        charts.prChart.data.labels = recall.map(r => r.toFixed(2));
        charts.prChart.data.datasets[0].data = precision.map(p => Math.min(Math.max(p, 0.3), 1));
        charts.prChart.update('none');
    }

    updateResultsTable({
        box_loss: { best: 0.3421, final: 0.3589, improvement: '+4.9%', trend: 'down' },
        cls_loss: { best: 0.2876, final: 0.3012, improvement: '+4.7%', trend: 'down' },
        dfl_loss: { best: 0.8923, final: 0.9145, improvement: '+2.5%', trend: 'down' },
        map50: { best: 0.9234, final: 0.9187, improvement: '-0.51%', trend: 'up' },
        map50_95: { best: 0.6845, final: 0.6789, improvement: '-0.82%', trend: 'up' },
        precision: { best: 0.8678, final: 0.8590, improvement: '-1.02%', trend: 'stable' },
        recall: { best: 0.8234, final: 0.8156, improvement: '-0.95%', trend: 'stable' },
    });
}

// ==================== Overview Stats ====================
function updateOverviewStats() {
    const stats = {
        'ov-map50': '92.34%',
        'ov-fps': '125 fps',
        'ov-params': '3.2M',
        'ov-time': '4h 23m'
    };
    Object.entries(stats).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    });
}

// ==================== Results Table (Enhanced with Trend) ====================
function updateResultsTable(data) {
    const tbody = document.getElementById('results-tbody');
    tbody.innerHTML = '';
    
    const icons = { box_loss:'functions', cls_loss:'category', dfl_loss:'show_chart', map50:'radar', map50_95:'gps_fixed', precision:'track_changes', recall:'replay' };
    const names = { box_loss:'Box Loss', cls_loss:'Classification Loss', dfl_loss:'DFL Loss', map50:'mAP@50', map50_95:'mAP@50-95', precision:'Precision', recall:'Recall' };
    const trendIcons = { up: '<span class="trend-icon up" style="color:#2E7D32">↑</span>', down: '<span class="trend-icon down" style="color:#F57C00">↓</span>', stable: '<span class="trend-icon stable" style="color:#49454F">→</span>' };
    
    Object.entries(data).forEach(([key, value]) => {
        const row = document.createElement('tr');
        const isPositive = value.improvement.includes('+') && !value.improvement.match(/\d+\.\d+%$/)?.includes('-') === false;
        const impColor = isPositive ? 'var(--md-success)' : 
                         value.improvement.includes('-') ? 'var(--md-warning)' : 'var(--md-primary)';
        row.innerHTML = `
            <td><span class="material-icons" style="vertical-align:-3px;margin-right:6px;font-size:16px;color:var(--md-primary)">${icons[key]||'analytics'}</span>${names[key]}</td>
            <td><strong>${value.best.toFixed(4)}</strong></td>
            <td>${value.final.toFixed(4)}</td>
            <td style="color:${impColor};font-weight:600">${value.improvement}</td>
            <td style="text-align:center">${trendIcons[value.trend||'stable']}</td>
        `;
        tbody.appendChild(row);
    });
}

// ==================== Loss Range Selector ====================
function updateLossRange() {
    const range = document.getElementById('loss-range-select').value;
    const chart = charts.lossChart;
    if (!chart) return;

    const labelsSource = chart._originalLabels || chart.data.labels;
    if (!labelsSource || !labelsSource.length) return;

    let startIdx = 0;
    const totalLabels = [...labelsSource];
    
    switch(range) {
        case 'last30': startIdx = Math.max(0, totalLabels.length - 30); break;
        case 'last10': startIdx = Math.max(0, totalLabels.length - 10); break;
        default: startIdx = 0;
    }

    // Re-filter data while preserving a stable original snapshot.
    chart.data.labels = totalLabels.slice(startIdx);
    chart.data.datasets.forEach(ds => {
        const source = ds._originalData || ds.data;
        ds.data = source.slice(startIdx);
    });
    chart.update('active');
    showToast(`显示 Epoch ${totalLabels[startIdx]} ~ ${totalLabels[totalLabels.length-1]}`, 'info');
}

// Override loadSampleData to store original data for filtering
const _origLoadSample = loadSampleData;
loadSampleData = function() {
    _origLoadSample();
    // Store original data references for range filtering
    Object.values(charts).forEach(chart => {
        if (chart && chart.data && chart.data.datasets) {
            chart.data.datasets.forEach(ds => {
                ds._originalData = [...ds.data];
            });
            chart._originalLabels = [...chart.data.labels];
        }
    });
};

// ==================== Auto Refresh ====================
function toggleAutoRefresh(checkbox) {
    stopAutoRefreshCountdown();

    if (checkbox.checked) {
        if (autoRefreshIntervalId) {
            clearInterval(autoRefreshIntervalId);
        }

        autoRefreshIntervalId = setInterval(() => {
            refreshMetrics(false);
            startAutoRefreshCountdown();
        }, 30000); // every 30s

        startAutoRefreshCountdown();
        showToast('已开启自动刷新（30秒间隔）', 'success');
    } else {
        stopAutoRefreshLoop();
        showToast('已关闭自动刷新', 'info');
    }
}

function stopAutoRefreshLoop() {
    if (autoRefreshIntervalId) {
        clearInterval(autoRefreshIntervalId);
        autoRefreshIntervalId = null;
    }
    stopAutoRefreshCountdown();
    const timerEl = document.getElementById('refresh-timer');
    if (timerEl) timerEl.textContent = '--';
}

function stopAutoRefreshCountdown() {
    if (refreshCountdownIntervalId) {
        clearInterval(refreshCountdownIntervalId);
        refreshCountdownIntervalId = null;
    }
}

function startAutoRefreshCountdown() {
    const timerEl = document.getElementById('refresh-timer');
    if (!timerEl) return;

    refreshSecondsLeft = 30;
    timerEl.textContent = `${refreshSecondsLeft}s 后刷新`;
    stopAutoRefreshCountdown();

    refreshCountdownIntervalId = setInterval(() => {
        refreshSecondsLeft -= 1;
        if (refreshSecondsLeft < 0) refreshSecondsLeft = 30;
        timerEl.textContent = `${refreshSecondsLeft}s 后刷新`;

        if (!document.getElementById('auto-refresh-toggle')?.checked) {
            stopAutoRefreshLoop();
        }
    }, 1000);
}

// ==================== Export Table ====================
function exportTable() {
    const table = document.getElementById('results-table');
    if (!table) return;
    
    let csv = '\uFEFF'; // BOM for Excel
    csv += Array.from(table.querySelectorAll('th')).map(th => th.textContent).join(',') + '\n';
    table.querySelectorAll('tbody tr').forEach(row => {
        csv += Array.from(row.querySelectorAll('td')).map(td => td.textContent.replace(/[↑↓→]/g,'').trim()).join(',') + '\n';
    });
    
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `training_results_${new Date().toISOString().slice(0,10)}.csv`;
    link.click();
    showToast('表格已导出 CSV', 'success');
}

// ==================== Data Loading & Refresh ====================
async function refreshMetrics(showMessage = true) {
    try {
        const data = await API.get('/api/metrics');
        const selector = document.getElementById('metrics-source');
        if (!selector) return;

        const currentValue = selector.value;
        selector.innerHTML = '<option value="">选择数据源...</option>';
        data.csv_metrics.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.name; opt.textContent = `${m.name} (${m.rows} 行)`;
            selector.appendChild(opt);
        });
        if (currentValue) selector.value = currentValue;
        if (showMessage) showToast('指标数据已刷新', 'success');
    } catch(e) { /* use sample */ }
}

async function loadMetricsData() {
    const source = document.getElementById('metrics-source').value;
    if (!source) return;
    showToast(`正在加载 ${source} 的数据...`, 'info');
}

// ==================== Chart Type Filter ====================
function setChartType(type) {
    currentChartType = type;
    
    // Update chip active state — map type values to expected Chinese labels
    const typeLabelMap = { loss: '损失', accuracy: '精度', lr: '学习率', all: '全部' };
    document.querySelectorAll('.chip').forEach(chip => {
        const isActive = chip.textContent.trim() === (typeLabelMap[type] || type);
        chip.classList.toggle('active', isActive);
    });
    
    const cardByType = [
        { type: 'loss', card: document.getElementById('loss-chart')?.closest('.chart-card') },
        { type: 'accuracy', card: document.getElementById('map-chart')?.closest('.chart-card') },
        { type: 'lr', card: document.getElementById('lr-chart')?.closest('.chart-card') },
        { type: 'accuracy', card: document.getElementById('distill-chart')?.closest('.chart-card') },
        { type: 'accuracy', card: document.getElementById('pr-chart')?.closest('.chart-card') },
        { type: 'accuracy', card: document.getElementById('class-chart')?.closest('.chart-card') },
        { type: 'always', card: document.getElementById('results-table')?.closest('.chart-card') }
    ];

    cardByType.forEach(({ type: cardType, card }) => {
        if (!card) return;
        const shouldShow = currentChartType === 'all' || cardType === currentChartType || cardType === 'always';
        card.style.display = shouldShow ? '' : 'none';
    });
}

// ==================== Export & Zoom ====================
function exportChart(chartId) {
    const canvas = document.getElementById(chartId);
    if (!canvas) return;
    const link = document.createElement('a');
    link.download = `${chartId}_${Date.now()}.png`;
    link.href = canvas.toDataURL('image/png');
    link.click();
    showToast('图表已导出 PNG', 'success');
}

function zoomChart(chartId) {
    const card = document.querySelector(`#${chartId}`).closest('.chart-card');
    if (card?.requestFullscreen) card.requestFullscreen();
}
