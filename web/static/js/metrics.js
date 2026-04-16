/* ============================================================
   EdgeDistillDet Web UI - Charts & Metrics Visualization (Enhanced)
   ============================================================ */

let currentChartType = 'loss';
const charts = {};
window.charts = charts;
const MetricsState = {
    sources: [],
    currentSource: '',
    currentData: null,
    initialized: false
};
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

    // mAP Chart (包含 Precision & Recall)
    charts.mapChart = new Chart(document.getElementById('map-chart'), {
        type: 'line',
        data: {
            labels: [], datasets: [
                { label: 'mAP@50', data: [], borderColor: ChartColors.success, backgroundColor: transparentize(ChartColors.success, 0.12), fill: true },
                { label: 'mAP@50-95', data: [], borderColor: ChartColors.tertiary, backgroundColor: transparentize(ChartColors.tertiary, 0.12), fill: true },
                { label: 'Precision', data: [], borderColor: '#F57C00', backgroundColor: transparentize('#F57C00', 0.08), fill: false, borderDash: [5, 3] },
                { label: 'Recall', data: [], borderColor: '#1565C0', backgroundColor: transparentize('#1565C0', 0.08), fill: false, borderDash: [5, 3] }
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
        }, options: {
            ...opts,
            plugins: {
                ...opts.plugins,
                tooltip: {
                    ...opts.plugins.tooltip,
                    callbacks: {
                        label(context) {
                            const value = context.raw;
                            const label = context.dataset.label || '';
                            if (value === null || value === undefined) return label;
                            const formatted = Number(value).toExponential(3);
                            return `${label}: ${formatted}`;
                        }
                    }
                }
            },
            scales: { ...opts.scales, y: { ...opts.scales.y, type: 'logarithmic', title: { display: true, text: 'Learning Rate (log)', color: ChartColors.titleColor } } }
        }
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
                labels: [],
                datasets: [
                    { label: 'mAP', data: [], backgroundColor: transparentize(ChartColors.primary, 0.75), borderColor: ChartColors.primary, borderRadius: 6, borderSkipped: false },
                    { label: 'Recall', data: [], backgroundColor: transparentize(ChartColors.success, 0.65), borderColor: ChartColors.success, borderRadius: 6, borderSkipped: false },
                    { label: 'Precision', data: [], backgroundColor: transparentize('#F57C00', 0.55), borderColor: '#F57C00', borderRadius: 6, borderSkipped: false }
                ]
            }, options: {
                ...opts,
                plugins: {
                    ...opts.plugins,
                    tooltip: {
                        ...opts.plugins.tooltip,
                        callbacks: {
                            label(context) {
                                const value = context.raw;
                                const label = context.dataset.label || '';
                                if (value === null || value === undefined) {
                                    return `${label}: 无数据`;
                                }
                                return `${label}: ${Number(value).toFixed(4)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: { ...opts.scales.x, grid: { display: false } },
                    y: { ...opts.scales.y, beginAtZero: true, max: 1, title: { display: true, text: 'Score', color: ChartColors.titleColor } }
                }
            }
        });
    } catch(e) {}

    refreshMetrics(false);
}

// Helper to create rgba from hex
function transparentize(hex, alpha) {
    if (hex.startsWith('rgba')) return hex;
    const r = parseInt(hex.slice(1,3), 16), g = parseInt(hex.slice(3,5), 16), b = parseInt(hex.slice(5,7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ==================== Sample Data ====================
function _showMetricsHidden(selector, hidden = true) {
    const el = document.getElementById(selector);
    if (!el) return;
    el.classList.toggle('hidden', hidden);
}

function showMetricsNoData(message = '暂无训练结果可展示。请先执行训练，并刷新后查看指标监控。') {
    _showMetricsHidden('metrics-empty', false);
    _showMetricsHidden('metrics-content', true);
    const tbody = document.getElementById('results-tbody');
    if (tbody) {
        tbody.innerHTML = `<tr><td colspan="5" class="empty-hint">${message}</td></tr>`;
    }
    updateOverviewStats();
    Object.values(charts).forEach(chart => {
        if (!chart || !chart.data) return;
        chart.data.labels = [];
        chart.data.datasets.forEach(ds => { ds.data = []; });
        chart.update('none');
    });
    document.getElementById('auto-refresh-toggle')?.checked && stopAutoRefreshLoop();
}

function showMetricsPanel() {
    _showMetricsHidden('metrics-empty', true);
    _showMetricsHidden('metrics-content', false);
}

function renderMetricsSource(data) {
    if (!data || !data.chart_series || !Array.isArray(data.chart_series.epochs) || !data.chart_series.epochs.length) {
        return showMetricsNoData();
    }

    showMetricsPanel();
    const { epochs, train_losses, map_series, lr_series, distill_series, pr_curve, class_performance } = data.chart_series;

    if (charts.lossChart) {
        charts.lossChart.data.labels = epochs;
        charts.lossChart.data.datasets[0].data = train_losses.box_loss || [];
        charts.lossChart.data.datasets[1].data = train_losses.cls_loss || [];
        charts.lossChart.data.datasets[2].data = train_losses.dfl_loss || [];
        charts.lossChart.update('none');
        charts.lossChart._originalLabels = [...epochs];
        charts.lossChart.data.datasets.forEach(ds => ds._originalData = [...ds.data]);
    }

    if (charts.mapChart) {
        charts.mapChart.data.labels = epochs;
        charts.mapChart.data.datasets[0].data = map_series.map50 || [];
        charts.mapChart.data.datasets[1].data = map_series.map50_95 || [];
        charts.mapChart.data.datasets[2].data = data.chart_series.precision_recall?.precision || [];
        charts.mapChart.data.datasets[3].data = data.chart_series.precision_recall?.recall || [];
        charts.mapChart.update('none');
        charts.mapChart._originalLabels = [...epochs];
        charts.mapChart.data.datasets.forEach(ds => ds._originalData = [...ds.data]);
    }

    if (charts.lrChart) {
        charts.lrChart.data.labels = epochs;
        charts.lrChart.data.datasets[0].data = lr_series.pg0 || [];
        charts.lrChart.data.datasets[1].data = lr_series.pg1 || [];
        charts.lrChart.data.datasets[2].data = lr_series.pg2 || [];
        charts.lrChart.update('none');
        charts.lrChart._originalLabels = [...epochs];
        charts.lrChart.data.datasets.forEach(ds => ds._originalData = [...ds.data]);
    }

    const distillCard = document.getElementById('card-distill-flow');
    if (distillCard) {
        const hasDistill = distill_series && Object.values(distill_series).some(arr => Array.isArray(arr) && arr.length);
        distillCard.style.display = hasDistill ? '' : 'none';
    }
    if (charts.distillChart) {
        charts.distillChart.data.labels = epochs;
        
        // 线性插值填充 null/undefined 值，确保图表连续显示
        const fillGaps = (arr) => {
            if (!arr || !arr.length) return [];
            const result = [...arr];
            // 前向填充：找到第一个非空值
            let lastValidIdx = result.findIndex(v => v !== null && v !== undefined);
            for (let i = 0; i < lastValidIdx; i++) result[i] = result[lastValidIdx];
            // 后向填充
            let lastValidVal = result[lastValidIdx];
            for (let i = lastValidIdx + 1; i < result.length; i++) {
                if (result[i] === null || result[i] === undefined) {
                    // 找到下一个有效值做线性插值
                    let nextIdx = -1;
                    for (let j = i + 1; j < result.length; j++) {
                        if (result[j] !== null && result[j] !== undefined) { nextIdx = j; break; }
                    }
                    if (nextIdx > 0) {
                        const t = (i - lastValidIdx) / (nextIdx - lastValidIdx);
                        result[i] = lastValidVal + (result[nextIdx] - lastValidVal) * t;
                    } else {
                        result[i] = lastValidVal;
                    }
                } else {
                    lastValidIdx = i;
                    lastValidVal = result[i];
                }
            }
            return result;
        };
        
        charts.distillChart.data.datasets[0].data = fillGaps(distill_series?.alpha || []);
        charts.distillChart.data.datasets[1].data = fillGaps(distill_series?.temperature || []);
        charts.distillChart.data.datasets[2].data = fillGaps(distill_series?.kd_loss || []);
        charts.distillChart.update('none');
    }

    const prData = pr_curve || {};
    const prCard = document.getElementById('pr-chart')?.closest('.chart-card');
    if (prCard) {
        const hasPr = Array.isArray(prData.recall) && prData.recall.length && Array.isArray(prData.precision) && prData.precision.length;
        prCard.style.display = hasPr ? '' : 'none';
    }
    if (charts.prChart && Array.isArray(prData.recall) && Array.isArray(prData.precision)) {
        charts.prChart.data.labels = prData.recall.map(r => Number(r).toFixed(2));
        charts.prChart.data.datasets[0].data = prData.precision;
        charts.prChart.update('none');
    }

    const classCard = document.getElementById('class-chart')?.closest('.chart-card');
    if (classCard) {
        // 即使没有按类数据，也显示卡片，使用总体 P/R 作为 fallback
        classCard.style.display = '';
        // 清理旧的按钮（如果存在）
        const oldBtn = classCard.querySelector('.gen-class-btn');
        if (oldBtn) oldBtn.remove();
    }
    if (charts.classChart) {
        const normalizeChartValue = (value) => {
            return value === null || value === undefined || value === '' || Number.isNaN(Number(value))
                ? null
                : Number(value);
        };

        if (data.chart_series.class_performance && Array.isArray(data.chart_series.class_performance.labels) && data.chart_series.class_performance.labels.length > 1) {
            // 过滤掉完全没有有效指标的类别，但保留只有 P/R 或只有 mAP 数据的类
            const labels = data.chart_series.class_performance.labels;
            const mapData = data.chart_series.class_performance.map || [];
            const recData = data.chart_series.class_performance.recall || [];
            const precData = data.chart_series.class_performance.precision || [];

            const validIndices = [];
            for (let i = 0; i < labels.length; i++) {
                const mapValue = normalizeChartValue(mapData[i]);
                const recValue = normalizeChartValue(recData[i]);
                const precValue = normalizeChartValue(precData[i]);
                if (mapValue !== null || recValue !== null || precValue !== null) {
                    validIndices.push(i);
                }
            }

            charts.classChart.data.labels = validIndices.map(i => labels[i]);
            charts.classChart.data.datasets[0].data = validIndices.map(i => normalizeChartValue(mapData[i]));
            charts.classChart.data.datasets[1].data = validIndices.map(i => normalizeChartValue(recData[i]));
            charts.classChart.data.datasets[2].data = validIndices.map(i => normalizeChartValue(precData[i]));
            charts.classChart.data.datasets[2].hidden = !precData.some(v => normalizeChartValue(v) !== null);
        } else {
            // 无按类别数据时 fallback 到总体指标
            const prData = data.chart_series.precision_recall || {};
            const lastP = prData.precision?.length ? normalizeChartValue(prData.precision[prData.precision.length - 1]) : null;
            const lastR = prData.recall?.length ? normalizeChartValue(prData.recall[prData.recall.length - 1]) : null;
            const mapSeries = data.chart_series.map_series || {};
            const lastMap = normalizeChartValue(mapSeries.map50?.length ? mapSeries.map50[mapSeries.map50.length - 1]
                : mapSeries.map50_95?.length ? mapSeries.map50_95[mapSeries.map50_95.length - 1] : null);
            charts.classChart.data.labels = ['Overall'];
            charts.classChart.data.datasets[0].data = [lastMap];
            charts.classChart.data.datasets[1].data = [lastR];
            charts.classChart.data.datasets[2].data = [lastP];
            charts.classChart.data.datasets[2].hidden = lastP === null;
        }
        charts.classChart.update('none');
    }

    updateOverviewStats(data.overview_stats);
    updateResultsTable(data.summary_metrics || {});
}

function loadSampleData() {
    // intentionally no-op: metrics visualization now depends on actual training outputs.
}

// ==================== Overview Stats ====================
function updateOverviewStats(stats = {}) {
    const defaults = {
        'ov-map50': '--',
        'ov-fps': '--',
        'ov-params': '--',
        'ov-time': '--'
    };
    const values = { ...defaults, ...stats };
    Object.entries(values).forEach(([id, value]) => {
        const el = document.getElementById(id);
        if (!el) return;
        const rawValue = typeof value === 'string' ? value.trim() : value;
        const numericValue = Number(rawValue);
        if (typeof Animations !== 'undefined' && !Animations.prefersReducedMotion && rawValue !== '--' && !Number.isNaN(numericValue)) {
            try {
                Animations.CountUpAnim.animateStat(id, numericValue);
                return;
            } catch (error) {
                console.warn('CountUpAnim failed for', id, error);
            }
        }
        el.textContent = rawValue === '--' ? '--' : String(value);
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
        selector.innerHTML = '';
        MetricsState.sources = data.csv_metrics || [];
        MetricsState.currentSource = '';

        if (!MetricsState.sources.length) {
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = '暂无训练结果可选';
            selector.appendChild(opt);
            showMetricsNoData('尚未检测到训练结果，请先训练模型并刷新。');
            return;
        }

        MetricsState.sources.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.has_results ? m.path : '';
            opt.textContent = m.display_name || m.name;
            if (!m.has_results) {
                opt.disabled = true;
            }
            selector.appendChild(opt);
        });

        const preferred = MetricsState.sources.some(m => m.has_results && m.path === currentValue)
            ? currentValue
            : MetricsState.sources.find(m => m.has_results)?.path || '';
        selector.value = preferred;
        if (preferred) {
            MetricsState.currentSource = preferred;
            await loadMetricsData();
        }
        if (showMessage) showToast('指标数据已刷新', 'success');
    } catch (e) {
        showMetricsNoData('指标加载失败，请检查训练结果路径或后端服务。');
        console.error(e);
    }
}

async function loadMetricsData() {
    const selector = document.getElementById('metrics-source');
    const source = selector?.value || MetricsState.currentSource;
    if (!source) {
        return showMetricsNoData();
    }
    showToast(`正在加载训练结果数据...`, 'info');
    try {
        const data = await API.get(`/api/metrics?source=${encodeURIComponent(source)}`);
        MetricsState.currentSource = source;
        MetricsState.currentData = data;
        renderMetricsSource(data);
        showToast(`已加载数据源：${source}`, 'success');
    } catch (e) {
        showMetricsNoData('加载训练结果失败，请稍后重试。');
        console.error(e);
    }
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
