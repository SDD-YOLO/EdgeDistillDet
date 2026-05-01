import { useEffect, useRef, useState } from "react";
import {
  BarController,
  BarElement,
  CategoryScale,
  Chart,
  Legend,
  LineController,
  LineElement,
  LinearScale,
  LogarithmicScale,
  PointElement,
  Tooltip
} from "chart.js";
import { M3Select } from "../../components/forms/M3Select";
import { Button } from "../../components/ui/button";
import { EpochRangeHint } from "./components/EpochRangeHint";
import { OverviewCard } from "./components/OverviewCard";
import { useMetricsData } from "./hooks/useMetricsData";
import { useMetricsUiStore } from "../../stores/metricsUiStore";

Chart.register(
  BarController,
  BarElement,
  CategoryScale,
  LinearScale,
  LogarithmicScale,
  PointElement,
  LineElement,
  LineController,
  Legend,
  Tooltip
);

const EPOCH_RANGE_OPTIONS = [
  { value: "all", label: "全部 Epochs" },
  { value: "last30", label: "最近 30" },
  { value: "last10", label: "最近 10" }
];

function normalizeEpochRange(range) {
  if (range == null || range === "") return "all";
  const s = String(range).trim();
  if (s === "last30") return "last30";
  if (s === "last10") return "last10";
  return "all";
}

function epochStartIndex(range, totalLen) {
  const r = normalizeEpochRange(range);
  if (!totalLen) return 0;
  if (r === "last30") return Math.max(0, totalLen - 30);
  if (r === "last10") return Math.max(0, totalLen - 10);
  return 0;
}

function minPositiveLen(...lengths) {
  const nums = lengths.filter((n) => typeof n === "number" && n > 0);
  if (!nums.length) return 0;
  return Math.min(...nums);
}

function padToLength(array, targetLen) {
  /**
   * 将数组补充到指定长度，缺失的位置用 null 填充。
   * 这允许不同长度的数据数组一起显示，缺失点显示为空白而不是完全隐藏。
   */
  if (!Array.isArray(array)) return new Array(targetLen).fill(null);
  const result = [...array];
  while (result.length < targetLen) {
    result.push(null);
  }
  return result.slice(0, targetLen);
}

function renderTrendIcon(trend) {
  const normalized = String(trend || "stable").toLowerCase();
  if (normalized === "up") {
    return <span className="material-icons trend-icon trend-up" title="上升" aria-label="上升">trending_up</span>;
  }
  if (normalized === "down") {
    return <span className="material-icons trend-icon trend-down" title="下降" aria-label="下降">trending_down</span>;
  }
  return <span className="material-icons trend-icon trend-stable" title="持平" aria-label="持平">trending_flat</span>;
}

function MetricsPanel({ toast, active }) {
  const chartType = useMetricsUiStore((state) => state.chartType);
  const rangeLoss = useMetricsUiStore((state) => state.rangeLoss);
  const rangeMap = useMetricsUiStore((state) => state.rangeMap);
  const rangeLr = useMetricsUiStore((state) => state.rangeLr);
  const rangeDistill = useMetricsUiStore((state) => state.rangeDistill);
  const autoRefresh = useMetricsUiStore((state) => state.autoRefresh);
  const refreshLeft = useMetricsUiStore((state) => state.refreshLeft);
  const setChartType = useMetricsUiStore((state) => state.setChartType);
  const setRangeLoss = useMetricsUiStore((state) => state.setRangeLoss);
  const setRangeMap = useMetricsUiStore((state) => state.setRangeMap);
  const setRangeLr = useMetricsUiStore((state) => state.setRangeLr);
  const setRangeDistill = useMetricsUiStore((state) => state.setRangeDistill);
  const setAutoRefresh = useMetricsUiStore((state) => state.setAutoRefresh);
  const setRefreshLeft = useMetricsUiStore((state) => state.setRefreshLeft);
  const [themeMode, setThemeMode] = useState(() => document.documentElement.getAttribute("data-theme") || "light");

  const lossRef = useRef(null);
  const mapRef = useRef(null);
  const lrRef = useRef(null);
  const distillRef = useRef(null);
  const classRef = useRef(null);
  const chartInstances = useRef({});

  const {
    sources,
    source,
    setSource,
    overview,
    summaryMetrics,
    hasData,
    chartSeriesState,
    rawSeriesRef,
    refreshSources,
    loadMetricsData
  } = useMetricsData({ toast, autoRefresh });

  useEffect(() => {
    if (!active || !rawSeriesRef.current) return;
    renderAllCharts(chartInstances, {
      lossRef,
      mapRef,
      lrRef,
      distillRef,
      classRef
    }, rawSeriesRef.current, {
      loss: rangeLoss,
      map: rangeMap,
      lr: rangeLr,
      distill: rangeDistill
    });
  }, [rangeLoss, rangeMap, rangeLr, rangeDistill, active]);

  useEffect(() => {
    if (!active || !hasData || !chartSeriesState) return;
    renderAllCharts(chartInstances, {
      lossRef,
      mapRef,
      lrRef,
      distillRef,
      classRef
    }, chartSeriesState, {
      loss: rangeLoss,
      map: rangeMap,
      lr: rangeLr,
      distill: rangeDistill
    });
  }, [hasData, chartSeriesState, rangeLoss, rangeMap, rangeLr, rangeDistill, themeMode, active]);

  useEffect(() => {
    const root = document.documentElement;
    const observer = new MutationObserver(() => {
      const nextTheme = root.getAttribute("data-theme") || "light";
      setThemeMode((prev) => (prev === nextTheme ? prev : nextTheme));
    });
    observer.observe(root, { attributes: true, attributeFilter: ["data-theme"] });
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!autoRefresh || !source) {
      setRefreshLeft(30);
      return undefined;
    }
    setRefreshLeft(30);
    const tick = window.setInterval(() => {
      if (document.hidden) return;
      setRefreshLeft((prev) => (prev <= 1 ? 30 : prev - 1));
    }, 1000);
    return () => {
      window.clearInterval(tick);
    };
  }, [autoRefresh, source, setRefreshLeft]);

  useEffect(() => () => {
    Object.values(chartInstances.current).forEach((chart) => chart.destroy());
  }, []);

  const exportChart = (canvasRef, fileName) => {
    const canvas = canvasRef?.current;
    if (!canvas) return;
    const link = document.createElement("a");
    link.download = `${fileName}_${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  const exportTable = () => {
    const entries = Object.entries(summaryMetrics || {});
    if (!entries.length) {
      toast("暂无可导出数据", "warning");
      return;
    }
    let csv = "\uFEFF指标,最优值,最终值,改善幅度,趋势\n";
    entries.forEach(([name, val]) => {
      csv += `${name},${val.best},${val.final},${val.improvement || ""},${val.trend || ""}\n`;
    });
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `training_results_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const showLoss = chartType === "all" || chartType === "loss";
  const showAcc = chartType === "all" || chartType === "accuracy";
  const showLr = chartType === "all" || chartType === "lr";

  const epochTotal = Array.isArray(chartSeriesState?.epochs) ? chartSeriesState.epochs.length : 0;

  return (
    <div className={`tab-panel console-module-panel ${active ? "active" : ""}`} id="panel-metrics" aria-hidden={!active}>
      <div className="metrics-toolbar">
        <div className="toolbar-left">
          <Button variant="outline" className="metrics-refresh-btn" onClick={() => refreshSources(true)}>
            <span className="material-icons">refresh</span>刷新数据
          </Button>
          <M3Select
            className="metrics-source-select"
            value={source}
            onChange={(next) => setSource(next)}
            options={
              sources.length === 0
                ? [{ value: "", label: "暂无训练结果可选" }]
                : sources.map((item) => ({ value: item.path, label: item.display_name || item.name }))
            }
            ariaLabel="选择指标来源"
          />
        </div>
        <div className="toolbar-right metrics-toolbar-right">
          <div className="chip-group">
            <Button variant={chartType === "loss" ? "secondary" : "outline"} className={`chip ${chartType === "loss" ? "active" : ""}`} onClick={() => setChartType("loss")}>损失</Button>
            <Button variant={chartType === "accuracy" ? "secondary" : "outline"} className={`chip ${chartType === "accuracy" ? "active" : ""}`} onClick={() => setChartType("accuracy")}>精度</Button>
            <Button variant={chartType === "lr" ? "secondary" : "outline"} className={`chip ${chartType === "lr" ? "active" : ""}`} onClick={() => setChartType("lr")}>学习率</Button>
            <Button variant={chartType === "all" ? "secondary" : "outline"} className={`chip ${chartType === "all" ? "active" : ""}`} onClick={() => setChartType("all")}>全部</Button>
          </div>
          <div className="auto-refresh-bar">
            <label className="auto-refresh-label">
              <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
              自动刷新图表数据
            </label>
            <span className="auto-refresh-interval">{autoRefresh ? `${refreshLeft}s 后刷新` : "--"}</span>
          </div>
        </div>
      </div>

      {!hasData ? (
        <div className="metrics-empty">
          <div className="empty-placeholder">
            <span className="material-icons">analytics</span>
            <p>暂无训练结果可展示。请先执行训练，并刷新后查看指标监控。</p>
          </div>
        </div>
      ) : (
        <div id="metrics-content">
          <div className="metrics-overview">
            <OverviewCard label="最佳 mAP@50" value={overview["ov-map50"] || "--"} icon="trending_up" />
            <OverviewCard label="推理 FPS (GPU)" value={overview["ov-fps"] || "--"} icon="speed" />
            <OverviewCard label="模型参数量" value={overview["ov-params"] || "--"} icon="memory" />
            <OverviewCard label="训练总耗时" value={overview["ov-time"] || "--"} icon="timer" />
          </div>

          <div className="charts-grid">
            <div className="chart-card wide" style={{ display: showLoss ? "" : "none" }}>
              <div className="chart-header">
                <div className="chart-title-block">
                  <h3>训练损失曲线</h3>
                  <EpochRangeHint total={epochTotal} range={rangeLoss} />
                </div>
                <div className="chart-actions">
                  <M3Select
                    className="mini-select"
                    value={rangeLoss}
                    onChange={(next) => setRangeLoss(next)}
                    options={EPOCH_RANGE_OPTIONS}
                    ariaLabel="选择损失范围"
                  />
                  <Button size="icon" variant="outline" className="btn-icon-sm" onClick={() => exportChart(lossRef, "loss-chart")}><span className="material-icons">download</span></Button>
                </div>
              </div>
              <div className="chart-body"><canvas ref={lossRef} /></div>
            </div>

            <div className="chart-card" style={{ display: showAcc ? "" : "none" }}>
              <div className="chart-header">
                <div className="chart-title-block">
                  <h3>mAP 曲线</h3>
                  <EpochRangeHint total={epochTotal} range={rangeMap} />
                </div>
                <div className="chart-actions">
                  <M3Select
                    className="mini-select"
                    value={rangeMap}
                    onChange={(next) => setRangeMap(next)}
                    options={EPOCH_RANGE_OPTIONS}
                    ariaLabel="选择 mAP 曲线范围"
                  />
                  <Button size="icon" variant="outline" className="btn-icon-sm" onClick={() => exportChart(mapRef, "map-chart")}><span className="material-icons">download</span></Button>
                </div>
              </div>
              <div className="chart-body"><canvas ref={mapRef} /></div>
            </div>

            <div className="chart-card" style={{ display: showLr ? "" : "none" }}>
              <div className="chart-header">
                <div className="chart-title-block">
                  <h3>学习率变化</h3>
                  <EpochRangeHint total={epochTotal} range={rangeLr} />
                </div>
                <div className="chart-actions">
                  <M3Select
                    className="mini-select"
                    value={rangeLr}
                    onChange={(next) => setRangeLr(next)}
                    options={EPOCH_RANGE_OPTIONS}
                    ariaLabel="选择学习率曲线范围"
                  />
                  <Button size="icon" variant="outline" className="btn-icon-sm" onClick={() => exportChart(lrRef, "lr-chart")}><span className="material-icons">download</span></Button>
                </div>
              </div>
              <div className="chart-body"><canvas ref={lrRef} /></div>
            </div>

            <div className="chart-card wide" style={{ display: showAcc ? "" : "none" }}>
              <div className="chart-header">
                <div className="chart-title-block">
                  <h3>蒸馏指标 (Alpha/Temperature 动态变化)</h3>
                  <EpochRangeHint total={epochTotal} range={rangeDistill} />
                </div>
                <div className="chart-actions">
                  <M3Select
                    className="mini-select"
                    value={rangeDistill}
                    onChange={(next) => setRangeDistill(next)}
                    options={EPOCH_RANGE_OPTIONS}
                    ariaLabel="选择蒸馏指标范围"
                  />
                </div>
              </div>
              <div className="chart-body"><canvas ref={distillRef} /></div>
            </div>

            <div className="chart-card wide" style={{ display: showAcc ? "" : "none" }}>
              <div className="chart-header">
                <div className="chart-title-block">
                  <h3>各类别性能分布</h3>
                </div>
              </div>
              <div className="chart-body"><canvas ref={classRef} /></div>
            </div>

            <div className="chart-card full-width">
              <div className="chart-header">
                <h3>训练结果摘要</h3>
                <Button variant="secondary" className="sm-btn" onClick={exportTable}>
                  <span className="material-icons">table_chart</span>导出表格
                </Button>
              </div>
              <div className="table-container">
                <table className="md-table">
                  <thead>
                    <tr><th>指标</th><th>最优值</th><th>最终值</th><th>改善幅度</th><th>趋势</th></tr>
                  </thead>
                  <tbody>
                    {Object.entries(summaryMetrics).length === 0 ? (
                      <tr><td colSpan={5} className="empty-hint">暂无数据，请先完成一次训练</td></tr>
                    ) : (
                      Object.entries(summaryMetrics).map(([metric, val]) => (
                        <tr key={metric}>
                          <td>{metric}</td>
                          <td>{Number(val.best).toFixed(4)}</td>
                          <td>{Number(val.final).toFixed(4)}</td>
                          <td>{val.improvement || "--"}</td>
                          <td>{renderTrendIcon(val.trend)}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function getChartTheme() {
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  return {
    text: isDark ? "rgba(230, 237, 243, 0.92)" : "rgba(28, 27, 31, 0.85)",
    muted: isDark ? "rgba(150, 160, 175, 0.78)" : "rgba(98, 91, 113, 0.75)",
    grid: isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(103, 80, 164, 0.10)",
    tooltipBg: isDark ? "rgba(17, 22, 28, 0.96)" : "rgba(255, 255, 255, 0.96)",
    tooltipBorder: isDark ? "rgba(80, 90, 108, 0.5)" : "rgba(103, 80, 164, 0.25)"
  };
}

function formatMetricValueForTooltip(chartKey, value) {
  // JSON null → 此处应为「无数据」；避免 Number(null)===0 误显示为 0
  if (value === null || value === undefined || value === "") return "--";
  const num = Number(value);
  if (!Number.isFinite(num)) return "--";
  if (chartKey === "lr") {
    if (num === 0) return "0";
    if (Math.abs(num) < 0.001) return num.toExponential(6);
    return num.toFixed(6);
  }
  return num.toFixed(4);
}

function compactClassSeries(cls = {}) {
  const labels = Array.isArray(cls.labels) ? cls.labels : [];
  const map = Array.isArray(cls.map) ? cls.map : [];
  const recall = Array.isArray(cls.recall) ? cls.recall : [];
  const precision = Array.isArray(cls.precision) ? cls.precision : [];
  const toMetric = (arr, idx) => {
    const raw = arr[idx];
    if (raw == null || raw === "") return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  };
  const next = { labels: [], map: [], recall: [], precision: [] };
  for (let i = 0; i < labels.length; i += 1) {
    const m = toMetric(map, i);
    const r = toMetric(recall, i);
    const p = toMetric(precision, i);
    if (m == null && r == null && p == null) continue;
    next.labels.push(labels[i]);
    next.map.push(m);
    next.recall.push(r);
    next.precision.push(p);
  }
  return next;
}

function renderLineChart(instancesRef, canvas, key, labels, datasets, options = {}) {
  if (!canvas) return;
  if (instancesRef.current[key]) instancesRef.current[key].destroy();
  try {
    const theme = getChartTheme();
    const { scales: extraScales = {}, ...restOptions } = options;
    instancesRef.current[key] = new Chart(canvas, {
      type: "line",
      data: {
        labels,
        datasets: datasets.map((item) => ({
          label: item.label,
          data: item.data,
          borderColor: item.color,
          backgroundColor: `${item.color}20`,
          fill: false,
          borderWidth: 2.5,
          tension: 0.38,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHitRadius: 10
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: "index",
          intersect: false
        },
        animation: false,
        transitions: {
          active: { animation: { duration: 0 } },
          show: { animation: { duration: 0 } },
          hide: { animation: { duration: 0 } }
        },
        elements: {
          line: {
            capBezierPoints: true
          }
        },
        scales: {
          x: {
            type: "category",
            grid: { color: theme.grid, drawBorder: false },
            ticks: { color: theme.muted, maxTicksLimit: 10 },
            ...(extraScales.x || {})
          },
          y: {
            grid: { color: theme.grid, drawBorder: false },
            ticks: { color: theme.muted },
            ...(extraScales.y || {})
          }
        },
        plugins: {
          legend: {
            position: "top",
            labels: {
              color: theme.text,
              usePointStyle: true,
              pointStyle: "circle",
              boxWidth: 8,
              boxHeight: 8,
              padding: 14
            }
          },
          tooltip: {
            backgroundColor: theme.tooltipBg,
            borderColor: theme.tooltipBorder,
            borderWidth: 1,
            titleColor: theme.text,
            bodyColor: theme.text,
            displayColors: true,
            padding: 10,
            callbacks: {
              label: (ctx) => {
                const chartKey = key;
                const value = chartKey === "lr" ? ctx?.raw : (ctx?.parsed?.y ?? ctx?.raw);
                return `${ctx.dataset.label}: ${formatMetricValueForTooltip(chartKey, value)}`;
              }
            }
          }
        },
        ...restOptions
      }
    });
  } catch {}
}

function renderBarChart(instancesRef, canvas, key, labels, datasets) {
  if (!canvas) return;
  if (instancesRef.current[key]) instancesRef.current[key].destroy();
  try {
    const theme = getChartTheme();
    instancesRef.current[key] = new Chart(canvas, {
      type: "bar",
      data: {
        labels,
        datasets: (datasets || []).map((item) => ({
          ...item,
          borderWidth: 1.2,
          borderRadius: 8,
          borderSkipped: false,
          maxBarThickness: 26,
          categoryPercentage: 0.72
        }))
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        animation: false,
        transitions: {
          active: { animation: { duration: 0 } },
          show: { animation: { duration: 0 } },
          hide: { animation: { duration: 0 } }
        },
        scales: {
          x: {
            grid: { color: theme.grid, drawBorder: false },
            ticks: { color: theme.muted, maxRotation: 35, minRotation: 0 }
          },
          y: {
            grid: { color: theme.grid, drawBorder: false },
            ticks: { color: theme.muted },
            suggestedMin: 0,
            suggestedMax: 1
          }
        },
        plugins: {
          legend: {
            position: "top",
            labels: {
              color: theme.text,
              usePointStyle: true,
              pointStyle: "rectRounded",
              boxWidth: 10,
              boxHeight: 10,
              padding: 14
            }
          },
          tooltip: {
            backgroundColor: theme.tooltipBg,
            borderColor: theme.tooltipBorder,
            borderWidth: 1,
            titleColor: theme.text,
            bodyColor: theme.text,
            padding: 10,
            callbacks: {
              label: (ctx) => {
                const rawPt =
                  ctx?.dataset?.data != null && ctx.dataIndex != null
                    ? ctx.dataset.data[ctx.dataIndex]
                    : undefined;
                const value = rawPt !== undefined ? rawPt : (ctx?.parsed?.y ?? ctx?.raw);
                return `${ctx.dataset.label}: ${formatMetricValueForTooltip(key, value)}`;
              }
            }
          }
        }
      }
    });
  } catch {}
}

function fillGaps(values) {
  if (!Array.isArray(values)) return [];
  const result = [...values];
  let last = null;
  for (let i = 0; i < result.length; i += 1) {
    const v = result[i];
    if (v == null || Number.isNaN(Number(v))) {
      if (last != null) result[i] = last;
      continue;
    }
    last = Number(v);
    result[i] = Number(v);
  }
  return result;
}

function renderAllCharts(instancesRef, refs, chartSeries, ranges) {
  const r = ranges || {};
  const epochsAll = chartSeries?.epochs || [];
  const trainLoss = chartSeries?.train_losses || {};
  const mapSeries = chartSeries?.map_series || {};
  const lrSeries = chartSeries?.lr_series || {};
  const distill = chartSeries?.distill_series || {};
  const prEpoch = chartSeries?.precision_recall || {};

  // ✅ 改进：移除过于严格的长度限制，允许数据缺失
  // 前：使用 minPositiveLen 强制所有数据长度相同，导致缺任何一个就整个图表无法显示
  // 后：直接使用 epochs 长度，缺失的数据点保留为 null

  const capLoss = epochsAll.length;
  const eLoss = epochsAll.slice(0, capLoss);
  const startLoss = epochStartIndex(r.loss, eLoss.length);
  const epochsLoss = eLoss.slice(startLoss);
  const boxLoss = padToLength((trainLoss.box_loss || []).slice(0, capLoss), capLoss).slice(startLoss);
  const clsLoss = padToLength((trainLoss.cls_loss || []).slice(0, capLoss), capLoss).slice(startLoss);
  const dflLoss = padToLength((trainLoss.dfl_loss || []).slice(0, capLoss), capLoss).slice(startLoss);

  const capMap = epochsAll.length;
  const eMap = epochsAll.slice(0, capMap);
  const startMap = epochStartIndex(r.map, eMap.length);
  const epochsMap = eMap.slice(startMap);
  const map50 = padToLength((mapSeries.map50 || []).slice(0, capMap), capMap).slice(startMap);
  const map5095 = padToLength((mapSeries.map50_95 || []).slice(0, capMap), capMap).slice(startMap);
  const prPrec = padToLength((prEpoch.precision || []).slice(0, capMap), capMap).slice(startMap);
  const prRec = padToLength((prEpoch.recall || []).slice(0, capMap), capMap).slice(startMap);

  const capLr = epochsAll.length;
  const eLr = epochsAll.slice(0, capLr);
  const startLr = epochStartIndex(r.lr, eLr.length);
  const epochsLr = eLr.slice(startLr);
  const lr0 = padToLength((lrSeries.pg0 || []).slice(0, capLr), capLr).slice(startLr);
  const lr1 = padToLength((lrSeries.pg1 || []).slice(0, capLr), capLr).slice(startLr);
  const lr2 = padToLength((lrSeries.pg2 || []).slice(0, capLr), capLr).slice(startLr);

  // ✅ 蒸馏指标改进：允许部分缺失
  const dLenA = (distill.alpha || []).length;
  const dLenT = (distill.temperature || []).length;
  const dLenK = (distill.kd_loss || []).length;
  const dMax = Math.max(dLenA, dLenT, dLenK, 0);
  const capDistill = dMax > 0 ? Math.min(epochsAll.length, dMax) : 0;
  
  let epochsDistill = [];
  let dAlpha = [];
  let dTemp = [];
  let dKd = [];
  
  if (capDistill > 0) {
    const eDistill = epochsAll.slice(0, capDistill);
    const startDistill = epochStartIndex(r.distill, eDistill.length);
    epochsDistill = eDistill.slice(startDistill);
    dAlpha = padToLength((distill.alpha || []).slice(0, capDistill), capDistill).slice(startDistill);
    dTemp = padToLength((distill.temperature || []).slice(0, capDistill), capDistill).slice(startDistill);
    dKd = padToLength((distill.kd_loss || []).slice(0, capDistill), capDistill).slice(startDistill);
  }

  const cls = compactClassSeries(chartSeries?.class_performance || {});
  const clsLabelsFull = cls.labels?.length ? cls.labels : ["Overall"];

  renderLineChart(instancesRef, refs.lossRef.current, "loss", epochsLoss, [
    { label: "Box Loss", data: boxLoss, color: "#6750A4" },
    { label: "CLS Loss", data: clsLoss, color: "#F57C00" },
    { label: "DFL Loss", data: dflLoss, color: "#1565C0" }
  ]);

  renderLineChart(instancesRef, refs.mapRef.current, "map", epochsMap, [
    { label: "mAP50", data: map50, color: "#2E7D32" },
    { label: "mAP50-95", data: map5095, color: "#7D5260" },
    { label: "Precision", data: prPrec, color: "#F57C00" },
    { label: "Recall", data: prRec, color: "#1565C0" }
  ]);

  renderLineChart(instancesRef, refs.lrRef.current, "lr", epochsLr, [
    { label: "LR pg0", data: lr0, color: "#6750A4" },
    { label: "LR pg1", data: lr1, color: "#625B71" },
    { label: "LR pg2", data: lr2, color: "#7D5260" }
  ], { scales: { y: { type: "logarithmic" } } });

  // ✅ 蒸馏图表改进：即使数据不完整也可以显示
  if (epochsDistill.length > 0) {
    renderLineChart(instancesRef, refs.distillRef.current, "distill", epochsDistill, [
      { label: "Alpha", data: fillGaps(dAlpha), color: "#6750A4" },
      { label: "Temperature T", data: fillGaps(dTemp), color: "#B3261E" },
      { label: "KD Loss", data: fillGaps(dKd), color: "#F57C00" }
    ]);
  } else {
    // 清空蒸馏图表
    const distillChart = instancesRef.current?.distill;
    if (distillChart) {
      distillChart.destroy();
      delete instancesRef.current.distill;
    }
  }

  renderBarChart(instancesRef, refs.classRef.current, "class", clsLabelsFull, [
    { label: "mAP", data: cls.map || [0], backgroundColor: "#6750A4AA", borderColor: "#6750A4" },
    { label: "Recall", data: cls.recall || [0], backgroundColor: "#2E7D32AA", borderColor: "#2E7D32" },
    { label: "Precision", data: cls.precision || [0], backgroundColor: "#F57C00AA", borderColor: "#F57C00" }
  ]);
}

/** 侧边栏「常用工具」：点击即作为用户消息发送（高频问句） */

export default MetricsPanel;
