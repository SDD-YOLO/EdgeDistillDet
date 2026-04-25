import { useCallback, useEffect, useRef, useState } from "react";
import { fetchMetricsBySource, fetchMetricsList } from "../../../api/metricsApi";

export function useMetricsData({ toast }) {
  const [sources, setSources] = useState([]);
  const [source, setSource] = useState("");
  const [overview, setOverview] = useState({});
  const [summaryMetrics, setSummaryMetrics] = useState({});
  const [hasData, setHasData] = useState(false);
  const [chartSeriesState, setChartSeriesState] = useState(null);
  const rawSeriesRef = useRef(null);
  const lastDataFingerprintRef = useRef("");

  const loadMetricsData = useCallback(async (sourcePath, silent = false) => {
    if (!sourcePath) return;
    try {
      const data = await fetchMetricsBySource(sourcePath);
      const epochs = data.chart_series?.epochs || [];
      const lastEpoch = epochs.length ? epochs[epochs.length - 1] : null;
      const nextFingerprint = `${sourcePath}|${data.rows || 0}|${lastEpoch ?? "na"}`;
      const prevFingerprint = lastDataFingerprintRef.current;
      const changed = prevFingerprint !== nextFingerprint;
      lastDataFingerprintRef.current = nextFingerprint;
      setHasData(epochs.length > 0);
      setOverview(data.overview_stats || {});
      setSummaryMetrics(data.summary_metrics || {});
      const nextSeries = data.chart_series || null;
      rawSeriesRef.current = nextSeries;
      setChartSeriesState(nextSeries);
      if (!silent) toast(changed ? "图表已更新" : "暂无新数据，已是最新结果", changed ? "success" : "info");
    } catch (error) {
      setHasData(false);
      setChartSeriesState(null);
      toast(error.message, "error");
    }
  }, [toast]);

  const refreshSources = useCallback(async (showToast = false) => {
    try {
      const data = await fetchMetricsList();
      const available = Array.isArray(data.csv_metrics) ? data.csv_metrics.filter((x) => x.has_results) : [];
      const nextSource = available.some((it) => it.path === source) ? source : (available[0]?.path || "");
      setSources(available);
      if (!available.length) {
        setHasData(false);
        if (showToast) toast("暂无训练结果可展示", "info");
        return;
      }
      const sourceUnchanged = nextSource && nextSource === source;
      setSource(nextSource);
      if (sourceUnchanged) {
        loadMetricsData(nextSource, !showToast);
        return;
      }
      if (showToast) toast("指标来源已刷新", "success");
    } catch (error) {
      toast(error.message, "error");
    }
  }, [loadMetricsData, source, toast]);

  useEffect(() => {
    refreshSources();
  }, [refreshSources]);

  useEffect(() => {
    if (!source) return;
    loadMetricsData(source, true);
  }, [loadMetricsData, source]);

  return {
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
  };
}
