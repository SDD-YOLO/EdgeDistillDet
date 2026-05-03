import { useCallback, useEffect, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  fetchMetricsBySource,
  fetchMetricsList,
} from "../../../api/metricsApi";

export function useMetricsData({ toast, autoRefresh = false }) {
  const [sources, setSources] = useState([]);
  const [source, setSource] = useState("");
  const [refreshToken, setRefreshToken] = useState(0);
  const [overview, setOverview] = useState({});
  const [summaryMetrics, setSummaryMetrics] = useState({});
  const [hasData, setHasData] = useState(false);
  const [chartSeriesState, setChartSeriesState] = useState(null);
  const rawSeriesRef = useRef(null);
  const lastDataFingerprintRef = useRef("");

  const sourcesQuery = useQuery({
    queryKey: ["metrics", "sources"],
    queryFn: fetchMetricsList,
    staleTime: 30000,
    refetchOnWindowFocus: false,
    retry: 2,
  });

  useEffect(() => {
    const available = Array.isArray(sourcesQuery.data?.csv_metrics)
      ? sourcesQuery.data.csv_metrics.filter((x) => x.has_results)
      : [];
    setSources(available);
    if (!available.length) {
      setHasData(false);
      if (source) setSource("");
      return;
    }

    const nextSource = available.some((it) => it.path === source)
      ? source
      : available[0]?.path || "";
    if (nextSource !== source) {
      setSource(nextSource);
    }
  }, [source, sourcesQuery.data]);

  const metricsQuery = useQuery({
    queryKey: ["metrics", "source", source, refreshToken],
    queryFn: ({ signal }) => fetchMetricsBySource(source, { signal }),
    enabled: Boolean(source),
    staleTime: 10000,
    refetchOnWindowFocus: false,
    refetchInterval: autoRefresh ? 30000 : false,
    refetchIntervalInBackground: false,
    retry: 2,
    retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 10000),
  });

  useEffect(() => {
    if (!metricsQuery.data) return;
    const data = metricsQuery.data;
    const epochs = data.chart_series?.epochs || [];
    const lastEpoch = epochs.length ? epochs[epochs.length - 1] : null;
    const nextFingerprint = `${source}|${data.rows || 0}|${lastEpoch ?? "na"}`;
    const prevFingerprint = lastDataFingerprintRef.current;
    const changed = prevFingerprint !== nextFingerprint;
    lastDataFingerprintRef.current = nextFingerprint;
    setHasData(epochs.length > 0);
    setOverview(data.overview_stats || {});
    setSummaryMetrics(data.summary_metrics || {});
    const nextSeries = data.chart_series || null;
    rawSeriesRef.current = nextSeries;
    setChartSeriesState(nextSeries);
    if (autoRefresh && !changed) return;
  }, [autoRefresh, metricsQuery.data, source, toast]);

  useEffect(() => {
    if (!metricsQuery.isError) return;
    setHasData(false);
    setChartSeriesState(null);
    toast(metricsQuery.error?.message || "请求失败", "error");
  }, [metricsQuery.error, metricsQuery.isError, toast]);

  const loadMetricsData = useCallback(
    async (sourcePath, silent = false) => {
      if (!sourcePath) return;
      if (sourcePath !== source) {
        setSource(sourcePath);
      }
      setRefreshToken((prev) => prev + 1);
      if (!silent) {
        await Promise.resolve();
      }
    },
    [source],
  );

  const refreshSources = useCallback(
    async (showToast = false) => {
      try {
        const { data } = await sourcesQuery.refetch();
        const available = Array.isArray(data?.csv_metrics)
          ? data.csv_metrics.filter((x) => x.has_results)
          : [];
        setSources(available);
        if (!available.length) {
          setHasData(false);
          setChartSeriesState(null);
          if (showToast) toast("暂无训练结果可展示", "info");
          return;
        }

        const nextSource = available.some((it) => it.path === source)
          ? source
          : available[0]?.path || "";
        if (nextSource && nextSource !== source) {
          setSource(nextSource);
        }
        setRefreshToken((prev) => prev + 1);
        if (showToast) toast("指标来源已刷新", "success");
      } catch (error) {
        toast(error.message, "error");
      }
    },
    [source, sourcesQuery, toast],
  );

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
    loadMetricsData,
  };
}
