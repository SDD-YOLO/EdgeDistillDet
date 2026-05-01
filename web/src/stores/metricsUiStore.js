import { create } from "zustand";

const DEFAULT_RANGE = "all";

export const useMetricsUiStore = create((set) => ({
  chartType: "all",
  rangeLoss: DEFAULT_RANGE,
  rangeMap: DEFAULT_RANGE,
  rangeLr: DEFAULT_RANGE,
  rangeDistill: DEFAULT_RANGE,
  autoRefresh: false,
  refreshLeft: 30,
  setChartType: (chartType) => set({ chartType }),
  setRangeLoss: (rangeLoss) => set({ rangeLoss }),
  setRangeMap: (rangeMap) => set({ rangeMap }),
  setRangeLr: (rangeLr) => set({ rangeLr }),
  setRangeDistill: (rangeDistill) => set({ rangeDistill }),
  setAutoRefresh: (autoRefresh) => set({ autoRefresh, refreshLeft: 30 }),
  setRefreshLeft: (refreshLeft) => set({ refreshLeft })
}));