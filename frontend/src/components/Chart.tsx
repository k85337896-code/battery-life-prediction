import ReactECharts from "echarts-for-react";
import type { CurvePoint, DatasetItem } from "../types";

type MatchItem = DatasetItem & { cell_id?: number; curve?: CurvePoint[]; similarity?: number; correlation_score?: number };

interface Props {
  input?: CurvePoint[];
  predicted?: CurvePoint[];
  matches?: MatchItem[];
  sohCurve?: CurvePoint[];
  eol?: number;
  height?: number;
}

export function CurveChart({ input = [], predicted = [], matches = [], sohCurve, eol = 80, height = 360 }: Props) {
  const curve = sohCurve?.length ? sohCurve : predicted;
  const currentCycle = input.length ? Math.max(...input.map((p) => p.cycle)) : 0;
  const series = [
    input.length
      ? {
          name: "实测 SOH",
          type: "line",
          smooth: true,
          symbol: "none",
          data: input.map((p) => [p.cycle, p.soh]),
          lineStyle: { width: 3, color: "#0f766e" },
        }
      : null,
    curve.length
      ? {
          name: "预测 SOH",
          type: "line",
          smooth: true,
          symbol: "none",
          data: curve.filter((p) => p.cycle >= currentCycle).map((p) => [p.cycle, p.soh]),
          lineStyle: { width: 3, color: "#2563eb", type: "dashed" },
          markLine: {
            symbol: "none",
            data: [{ yAxis: eol, name: "EOL 80%" }],
            label: { formatter: "EOL 80%" },
            lineStyle: { color: "#ef4444", type: "dashed" },
          },
        }
      : null,
    curve.some((p) => p.lower !== undefined && p.upper !== undefined)
      ? {
          name: "置信下界",
          type: "line",
          smooth: true,
          symbol: "none",
          data: curve.map((p) => [p.cycle, p.lower ?? p.soh]),
          lineStyle: { opacity: 0 },
          stack: "confidence",
        }
      : null,
    curve.some((p) => p.lower !== undefined && p.upper !== undefined)
      ? {
          name: "置信区间",
          type: "line",
          smooth: true,
          symbol: "none",
          data: curve.map((p) => [p.cycle, Math.max((p.upper ?? p.soh) - (p.lower ?? p.soh), 0)]),
          lineStyle: { opacity: 0 },
          areaStyle: { color: "rgba(37, 99, 235, 0.16)" },
          stack: "confidence",
        }
      : null,
    ...matches.map((item) => ({
      name: item.cell_name || `Top ${item.cell_id || item.id}`,
      type: "line",
      smooth: true,
      symbol: "none",
      data: (item.curve || item.capacity_curve || []).map((p: CurvePoint) => [p.cycle, p.soh]),
      lineStyle: { width: 1.2, opacity: 0.36 },
    })),
  ].filter(Boolean);

  return (
    <ReactECharts
      style={{ height }}
      option={{
        color: ["#0f766e", "#2563eb", "#f59e0b", "#7c3aed", "#dc2626"],
        tooltip: { trigger: "axis" },
        toolbox: { top: 0, right: 8, feature: { saveAsImage: { title: "导出图片" }, dataZoom: { title: { zoom: "缩放", back: "还原" } }, restore: { title: "还原" } } },
        legend: { type: "scroll", top: 34, left: 8, right: 8 },
        grid: { top: 82, left: 62, right: 32, bottom: 70 },
        dataZoom: [
          { type: "inside", xAxisIndex: 0 },
          { type: "slider", xAxisIndex: 0, height: 18, bottom: 18 },
        ],
        xAxis: { type: "value", min: 0, name: "循环次数", nameLocation: "middle", nameGap: 46 },
        yAxis: { type: "value", name: "SOH (%)", min: 50, max: 110, nameLocation: "middle", nameGap: 48 },
        series,
      }}
    />
  );
}

interface DatasetSohChartProps {
  items: DatasetItem[];
  height?: number;
}

export function DatasetSohChart({ items, height = 460 }: DatasetSohChartProps) {
  const sortedItems = [...items].sort((a, b) => {
    const group = String(a.battery_type).localeCompare(String(b.battery_type));
    if (group !== 0) return group;
    return (a.cell_name || "").localeCompare(b.cell_name || "");
  });

  return (
    <ReactECharts
      style={{ height }}
      option={{
        title: sortedItems.length ? undefined : { text: "暂无曲线数据", left: "center", top: "middle", textStyle: { color: "#94a3b8", fontSize: 14 } },
        color: ["#0f766e", "#2563eb", "#f59e0b", "#7c3aed", "#dc2626", "#0891b2", "#65a30d", "#be123c"],
        tooltip: {
          trigger: "axis",
          valueFormatter: (value: number) => `${Number(value).toFixed(2)}%`,
        },
        toolbox: {
          top: 0,
          right: 8,
          feature: {
            saveAsImage: { title: "导出图片" },
            dataZoom: { title: { zoom: "缩放", back: "还原" } },
            restore: { title: "还原" },
          },
        },
        legend: {
          type: "scroll",
          top: 36,
          left: 8,
          right: 8,
          pageIconColor: "#0f766e",
          pageTextStyle: { color: "#475569" },
        },
        grid: { top: 92, left: 62, right: 34, bottom: 76 },
        dataZoom: [
          { type: "inside", xAxisIndex: 0 },
          { type: "slider", xAxisIndex: 0, height: 18, bottom: 18 },
        ],
        xAxis: { type: "value", min: 0, name: "循环次数", nameLocation: "middle", nameGap: 48 },
        yAxis: {
          type: "value",
          name: "SOH (%)",
          min: (value: { min: number }) => Math.max(Math.floor((value.min - 3) / 10) * 10, 0),
          max: (value: { max: number }) => Math.ceil((value.max + 3) / 10) * 10,
          nameLocation: "middle",
          nameGap: 44,
        },
        series: sortedItems.map((item) => ({
          name: item.cell_name || `${item.battery_type}-${item.id}`,
          type: "line",
          smooth: true,
          symbol: "none",
          data: item.capacity_curve.map((point) => [point.cycle, point.soh]),
          lineStyle: {
            width: Number(item.training_eligible ?? 1) === 1 ? 2 : 1.4,
            type: Number(item.training_eligible ?? 1) === 1 ? "solid" : "dashed",
            opacity: Number(item.training_eligible ?? 1) === 1 ? 0.95 : 0.55,
          },
          emphasis: { focus: "series" },
        })),
      }}
    />
  );
}
