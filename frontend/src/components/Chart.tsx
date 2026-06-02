import ReactECharts from "echarts-for-react";
import type { CurvePoint, DatasetItem } from "../types";

interface Props {
  input?: CurvePoint[];
  predicted?: CurvePoint[];
  matches?: DatasetItem[];
  height?: number;
}

export function CurveChart({ input = [], predicted = [], matches = [], height = 360 }: Props) {
  const series = [
    input.length
      ? {
          name: "输入曲线",
          type: "line",
          smooth: true,
          symbol: "none",
          data: input.map((p) => [p.cycle, p.specific_capacity]),
          lineStyle: { width: 3, color: "#0f766e" },
        }
      : null,
    predicted.length
      ? {
          name: "最佳匹配",
          type: "line",
          smooth: true,
          symbol: "none",
          data: predicted.map((p) => [p.cycle, p.specific_capacity]),
          lineStyle: { width: 3, color: "#2563eb" },
        }
      : null,
    ...matches.map((item) => ({
      name: `Top ${item.id}`,
      type: "line",
      smooth: true,
      symbol: "none",
      data: item.capacity_curve.map((p) => [p.cycle, p.specific_capacity]),
      lineStyle: { width: 1.5, type: "dashed" },
    })),
  ].filter(Boolean);

  return (
    <ReactECharts
      style={{ height }}
      option={{
        color: ["#0f766e", "#2563eb", "#f59e0b", "#7c3aed", "#dc2626"],
        tooltip: { trigger: "axis" },
        toolbox: { feature: { saveAsImage: { title: "导出图片" }, dataZoom: { title: { zoom: "缩放", back: "还原" } }, restore: { title: "还原" } } },
        legend: { top: 0 },
        grid: { top: 56, left: 62, right: 32, bottom: 48 },
        xAxis: { type: "value", name: "循环次数", nameLocation: "middle", nameGap: 30 },
        yAxis: { type: "value", name: "比容量 (mAh·g⁻¹)", nameLocation: "middle", nameGap: 48 },
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
          feature: {
            saveAsImage: { title: "导出图片" },
            dataZoom: { title: { zoom: "缩放", back: "还原" } },
            restore: { title: "还原" },
          },
        },
        legend: {
          type: "scroll",
          top: 0,
          pageIconColor: "#0f766e",
          pageTextStyle: { color: "#475569" },
        },
        grid: { top: 72, left: 62, right: 34, bottom: 54 },
        dataZoom: [
          { type: "inside", xAxisIndex: 0 },
          { type: "slider", xAxisIndex: 0, height: 18, bottom: 12 },
        ],
        xAxis: { type: "value", name: "循环次数", nameLocation: "middle", nameGap: 34 },
        yAxis: {
          type: "value",
          name: "SOH (%)",
          min: (value: { min: number }) => Math.max(Math.floor(value.min - 3), 0),
          max: (value: { max: number }) => Math.ceil(value.max + 3),
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
