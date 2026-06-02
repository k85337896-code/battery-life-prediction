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
