import { Progress, Tag } from "antd";

export function SohProgress({ value }: { value: number }) {
  const color = value >= 90 ? "#16a34a" : value >= 80 ? "#0891b2" : "#dc2626";
  const label = value >= 90 ? "健康" : value >= 80 ? "接近 EOL" : "电池损坏 / 低于 EOL";
  return (
    <div className="sohBox">
      <Progress percent={Number(value.toFixed(1))} strokeColor={color} />
      <Tag color={color}>80% 为 EOL 参考线 · {label}</Tag>
    </div>
  );
}
