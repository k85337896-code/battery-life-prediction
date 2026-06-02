import React from "react";
import { Button, Card, Drawer, Input, message, Modal, Space, Table, Tag } from "antd";
import { Eye, Trash2 } from "lucide-react";
import { api, apiError } from "../api/client";
import { CurveChart } from "../components/Chart";
import type { HistoryItem } from "../types";

export default function History() {
  const [items, setItems] = React.useState<HistoryItem[]>([]);
  const [keyword, setKeyword] = React.useState("");
  const [active, setActive] = React.useState<HistoryItem | null>(null);

  async function load() {
    try {
      setItems((await api.get("/history")).data);
    } catch (error) {
      message.error(apiError(error));
    }
  }
  React.useEffect(() => { load(); }, []);

  const filtered = items.filter((item) => JSON.stringify(item).includes(keyword));
  return (
    <Card title="历史记录" extra={<Input.Search placeholder="搜索类型、时间或数值" onSearch={setKeyword} allowClear onChange={(e) => setKeyword(e.target.value)} />}>
      <Table rowKey="id" dataSource={filtered} columns={[
        { title: "时间", dataIndex: "predict_time" },
        { title: "类型", dataIndex: "battery_type", render: (v) => <Tag color="cyan">{v}</Tag> },
        { title: "额定容量", dataIndex: "rated_capacity" },
        { title: "剩余寿命", dataIndex: "predicted_remaining_life", sorter: (a, b) => a.predicted_remaining_life - b.predicted_remaining_life },
        { title: "SOH", dataIndex: "soh_at_prediction", render: (v) => `${v.toFixed(1)}%`, sorter: (a, b) => a.soh_at_prediction - b.soh_at_prediction },
        { title: "相关性", dataIndex: "correlation_score" },
        { title: "操作", render: (_, record) => <Space><Button icon={<Eye size={16} />} onClick={() => setActive(record)} /><Button danger icon={<Trash2 size={16} />} onClick={() => Modal.confirm({ title: "确认删除该历史记录？", onOk: async () => { await api.delete(`/history/${record.id}`); load(); } })} /></Space> },
      ]} />
      <Drawer title="历史曲线详情" open={!!active} onClose={() => setActive(null)} width={760}>
        {active && <CurveChart input={active.input_curve} predicted={active.predicted_curve} height={480} />}
      </Drawer>
    </Card>
  );
}
