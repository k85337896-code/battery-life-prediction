import React from "react";
import { Button, Card, Form, Input, InputNumber, message, Modal, Select, Space, Table, Upload } from "antd";
import { Plus, RefreshCw, Trash2, UploadCloud } from "lucide-react";
import type { UploadFile } from "antd";
import { api, apiError } from "../api/client";
import type { DatasetItem } from "../types";

function demoCurve(rated: number, life: number) {
  return Array.from({ length: 36 }, (_, index) => {
    const cycle = Math.round((life / 35) * index);
    const soh = 100 - 20 * Math.pow(cycle / life, 1.18);
    return { cycle, soh, specific_capacity: Number((rated * soh / 100).toFixed(4)) };
  });
}

export default function DatasetManage() {
  const [items, setItems] = React.useState<DatasetItem[]>([]);
  const [modalOpen, setModalOpen] = React.useState(false);
  const [fileList, setFileList] = React.useState<UploadFile[]>([]);
  const [form] = Form.useForm();

  async function load() {
    setItems((await api.get("/datasets")).data);
  }
  React.useEffect(() => { load().catch((e) => message.error(apiError(e))); }, []);

  async function create(values: any) {
    try {
      await api.post("/datasets", { ...values, capacity_curve: demoCurve(values.rated_capacity, values.cycle_life), current_soh: 80, source: "教师录入" });
      setModalOpen(false);
      form.resetFields();
      load();
    } catch (error) {
      message.error(apiError(error));
    }
  }

  async function importCsv(values: any) {
    const file = fileList[0]?.originFileObj;
    if (!file) return message.warning("请先选择 CSV。");
    const body = new FormData();
    body.append("file", file);
    Object.entries(values).forEach(([key, value]) => body.append(key, String(value)));
    await api.post("/datasets/import", body);
    setFileList([]);
    load();
    message.success("导入完成。");
  }

  return (
    <div className="pageStack">
      <Card title="数据库管理" extra={<Space><Button icon={<RefreshCw size={16} />} onClick={load}>刷新</Button><Button type="primary" icon={<Plus size={16} />} onClick={() => setModalOpen(true)}>新增条目</Button></Space>}>
        <Table rowKey="id" dataSource={items} columns={[
          { title: "ID", dataIndex: "id" },
          { title: "类型", dataIndex: "battery_type" },
          { title: "额定容量", dataIndex: "rated_capacity" },
          { title: "倍率", dataIndex: "c_rate" },
          { title: "寿命", dataIndex: "cycle_life" },
          { title: "备注", dataIndex: "note" },
          { title: "操作", render: (_, record) => <Button danger icon={<Trash2 size={16} />} onClick={() => Modal.confirm({ title: "确认删除该数据库条目？", onOk: async () => { await api.delete(`/datasets/${record.id}`); load(); } })}>删除</Button> },
        ]} />
      </Card>
      <Card title="CSV 批量导入">
        <Form layout="inline" initialValues={{ battery_type: "G1", theoretical_capacity: 4, rated_capacity: 4, c_rate: 1 }} onFinish={importCsv}>
          <Form.Item name="battery_type"><Select style={{ width: 150 }} options={[{ value: "G1" }, { value: "G2" }, { value: "G3" }, { value: "G4" }, { value: "LCO" }, { value: "LFP" }, { value: "LS" }]} /></Form.Item>
          <Form.Item name="theoretical_capacity"><InputNumber placeholder="理论容量" /></Form.Item>
          <Form.Item name="rated_capacity"><InputNumber placeholder="额定容量" /></Form.Item>
          <Form.Item name="c_rate"><InputNumber placeholder="倍率" /></Form.Item>
          <Upload beforeUpload={() => false} maxCount={1} fileList={fileList} onChange={({ fileList }) => setFileList(fileList)}><Button icon={<UploadCloud size={16} />}>选择 CSV</Button></Upload>
          <Button type="primary" htmlType="submit">导入</Button>
        </Form>
      </Card>
      <Card title="真实 Excel 数据集">
        <Space>
          <Button
            type="primary"
            onClick={async () => {
              const hide = message.loading("正在导入真实 Excel 数据集并训练模型，这可能需要几分钟...", 0);
              try {
                const { data } = await api.post("/datasets/import-real");
                hide();
                message.success(`已导入 ${data.imported_count} 个电池文件，并完成模型训练。`);
                load();
              } catch (error) {
                hide();
                message.error(apiError(error));
              }
            }}
          >
            导入真实数据集并训练模型
          </Button>
        </Space>
      </Card>
      <Modal title="新增数据库条目" open={modalOpen} onCancel={() => setModalOpen(false)} footer={null}>
        <Form form={form} layout="vertical" initialValues={{ battery_type: "G1", c_rate: 1, cycle_life: 500, rated_capacity: 4, theoretical_capacity: 4 }} onFinish={create}>
          <Form.Item label="电池类型" name="battery_type"><Select options={[{ value: "G1" }, { value: "G2" }, { value: "G3" }, { value: "G4" }, { value: "LCO" }, { value: "LFP" }, { value: "LS" }]} /></Form.Item>
          <Form.Item label="理论容量" name="theoretical_capacity"><InputNumber className="full" /></Form.Item>
          <Form.Item label="额定容量" name="rated_capacity"><InputNumber className="full" /></Form.Item>
          <Form.Item label="倍率" name="c_rate"><InputNumber className="full" /></Form.Item>
          <Form.Item label="循环寿命" name="cycle_life"><InputNumber className="full" /></Form.Item>
          <Form.Item label="备注" name="note"><Input /></Form.Item>
          <Button type="primary" htmlType="submit">保存</Button>
        </Form>
      </Modal>
    </div>
  );
}
