import React from "react";
import { Alert, Button, Card, Col, Descriptions, Drawer, Form, Input, InputNumber, message, Modal, Row, Space, Table, Tag, Tree, Upload } from "antd";
import { Database, Eye, RefreshCw, Trash2, UploadCloud } from "lucide-react";
import { api, apiError } from "../api/client";
import { AuthContext } from "../main";
import { CurveChart, DatasetSohChart } from "../components/Chart";
import type { DatasetItem } from "../types";

type TreeNode = { title: string; key: string; children?: TreeNode[] };

function buildTree(items: DatasetItem[]): TreeNode[] {
  const chemistryMap = new Map<string, Map<string, DatasetItem[]>>();
  items.forEach((item) => {
    const chemistry = item.chemistry || "未标注化学成分";
    const dataset = item.dataset_name || "未命名数据集";
    if (!chemistryMap.has(chemistry)) chemistryMap.set(chemistry, new Map());
    const datasetMap = chemistryMap.get(chemistry)!;
    if (!datasetMap.has(dataset)) datasetMap.set(dataset, []);
    datasetMap.get(dataset)!.push(item);
  });

  return Array.from(chemistryMap.entries()).map(([chemistry, datasetMap]) => ({
    title: chemistry,
    key: `chemistry::${chemistry}`,
    children: Array.from(datasetMap.entries()).map(([dataset, cells]) => ({
      title: `${dataset}（${cells.length} 块）`,
      key: `dataset::${chemistry}::${dataset}`,
      children: cells.map((cell) => ({
        title: cell.cell_name || cell.note || `电池 ${cell.id}`,
        key: `cell::${cell.id}`,
      })),
    })),
  }));
}

export default function Datasets() {
  const { auth } = React.useContext(AuthContext);
  const [items, setItems] = React.useState<DatasetItem[]>([]);
  const [keyword, setKeyword] = React.useState("");
  const [selectedKey, setSelectedKey] = React.useState<string>("");
  const [active, setActive] = React.useState<DatasetItem | null>(null);
  const [importOpen, setImportOpen] = React.useState(false);
  const [fileList, setFileList] = React.useState<any[]>([]);
  const [importing, setImporting] = React.useState(false);
  const [form] = Form.useForm();
  const isTeacher = auth.role === "teacher";

  async function load() {
    setItems((await api.get("/datasets")).data);
  }

  React.useEffect(() => {
    load().catch((error) => message.error(apiError(error)));
  }, []);

  const filtered = items.filter((item) => {
    const haystack = JSON.stringify(item);
    if (keyword && !haystack.includes(keyword)) return false;
    if (!selectedKey) return true;
    if (selectedKey.startsWith("chemistry::")) return item.chemistry === selectedKey.split("::")[1];
    if (selectedKey.startsWith("dataset::")) {
      const [, chemistry, dataset] = selectedKey.split("::");
      return item.chemistry === chemistry && item.dataset_name === dataset;
    }
    if (selectedKey.startsWith("cell::")) return item.id === Number(selectedKey.split("::")[1]);
    return true;
  });

  const datasetCount = new Set(items.map((item) => `${item.chemistry || ""}/${item.dataset_name || ""}`)).size;
  const chemistryCount = new Set(items.map((item) => item.chemistry || "未标注化学成分")).size;
  const eligibleCount = items.filter((item) => Number(item.training_eligible ?? 1) === 1).length;
  const filteredEligibleCount = filtered.filter((item) => Number(item.training_eligible ?? 1) === 1).length;
  const filteredExcludedCount = filtered.length - filteredEligibleCount;
  const filteredLives = filtered.map((item) => Number(item.cycle_life)).filter(Number.isFinite);
  const lifeRange = filteredLives.length ? `${Math.min(...filteredLives)} - ${Math.max(...filteredLives)} 圈` : "-";

  async function importRealDataset() {
    const hide = message.loading("正在导入真实 Excel 数据集并重新训练模型，这可能需要几分钟...", 0);
    try {
      const { data } = await api.post("/datasets/import-real");
      hide();
      message.success(`已导入 ${data.imported_count} 个电池文件，并完成模型训练。`);
      load();
    } catch (error) {
      hide();
      message.error(apiError(error));
    }
  }

  async function importCsv(values: any) {
    const file = fileList[0]?.originFileObj;
    if (!file) return message.warning("请先选择 CSV 文件。");
    const body = new FormData();
    body.append("file", file);
    Object.entries(values).forEach(([key, value]) => body.append(key, String(value)));
    setImporting(true);
    try {
      await api.post("/datasets/import", body, { headers: { "Content-Type": "multipart/form-data" } });
      message.success("数据集导入成功。");
      setImportOpen(false);
      setFileList([]);
      form.resetFields();
      load();
    } catch (error) {
      message.error(apiError(error));
    } finally {
      setImporting(false);
    }
  }

  async function reviewActive(training_eligible: number) {
    if (!active) return;
    const next = {
      training_eligible,
      label_status: training_eligible ? "可靠EOL" : "教师剔除",
    };
    await api.put(`/datasets/${active.id}`, next);
    message.success(training_eligible ? "已标记为可靠样本。" : "已剔除出训练集。");
    setActive({ ...active, ...next });
    load();
  }

  return (
    <div className="pageStack">
      <section className="pageHero">
        <div>
          <span className="pageKicker"><Database size={15} /> Dataset Management</span>
          <h1>数据集管理</h1>
          <p>按照「化学成分 → 数据集 → 单个电池」组织真实实验数据，统一管理导入、查看、筛选和曲线复核。</p>
        </div>
        <div className="heroStats">
          <div><strong>{chemistryCount}</strong><span>化学成分</span></div>
          <div><strong>{datasetCount}</strong><span>数据集</span></div>
          <div><strong>{eligibleCount}/{items.length}</strong><span>可靠训练样本</span></div>
        </div>
      </section>

      <Card
        title="全数据集衰减曲线"
        className="controlPanel"
        extra={
          <Space>
            <Tag color="blue">当前 {filtered.length} 块</Tag>
            <Tag color="green">可靠EOL {filteredEligibleCount}</Tag>
            <Tag color="orange">仅入库 {filteredExcludedCount}</Tag>
            <Tag>寿命范围 {lifeRange}</Tag>
          </Space>
        }
      >
        <DatasetSohChart items={filtered} height={500} />
      </Card>

      <Row gutter={[18, 18]}>
        <Col xs={24} xl={7}>
          <Card title="层级分组" className="controlPanel">
            <Tree
              treeData={buildTree(items)}
              defaultExpandAll
              selectedKeys={selectedKey ? [selectedKey] : []}
              onSelect={(keys) => setSelectedKey(String(keys[0] || ""))}
            />
            <Button className="full clearTreeButton" onClick={() => setSelectedKey("")}>查看全部电池</Button>
          </Card>
        </Col>
        <Col xs={24} xl={17}>
          <Card
            title="电池数据"
            extra={
              <Space>
                <Input.Search placeholder="搜索组别、数据集、文件名" allowClear onSearch={setKeyword} onChange={(e) => setKeyword(e.target.value)} />
                <Button icon={<RefreshCw size={16} />} onClick={load}>刷新</Button>
                {isTeacher && <Button icon={<UploadCloud size={16} />} onClick={() => setImportOpen(true)}>导入时序 CSV</Button>}
                {isTeacher && <Button type="primary" icon={<UploadCloud size={16} />} onClick={importRealDataset}>导入真实数据集并训练</Button>}
              </Space>
            }
          >
            <Table
              className="nowrapTable"
              rowKey="id"
              dataSource={filtered}
              scroll={{ x: 1280 }}
              columns={[
                { title: "化学成分", dataIndex: "chemistry", render: (v) => <Tag color="cyan">{v || "未标注"}</Tag> },
                { title: "数据集", dataIndex: "dataset_name" },
                { title: "单体电池", dataIndex: "cell_name", render: (v, r) => v || r.note },
                { title: "组别", dataIndex: "battery_type", render: (v) => <Tag color="geekblue">{v}</Tag> },
                {
                  title: "标签质量",
                  dataIndex: "label_status",
                  render: (v) => {
                    const status = v || "未评估";
                    const color = status === "可靠EOL" ? "green" : status === "未达到EOL" ? "gold" : "orange";
                    return <Tag color={color}>{status}</Tag>;
                  },
                },
                {
                  title: "训练用途",
                  dataIndex: "training_eligible",
                  render: (v) => Number(v ?? 1) === 1 ? <Tag color="green">参与训练</Tag> : <Tag color="default">仅入库</Tag>,
                },
                { title: "循环寿命", dataIndex: "cycle_life", sorter: (a, b) => a.cycle_life - b.cycle_life },
                { title: "当前 SOH", dataIndex: "current_soh", render: (v) => `${Number(v).toFixed(1)}%` },
                { title: "容量", dataIndex: "rated_capacity", render: (v) => `${Number(v).toFixed(3)} mAh` },
                {
                  title: "操作",
                  render: (_, record) => (
                    <Space>
                      <Button icon={<Eye size={16} />} onClick={() => setActive(record)}>曲线</Button>
                      {isTeacher && (
                        <Button
                          danger
                          icon={<Trash2 size={16} />}
                          onClick={() => Modal.confirm({ title: "确认删除该电池数据？", onOk: async () => { await api.delete(`/datasets/${record.id}`); load(); } })}
                        >
                          删除
                        </Button>
                      )}
                    </Space>
                  ),
                },
              ]}
            />
          </Card>
        </Col>
      </Row>

      <Drawer title={active?.cell_name || active?.note || "电池曲线"} open={!!active} onClose={() => setActive(null)} width={820}>
        {active && (
          <Space direction="vertical" size={16} className="full">
            {isTeacher && (
              <Space>
                <Button type="primary" onClick={() => reviewActive(1)}>标记可靠</Button>
                <Button danger onClick={() => Modal.confirm({ title: "确认剔除该样本？", content: "剔除后该电池仍保留在数据库，但不会参与默认训练。", onOk: () => reviewActive(0) })}>剔除样本</Button>
              </Space>
            )}
            <Descriptions bordered size="small" column={2}>
              <Descriptions.Item label="标签质量">{active.label_status || "未评估"}</Descriptions.Item>
              <Descriptions.Item label="训练用途">{Number(active.training_eligible ?? 1) === 1 ? "参与训练" : "仅入库，不参与默认训练"}</Descriptions.Item>
              <Descriptions.Item label="寿命/截止循环">{active.cycle_life}</Descriptions.Item>
              <Descriptions.Item label="容量基准">{Number(active.capacity_baseline || active.rated_capacity).toFixed(3)} mAh</Descriptions.Item>
            </Descriptions>
            {!!active.quality_flags?.length && <Alert type="warning" showIcon message="质量提示" description={active.quality_flags.join("；")} />}
            <CurveChart predicted={active.capacity_curve} height={500} />
          </Space>
        )}
      </Drawer>

      <Modal title="导入长表时序 CSV" open={importOpen} onCancel={() => setImportOpen(false)} footer={null}>
        <Form form={form} layout="vertical" onFinish={importCsv}>
          <Form.Item label="化学体系" name="chemistry" rules={[{ required: true, message: "请输入或选择化学体系" }]}>
            <Input placeholder="例如 NCM / LFP / 实验组锂离子电池" />
          </Form.Item>
          <Form.Item label="数据集名称" name="dataset_name" rules={[{ required: true, message: "请输入数据集名称" }]}>
            <Input placeholder="例如 Severson 早期循环数据集" />
          </Form.Item>
          <Form.Item label="组别/电池类型" name="battery_type" initialValue="G1">
            <Input placeholder="例如 G1 / LFP" />
          </Form.Item>
          <Row gutter={12}>
            <Col span={12}><Form.Item label="额定容量 (Ah)" name="rated_capacity" rules={[{ required: true }]}><InputNumber className="full" min={0.001} max={10000} step={0.001} /></Form.Item></Col>
            <Col span={12}><Form.Item label="倍率 (C)" name="c_rate" initialValue={1}><InputNumber className="full" min={0.1} max={5} step={0.1} /></Form.Item></Col>
          </Row>
          <Form.Item label="CSV 文件" required>
            <Upload.Dragger accept=".csv" maxCount={1} beforeUpload={() => false} fileList={fileList} onChange={({ fileList }) => setFileList(fileList)}>
              <p>上传包含 cycle, time_s, voltage_V, current_A 的长表 CSV</p>
            </Upload.Dragger>
          </Form.Item>
          <Button type="primary" htmlType="submit" loading={importing} block>开始导入</Button>
        </Form>
      </Modal>
    </div>
  );
}
