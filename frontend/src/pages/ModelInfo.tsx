import React from "react";
import { Alert, Button, Card, Col, Descriptions, Empty, Form, InputNumber, message, Modal, Progress, Row, Select, Space, Statistic, Table, Tabs, Tag } from "antd";
import { RefreshCw } from "lucide-react";
import { api, apiError } from "../api/client";
import { AuthContext } from "../main";

const modelOptions = [
  { value: "xgboost", label: "XGBoost 回归模型" },
  { value: "lstm", label: "LSTM 循环神经网络" },
  { value: "tcn", label: "TCN 时序卷积网络" },
  { value: "cnn", label: "CNN 卷积神经网络" },
  { value: "gpr", label: "GPR 高斯过程回归" },
  { value: "all", label: "全部模型" },
];

const metricValue = (value: unknown, suffix = "") => value === undefined || value === null ? "-" : `${value}${suffix}`;

export default function ModelInfo() {
  const { auth } = React.useContext(AuthContext);
  const [models, setModels] = React.useState<any[]>([]);
  const [source, setSource] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [job, setJob] = React.useState<any>(null);
  const [datasets, setDatasets] = React.useState<any[]>([]);
  const [form] = Form.useForm();
  const isTeacher = auth.role === "teacher";
  const selectedChemistry = Form.useWatch("chemistry", form);

  async function loadModels() {
    setModels((await api.get("/model-info")).data);
  }

  React.useEffect(() => {
    loadModels().catch((error) => message.error(apiError(error)));
    if (isTeacher) {
      api.get("/datasets").then((res) => setDatasets(res.data)).catch(() => undefined);
      api.get("/source", { responseType: "text" }).then((res) => setSource(res.data)).catch(() => undefined);
    }
  }, []);

  async function train(values: any) {
    setLoading(true);
    try {
      const { data } = await api.post("/model/train", values);
      setJob(data);
      message.success("训练任务已提交，正在后台训练。");
      pollJob(data.job_id);
    } catch (error) {
      message.error(apiError(error));
    } finally {
      setLoading(false);
    }
  }

  async function pollJob(jobId: string) {
    const timer = window.setInterval(async () => {
      try {
        const { data } = await api.get(`/model/train/${jobId}`);
        setJob(data);
        if (["succeeded", "failed"].includes(data.status)) {
          window.clearInterval(timer);
          if (data.status === "succeeded") {
            message.success("模型训练完成。");
            loadModels();
          } else {
            message.error(data.error || "训练失败。");
          }
        }
      } catch (error) {
        window.clearInterval(timer);
        message.error(apiError(error));
      }
    }, 1800);
  }

  async function setPublish(modelKey: string, status: "published" | "archived") {
    await api.post(`/model-info/${modelKey}/publish`, { status });
    message.success(status === "published" ? "模型已发布。" : "模型已下架。");
    loadModels();
  }

  const best = [...models].sort((a, b) => (b.metrics?.R2 ?? -999) - (a.metrics?.R2 ?? -999))[0] || {};
  const bestMetrics = best.metrics || {};
  const chemistries = Array.from(new Set(datasets.map((item) => item.chemistry).filter(Boolean)));
  const datasetOptions = Array.from(new Set(datasets.filter((item) => !selectedChemistry || item.chemistry === selectedChemistry).map((item) => item.dataset_name).filter(Boolean))).map((value) => ({ value, label: value }));

  return (
    <div className="pageStack">
      <section className="pageHero">
        <div>
          <span className="pageKicker"><RefreshCw size={15} /> Model Center</span>
          <h1>模型中心</h1>
          <p>查看完整寿命曲线训练后的早期预测精度；教师端可调整参数并重新训练。</p>
        </div>
        <div className="heroStats">
          <div><strong>{models.length}</strong><span>已训练模型</span></div>
          <div><strong>{bestMetrics["可靠EOL样本"] || best.training_data_size || 0}</strong><span>可靠训练样本</span></div>
          <div><strong>{bestMetrics.R2 ?? "-"}</strong><span>最佳 R²</span></div>
        </div>
      </section>

      {isTeacher && (
        <Card title="训练参数编辑" className="controlPanel">
          <Form form={form} layout="inline" initialValues={{ model_key: "xgboost", n_estimators: 80, max_depth: 2, learning_rate: 0.05, training_observation_fraction: 0.1, publish: true }} onFinish={train}>
            <Form.Item label="化学体系" name="chemistry" rules={[{ required: true, message: "请选择化学体系" }]}><Select style={{ width: 190 }} options={chemistries.map((value) => ({ value, label: value }))} /></Form.Item>
            <Form.Item label="数据来源" name="dataset_ids" rules={[{ required: true, message: "请选择训练数据集" }]}><Select mode="multiple" maxTagCount="responsive" style={{ width: 260 }} options={datasetOptions} /></Form.Item>
            <Form.Item label="模型" name="model_key"><Select style={{ width: 210 }} options={modelOptions} /></Form.Item>
            <Form.Item label="树数量" name="n_estimators"><InputNumber min={20} max={500} /></Form.Item>
            <Form.Item label="最大深度" name="max_depth"><InputNumber min={2} max={10} /></Form.Item>
            <Form.Item label="学习率" name="learning_rate"><InputNumber min={0.001} max={0.5} step={0.001} /></Form.Item>
            <Form.Item label="评估前缀" name="training_observation_fraction"><InputNumber min={0.05} max={0.5} step={0.05} /></Form.Item>
            <Button type="primary" htmlType="submit" icon={<RefreshCw size={16} />} loading={loading}>重新训练</Button>
          </Form>
          {job && <Progress percent={Number(job.progress || 0)} status={job.status === "failed" ? "exception" : job.status === "succeeded" ? "success" : "active"} format={() => job.message || job.status} />}
        </Card>
      )}

      <Alert
        type="info"
        showIcon
        message="早期寿命预测难度说明"
        description="当前评估是在整块电池留出的条件下，仅用前 10%/20%/30% 循环预测完整寿命。页面默认展示百分比误差，MAPE 表示平均绝对百分比误差，NRMSE 表示归一化均方根误差。早期 SOH 差异很小，误差偏大是任务本身的信息不足，不代表系统故障。"
      />

      <Row gutter={16}>
        <Col span={8}><Card><Statistic title="最佳模型" value={best.model_type || "-"} /></Card></Col>
        <Col span={8}><Card><Statistic title="MAPE（百分比）" value={bestMetrics.MAPE || 0} precision={2} suffix="%" /></Card></Col>
        <Col span={8}><Card><Statistic title="NRMSE（百分比）" value={bestMetrics.NRMSE || 0} precision={2} suffix="%" /></Card></Col>
      </Row>

      <Card title="已训练模型">
        <Table
          rowKey="model_key"
          dataSource={models}
          columns={[
            { title: "模型", dataIndex: "model_type" },
            { title: "版本", render: (_, record) => <Tag>v{record.version || 1}</Tag> },
            { title: "化学体系", dataIndex: "chemistry" },
            { title: "状态", render: (_, record) => <Tag color={record.status === "published" ? "green" : "default"}>{record.status === "published" ? "已发布" : "未发布"}</Tag> },
            { title: "训练规模", dataIndex: "training_data_size" },
            { title: "候选样本", render: (_, record) => <Tag>{record.metrics?.["候选样本"] ?? record.training_data_size}</Tag> },
            { title: "排除样本", render: (_, record) => <Tag color={(record.metrics?.["排除样本"] ?? 0) ? "orange" : "green"}>{record.metrics?.["排除样本"] ?? 0}</Tag> },
            { title: "MAPE", render: (_, record) => <Tag color="cyan">{metricValue(record.metrics?.MAPE, "%")}</Tag> },
            { title: "NRMSE", render: (_, record) => <Tag color="purple">{metricValue(record.metrics?.NRMSE, "%")}</Tag> },
            { title: "R²", render: (_, record) => <Tag color={(record.metrics?.R2 ?? 0) >= 0 ? "green" : "orange"}>{record.metrics?.R2}</Tag> },
            {
              title: "窗口误差",
              render: (_, record) => (
                <Space>
                  {["前10%", "前20%", "前30%"].map((key) => (
                    <Tag key={key}>
                      {key} MAPE {metricValue(record.metrics?.["窗口评估"]?.[key]?.MAPE, "%")}
                    </Tag>
                  ))}
                </Space>
              ),
            },
            { title: "评估方式", render: (_, record) => <Tag color="blue">{record.metrics?.["评估方式"] || "-"}</Tag> },
            { title: "训练时间", dataIndex: "trained_at" },
            ...(isTeacher ? [{
              title: "操作",
              render: (_: unknown, record: any) => (
                <Space>
                  {record.status === "published"
                    ? <Button danger onClick={() => Modal.confirm({ title: "确认下架该模型？", onOk: () => setPublish(record.model_key, "archived") })}>下架</Button>
                    : <Button type="primary" onClick={() => setPublish(record.model_key, "published")}>发布</Button>}
                </Space>
              ),
            }] : []),
          ]}
          scroll={{ x: 1360 }}
          className="nowrapTable"
        />
      </Card>

      <Card>
        <Tabs
          items={[
            ...models.map((info) => ({
              key: info.model_key,
              label: info.model_type,
              children: (
                <Descriptions bordered column={1}>
                  <Descriptions.Item label="训练样本">{info.training_data_size} 条电池记录</Descriptions.Item>
                  <Descriptions.Item label="样本筛选">{info.metrics?.["训练样本筛选"] || "未记录"}</Descriptions.Item>
                  <Descriptions.Item label="早期预测窗口">{info.metrics?.["观测窗口"] || "未记录"}</Descriptions.Item>
                  <Descriptions.Item label="窗口评估">
                    <Space>
                      {["前10%", "前20%", "前30%"].map((key) => (
                        <Tag key={key}>
                          {key}: MAPE {metricValue(info.metrics?.["窗口评估"]?.[key]?.MAPE, "%")} / NRMSE {metricValue(info.metrics?.["窗口评估"]?.[key]?.NRMSE, "%")}
                        </Tag>
                      ))}
                    </Space>
                  </Descriptions.Item>
                  <Descriptions.Item label="不确定性">{info.metrics?.["预测不确定性"] || "未记录"}</Descriptions.Item>
                  <Descriptions.Item label="扩展特征">{info.metrics?.["扩展特征"] || "未记录"}</Descriptions.Item>
                  <Descriptions.Item label="样本统计">
                    <Space>
                      <Tag>候选 {info.metrics?.["候选样本"] ?? info.training_data_size}</Tag>
                      <Tag color="green">可靠EOL {info.metrics?.["可靠EOL样本"] ?? info.training_data_size}</Tag>
                      <Tag color="orange">排除 {info.metrics?.["排除样本"] ?? 0}</Tag>
                    </Space>
                  </Descriptions.Item>
                  <Descriptions.Item label="划分方式">{info.metrics?.["评估方式"] || "未记录"}</Descriptions.Item>
                  <Descriptions.Item label="误差口径">页面展示百分比误差：MAPE 为平均绝对百分比误差，NRMSE 为 RMSE 按平均寿命归一化后的百分比。</Descriptions.Item>
                  <Descriptions.Item label="模型精度">
                    <Space>
                      <Tag>MAPE {metricValue(info.metrics?.MAPE, "%")}</Tag>
                      <Tag>NRMSE {metricValue(info.metrics?.NRMSE, "%")}</Tag>
                      <Tag>R² {info.metrics?.R2}</Tag>
                    </Space>
                  </Descriptions.Item>
                  <Descriptions.Item label="特征列表">{(info.feature_list || []).map((f: string) => <Tag key={f}>{f}</Tag>)}</Descriptions.Item>
                  <Descriptions.Item label="超参数"><pre>{JSON.stringify(info.hyperparameters || {}, null, 2)}</pre></Descriptions.Item>
                </Descriptions>
              ),
            })),
            ...(models.length ? [] : [{ key: "empty", label: "模型信息", children: <Empty description="教师端训练后显示模型信息" /> }]),
            ...(isTeacher ? [{ key: "source", label: "训练源码", children: <pre className="sourceBlock">{source}</pre> }] : []),
          ]}
        />
      </Card>
    </div>
  );
}
