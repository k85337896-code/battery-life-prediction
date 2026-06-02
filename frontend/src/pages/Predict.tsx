import React from "react";
import { Alert, Button, Card, Col, Form, InputNumber, message, Row, Select, Space, Statistic, Tag, Upload } from "antd";
import { Download, FileUp, Gauge, Play, Sparkles } from "lucide-react";
import type { UploadFile } from "antd";
import { useNavigate } from "react-router-dom";
import { api, apiError } from "../api/client";
import { CurveChart } from "../components/Chart";
import { SohProgress } from "../components/SohProgress";
import type { PredictionResult } from "../types";

export default function Predict() {
  const navigate = useNavigate();
  const [fileList, setFileList] = React.useState<UploadFile[]>([]);
  const [result, setResult] = React.useState<PredictionResult | null>(null);
  const [loading, setLoading] = React.useState(false);
  const [modelOptions, setModelOptions] = React.useState<Array<{ value: string; label: string }>>([]);
  const [batteryOptions, setBatteryOptions] = React.useState<Array<{ value: string; label: string }>>([]);
  const isTeacher = localStorage.getItem("role") === "teacher";
  const [form] = Form.useForm();

  React.useEffect(() => {
    api.get("/meta").then((res) => {
      const options = Object.entries(res.data.model_options || {}).map(([value, label]) => ({ value, label: String(label) }));
      setModelOptions(options);
      setBatteryOptions(Object.entries(res.data.battery_types || {}).map(([value, label]) => ({ value, label: `${value} ${label}` })));
    });
  }, []);

  async function submit(values: any) {
    const file = fileList[0]?.originFileObj;
    if (!file) return message.warning("请先上传 CSV 文件。");
    const body = new FormData();
    body.append("file", file);
    Object.entries(values).forEach(([key, value]) => body.append(key, String(value)));
    setLoading(true);
    try {
      const { data } = await api.post("/predict", body, { headers: { "Content-Type": "multipart/form-data" } });
      setResult(data);
      message.success("预测完成，历史记录已保存。");
    } catch (error) {
      message.error(apiError(error));
    } finally {
      setLoading(false);
    }
  }

  function exportCsv() {
    if (!result) return;
    const rows = ["cycle,specific_capacity,soh", ...result.predicted_curve.map((p) => `${p.cycle},${p.specific_capacity},${p.soh}`)];
    const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "预测曲线.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="pageStack">
      <section className="pageHero">
        <div>
          <span className="pageKicker"><Sparkles size={15} /> Battery Life Prediction</span>
          <h1>寿命预测工作台</h1>
          <p>上传循环容量数据，系统会先进行同类型曲线匹配，再调用所选机器学习模型给出直接预测结果。</p>
        </div>
        <div className="heroActions">
          {isTeacher && <Button size="large" onClick={() => navigate("/model-manage")}>去训练模型</Button>}
          <div className="heroStats">
            <div><strong>80%</strong><span>EOL 判据</span></div>
            <div><strong>Top-3</strong><span>相似曲线</span></div>
          <div><strong>4</strong><span>可选模型</span></div>
          </div>
        </div>
      </section>

      <Row gutter={[18, 18]} align="stretch">
        <Col xs={24} xl={9}>
          <Card className="controlPanel" title="输入参数">
            <Form form={form} layout="vertical" initialValues={{ battery_type: "G1", theoretical_capacity: 4, rated_capacity: 4, c_rate: 1, model_key: "xgboost" }} onFinish={submit}>
              <Form.Item label="循环数据 CSV" required>
                <Upload.Dragger className="uploadBox" accept=".csv" maxCount={1} beforeUpload={() => false} fileList={fileList} onChange={({ fileList }) => setFileList(fileList)}>
                  <FileUp size={30} />
                  <p>拖拽或点击上传 CSV</p>
                  <span>至少包含 cycle 与 capacity/specific_capacity 列</span>
                </Upload.Dragger>
              </Form.Item>
              <Form.Item label="电池类型" name="battery_type" rules={[{ required: true }]}>
                <Select size="large" options={batteryOptions} />
              </Form.Item>
              <Row gutter={12}>
                <Col span={12}><Form.Item label="理论容量" name="theoretical_capacity" rules={[{ required: true }]}><InputNumber size="large" min={1} className="full" /></Form.Item></Col>
                <Col span={12}><Form.Item label="额定容量" name="rated_capacity" rules={[{ required: true }]}><InputNumber size="large" min={1} className="full" /></Form.Item></Col>
              </Row>
              <Row gutter={12}>
                <Col span={12}><Form.Item label="倍率 (C)" name="c_rate" rules={[{ required: true }]}><InputNumber size="large" min={0.1} max={5} step={0.1} className="full" /></Form.Item></Col>
                <Col span={12}><Form.Item label="预测模型" name="model_key" rules={[{ required: true }]}><Select size="large" options={modelOptions} /></Form.Item></Col>
              </Row>
              <Button className="primaryAction" type="primary" htmlType="submit" icon={<Play size={16} />} loading={loading} block>开始预测</Button>
            </Form>
          </Card>
        </Col>

        <Col xs={24} xl={15}>
          <div className="resultGrid">
            <Card className="resultCard highlightMetric">
              <Statistic title="预测循环寿命" value={result?.predicted_cycle_life ?? "-"} suffix={result ? "次" : ""} />
              <Tag color="cyan">早期曲线外推</Tag>
            </Card>
            <Card className="resultCard">
              <Statistic title="预测剩余寿命" value={result?.predicted_remaining_life ?? "-"} suffix={result ? "次" : ""} />
              <Tag color="blue">扣除当前循环</Tag>
            </Card>
            <Card className="resultCard">
              <Statistic title="匹配相关性" value={result?.correlation_score ?? "-"} precision={3} />
              <Tag color="green">Top-1 匹配</Tag>
            </Card>
            <Card className="resultCard">
              <Statistic title={result?.selected_model_name || "模型直接预测"} value={result?.model_predicted_life ?? "-"} suffix={result?.model_predicted_life ? "次" : ""} />
              <Tag color="purple">ML 对照结果</Tag>
            </Card>
          </div>

          {result?.prediction_uncertainty_cycles && (
            <Alert
              className="sohPanel"
              type="warning"
              showIcon
              message={`预测寿命约 ${result.predicted_cycle_life} ± ${result.prediction_uncertainty_cycles} 圈`}
              description={`参考区间：${result.predicted_life_lower} - ${result.predicted_life_upper} 圈。早期循环信息有限，该区间来自当前模型留一评估误差。`}
            />
          )}

          <Card className="sohPanel" title={<Space><Gauge size={18} />SOH 健康状态</Space>}>
            {result ? <SohProgress value={result.soh_at_prediction} /> : <div className="emptyHint">完成一次预测后显示健康状态与 EOL 参考。</div>}
          </Card>
        </Col>
      </Row>

      <Card className="chartPanel" title="比容量衰减曲线" extra={<Button icon={<Download size={16} />} onClick={exportCsv} disabled={!result}>导出数据</Button>}>
        {result ? (
          <CurveChart input={result.input_curve} predicted={result.predicted_curve} matches={result.top_matches} height={430} />
        ) : (
          <div className="chartPlaceholder">上传示例 CSV 并点击「开始预测」后，曲线对比图将在这里生成。</div>
        )}
      </Card>
    </div>
  );
}
