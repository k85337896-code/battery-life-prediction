import React from "react";
import { Alert, Button, Card, Col, Form, InputNumber, message, Row, Select, Skeleton, Space, Statistic, Tag, Upload } from "antd";
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
  const [models, setModels] = React.useState<Array<{ model_key: string; model_type: string; chemistry: string; version: number }>>([]);
  const [chemistries, setChemistries] = React.useState<string[]>([]);
  const [errorText, setErrorText] = React.useState("");
  const isTeacher = localStorage.getItem("role") === "teacher";
  const [form] = Form.useForm();

  React.useEffect(() => {
    api.get("/meta").then((res) => {
      const nextChemistries = res.data.chemistries || [];
      setChemistries(nextChemistries);
      if (!form.getFieldValue("chemistry") && nextChemistries[0]) form.setFieldValue("chemistry", nextChemistries[0]);
      api.get("/models/published").then((modelRes) => setModels(modelRes.data || []));
    });
  }, []);

  const selectedChemistry = Form.useWatch("chemistry", form);
  const modelOptions = models
    .filter((model) => !selectedChemistry || model.chemistry === selectedChemistry)
    .map((model) => ({ value: model.model_key, label: `${model.model_type} v${model.version}` }));

  async function submit(values: any) {
    const file = fileList[0]?.originFileObj;
    if (!file) return message.warning("请先上传 CSV 文件。");
    const body = new FormData();
    body.append("file", file);
    Object.entries(values).forEach(([key, value]) => body.append(key, String(value)));
    setLoading(true);
    setErrorText("");
    try {
      const { data } = await api.post("/predict", body, { headers: { "Content-Type": "multipart/form-data" } });
      setResult(data);
      message.success("预测完成，历史记录已保存。");
    } catch (error) {
      const text = apiError(error);
      setErrorText(text);
      message.error(text);
    } finally {
      setLoading(false);
    }
  }

  function exportCsv() {
    if (!result) return;
    const rows = ["cycle,soh,lower,upper", ...(result.soh_curve || result.predicted_curve).map((p) => `${p.cycle},${p.soh},${p.lower ?? ""},${p.upper ?? ""}`)];
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
          <p>上传单块电池的电压/电流长表时序，系统提取早期特征并输出完整 SOH 衰减曲线与 EOL。</p>
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
            <Form form={form} layout="vertical" initialValues={{ chemistry: chemistries[0], rated_capacity: 4, c_rate: 1 }} onFinish={submit}>
              <Form.Item label="循环数据 CSV" required>
                <Upload.Dragger className="uploadBox" accept=".csv" maxCount={1} beforeUpload={(file) => {
                  if (file.size > 20 * 1024 * 1024) {
                    message.error("CSV 文件不能超过 20MB。");
                    return Upload.LIST_IGNORE;
                  }
                  return false;
                }} fileList={fileList} onChange={({ fileList }) => setFileList(fileList)}>
                  <FileUp size={30} />
                  <p>拖拽或点击上传 CSV</p>
                  <span>必需列：cycle, time_s, voltage_V, current_A；可选 capacity_Ah, temperature_C</span>
                </Upload.Dragger>
              </Form.Item>
              <Form.Item label="化学体系" name="chemistry" rules={[{ required: true, message: "请选择化学体系" }]}>
                <Select size="large" options={chemistries.map((value) => ({ value, label: value }))} onChange={() => form.setFieldValue("model_key", undefined)} />
              </Form.Item>
              <Row gutter={12}>
                <Col span={12}><Form.Item label="额定容量 (Ah)" name="rated_capacity" rules={[{ required: true, message: "请输入额定容量" }]}><InputNumber size="large" min={0.001} max={10000} step={0.001} className="full" /></Form.Item></Col>
                <Col span={12}><Form.Item label="倍率 (C)" name="c_rate" rules={[{ required: true }]}><InputNumber size="large" min={0.1} max={5} step={0.1} className="full" /></Form.Item></Col>
              </Row>
              <Form.Item label="选择模型" name="model_key" rules={[{ required: true, message: "请选择教师已发布模型" }]}>
                <Select size="large" options={modelOptions} placeholder={selectedChemistry ? "选择已发布模型" : "先选择化学体系"} />
              </Form.Item>
              <Button className="primaryAction" type="primary" htmlType="submit" icon={<Play size={16} />} loading={loading} block>开始预测</Button>
            </Form>
          </Card>
        </Col>

        <Col xs={24} xl={15}>
          <div className="resultGrid">
            {loading && <Skeleton active paragraph={{ rows: 8 }} />}
            {errorText && <Alert className="sohPanel" type="error" showIcon message="预测失败" description={errorText} />}
            <Card className="resultCard highlightMetric">
              <Statistic title="预测 EOL" value={result?.predicted_eol_cycle ?? result?.predicted_cycle_life ?? "-"} suffix={result ? "次" : ""} />
              <Tag color="cyan">早期曲线外推</Tag>
            </Card>
            <Card className="resultCard">
              <Statistic title="预测剩余寿命" value={result?.remaining_cycles ?? result?.predicted_remaining_life ?? "-"} suffix={result ? "次" : ""} />
              <Tag color="blue">扣除当前循环</Tag>
            </Card>
            <Card className="resultCard">
              <Statistic title="匹配相关性" value={result?.correlation_score ?? "-"} precision={3} />
              <Tag color="green">Top-1 匹配</Tag>
            </Card>
            <Card className="resultCard">
              <Statistic title="规则校正后预测" value={result?.model_predicted_life ?? "-"} suffix={result?.model_predicted_life ? "次" : ""} />
              <Tag color="purple">{result?.selected_model_name || "模型输出"}</Tag>
            </Card>
          </div>

          {!!result?.warnings?.length && (
            <Alert className="sohPanel" type="warning" showIcon message="预测提示" description={result.warnings.join("；")} />
          )}

          {result?.prediction_uncertainty_cycles && (
            <Alert
              className="sohPanel"
              type="warning"
              showIcon
              message={`预测到 80% EOL 约 ${result.predicted_cycle_life} ± ${result.prediction_uncertainty_cycles} 圈`}
              description={`参考区间：${result.predicted_life_lower} - ${result.predicted_life_upper} 圈。曲线只展示到 80% SOH，低于该阈值按电池损坏处理。`}
            />
          )}

          <Card className="sohPanel" title={<Space><Gauge size={18} />SOH 健康状态</Space>}>
            {result ? <SohProgress value={result.soh_at_prediction} /> : <div className="emptyHint">完成一次预测后显示健康状态与 EOL 参考。</div>}
          </Card>
        </Col>
      </Row>

      <Card className="chartPanel" title="SOH 寿命曲线" extra={<Button icon={<Download size={16} />} onClick={exportCsv} disabled={!result}>导出数据</Button>}>
        {result ? (
          <CurveChart input={result.input_curve} predicted={result.predicted_curve} sohCurve={result.soh_curve} matches={result.top_matches} height={430} />
        ) : (
          <div className="chartPlaceholder">上传示例 CSV 并点击「开始预测」后，曲线对比图将在这里生成。</div>
        )}
      </Card>
    </div>
  );
}
