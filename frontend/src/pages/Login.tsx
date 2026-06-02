import React from "react";
import { Button, Card, Form, Input, message, Typography } from "antd";
import { BatteryCharging, Database, LineChart, ShieldCheck } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { api, apiError } from "../api/client";
import { AuthContext } from "../main";

export default function Login() {
  const navigate = useNavigate();
  const { setAuth } = React.useContext(AuthContext);
  const [loading, setLoading] = React.useState(false);

  async function submit(values: { username: string; password: string }) {
    setLoading(true);
    try {
      const { data } = await api.post("/login", values);
      setAuth({ role: data.role, name: data.name });
      navigate("/predict");
    } catch (error) {
      message.error(apiError(error));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="loginPage">
      <section className="loginHero">
        <div className="loginMark"><BatteryCharging size={36} /></div>
        <Typography.Title>电池寿命预测与健康评估系统</Typography.Title>
        <p>面向竞赛答辩的锂电池循环数据分析、曲线匹配与多模型预测平台。</p>
        <div className="loginMetrics">
          <div><LineChart size={18} /><strong>SOH 曲线</strong><span>趋势对比</span></div>
          <div><Database size={18} /><strong>样本数据库</strong><span>本地持久化</span></div>
          <div><ShieldCheck size={18} /><strong>师生权限</strong><span>演示可控</span></div>
        </div>
      </section>
      <Card className="loginCard">
        <Typography.Title level={3}>进入系统</Typography.Title>
        <Form layout="vertical" initialValues={{ username: "student", password: "123456" }} onFinish={submit}>
          <Form.Item label="账号" name="username" rules={[{ required: true, message: "请输入账号" }]}>
            <Input size="large" placeholder="student / teacher" />
          </Form.Item>
          <Form.Item label="密码" name="password" rules={[{ required: true, message: "请输入密码" }]}>
            <Input.Password size="large" placeholder="123456" />
          </Form.Item>
          <Button size="large" type="primary" htmlType="submit" block loading={loading}>登录系统</Button>
        </Form>
        <div className="loginHint">预置账号：student/123456，teacher/123456</div>
      </Card>
    </main>
  );
}
