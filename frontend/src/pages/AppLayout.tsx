import React from "react";
import { Layout, Menu, Button, Typography } from "antd";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import { Activity, BatteryCharging, Database, Gauge, LogOut, ClipboardList, Users } from "lucide-react";
import { AuthContext } from "../main";

const { Sider, Header, Content } = Layout;

export default function AppLayout() {
  const navigate = useNavigate();
  const location = useLocation();
  const { auth, setAuth } = React.useContext(AuthContext);
  const items = [
    { key: "/predict", icon: <Gauge size={18} />, label: "寿命预测" },
    { key: "/datasets", icon: <Database size={18} />, label: "数据集管理" },
    { key: "/model", icon: <Activity size={18} />, label: auth.role === "teacher" ? "模型中心" : "模型展示" },
    { key: auth.role === "teacher" ? "/student-records" : "/history", icon: <ClipboardList size={18} />, label: auth.role === "teacher" ? "学生预测记录" : "我的预测记录" },
    ...(auth.role === "teacher" ? [{ key: "/students", icon: <Users size={18} />, label: "学生信息管理" }] : []),
  ];

  return (
    <Layout className="shell">
      <Sider width={268} className="side">
        <div className="brand">
          <div className="brandIcon"><BatteryCharging size={25} /></div>
          <div>
            <Typography.Title level={4}>电池寿命预测系统</Typography.Title>
            <span>Battery SOH Research Console</span>
          </div>
        </div>
        <Menu mode="inline" selectedKeys={[location.pathname]} items={items} onClick={(item) => navigate(item.key)} />
        <div className="sideStatus">
          <span className="statusDot" />
          <div>
            <strong>本地真实数据集</strong>
            <small>化学成分 · 数据集 · 单体电池</small>
          </div>
        </div>
      </Sider>
      <Layout className="mainArea">
        <Header className="topbar">
          <div>
            <span className="topEyebrow">锂电池循环寿命与健康状态评估</span>
            <strong>{auth.name}</strong>
            <span className="rolePill">{auth.role === "teacher" ? "教师端" : "学生端"}</span>
          </div>
          <Button icon={<LogOut size={16} />} onClick={() => setAuth({ role: null, name: "" })}>退出</Button>
        </Header>
        <Content className="content"><Outlet /></Content>
      </Layout>
    </Layout>
  );
}
