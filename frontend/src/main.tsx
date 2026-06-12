import React, { useMemo, useState } from "react";
import ReactDOM from "react-dom/client";
import { ConfigProvider } from "antd";
import zhCN from "antd/locale/zh_CN";
import { BrowserRouter, Navigate, Route, Routes, useNavigate } from "react-router-dom";
import AppLayout from "./pages/AppLayout";
import Login from "./pages/Login";
import Predict from "./pages/Predict";
import Datasets from "./pages/Datasets";
import ModelInfo from "./pages/ModelInfo";
import ModelManage from "./pages/ModelManage";
import History from "./pages/History";
import "./styles.css";

export interface AuthState {
  role: "student" | "teacher" | null;
  name: string;
  username?: string;
  token?: string;
}

export const AuthContext = React.createContext<{
  auth: AuthState;
  setAuth: (auth: AuthState) => void;
}>({ auth: { role: null, name: "" }, setAuth: () => undefined });

function Guard({ children, teacherOnly = false }: { children: React.ReactNode; teacherOnly?: boolean }) {
  const { auth } = React.useContext(AuthContext);
  if (!auth.role) return <Navigate to="/login" replace />;
  if (teacherOnly && auth.role !== "teacher") return <Navigate to="/predict" replace />;
  return <>{children}</>;
}

function Router() {
  const navigate = useNavigate();
  const [auth, setAuthState] = useState<AuthState>({
    role: (localStorage.getItem("role") as AuthState["role"]) || null,
    name: localStorage.getItem("name") || "",
    username: localStorage.getItem("username") || "",
    token: localStorage.getItem("token") || "",
  });
  const setAuth = (next: AuthState) => {
    if (next.role) {
      localStorage.setItem("role", next.role);
      localStorage.setItem("name", next.name);
      if (next.username) localStorage.setItem("username", next.username);
      if (next.token) localStorage.setItem("token", next.token);
    } else {
      localStorage.clear();
      navigate("/login");
    }
    setAuthState(next);
  };
  const value = useMemo(() => ({ auth, setAuth }), [auth]);

  return (
    <AuthContext.Provider value={value}>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/" element={<Guard><AppLayout /></Guard>}>
          <Route index element={<Navigate to="/predict" replace />} />
          <Route path="predict" element={<Predict />} />
          <Route path="datasets" element={<Datasets />} />
          <Route path="model" element={<ModelInfo />} />
          <Route path="model-manage" element={<Guard teacherOnly><ModelManage /></Guard>} />
          <Route path="history" element={<History />} />
          <Route path="student-records" element={<Guard teacherOnly><History scope="all" /></Guard>} />
          <Route path="dataset-manage" element={<Navigate to="/datasets" replace />} />
        </Route>
      </Routes>
    </AuthContext.Provider>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <ConfigProvider locale={zhCN} theme={{ token: { colorPrimary: "#0f766e", borderRadius: 8, fontFamily: "Microsoft YaHei, Segoe UI, sans-serif" } }}>
    <BrowserRouter>
      <Router />
    </BrowserRouter>
  </ConfigProvider>,
);
