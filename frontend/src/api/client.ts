import axios from "axios";
import { message } from "antd";

export const api = axios.create({
  baseURL: "/api",
  timeout: 55000,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(undefined, async (error) => {
  const config = error.config || {};
  if (!config.__retried && (!error.response || error.code === "ECONNABORTED")) {
    config.__retried = true;
    message.info("服务正在唤醒，请稍候，系统将自动重试一次。");
    return api(config);
  }
  return Promise.reject(error);
});

export function apiError(error: unknown) {
  if (axios.isAxiosError(error)) {
    return error.response?.data?.detail || error.message;
  }
  return "请求失败，请稍后重试。";
}
