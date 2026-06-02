import axios from "axios";

export const api = axios.create({
  baseURL: "/api",
});

api.interceptors.request.use((config) => {
  const role = localStorage.getItem("role") || "student";
  config.headers["X-Role"] = role;
  return config;
});

export function apiError(error: unknown) {
  if (axios.isAxiosError(error)) {
    return error.response?.data?.detail || error.message;
  }
  return "请求失败，请稍后重试。";
}
