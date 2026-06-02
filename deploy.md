# 云端部署说明

本项目已经整理成单服务部署：FastAPI 同时提供后端 API 和前端页面。

## 推荐方案：Render

1. 把项目上传到 GitHub。
2. 打开 Render，选择 `New +` -> `Blueprint`。
3. 选择这个 GitHub 仓库。
4. Render 会自动读取 `render.yaml`，并使用 `Dockerfile` 构建。
5. 部署完成后会得到公网地址，例如：

```text
https://battery-life-prediction.onrender.com
```

这个地址可以在手机、平板、外网电脑上访问，不需要处在同一个局域网。

## 本地验证单服务

如果要在本机验证云端同款运行方式：

```powershell
cd frontend
npm install
npm run build
cd ..
.\.venv\Scripts\python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8010
```

然后访问：

```text
http://127.0.0.1:8010
```

## 注意事项

- 临时 tunnel 地址不稳定，不适合长期使用。
- 真正“随时随地访问”需要部署到云平台。
- 当前 SQLite 数据库位于 `backend/data/battery_system.sqlite`，当前模型文件位于 `backend/models`。它们会随项目一起部署。
- 免费云平台如果重建容器，运行时新写入的数据可能丢失。后续正式使用时，建议改成云数据库或持久化磁盘。
