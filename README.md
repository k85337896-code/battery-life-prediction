# 电池寿命预测与健康评估 Web 系统

这是一个用于本科生实验竞赛演示的本地 Web 系统，包含学生端与教师端。系统支持上传电池循环 CSV，基于数据库曲线匹配预测循环寿命与 SOH，并展示 XGBoost 模型信息。

## 技术栈

- 后端：Python 3.10+、FastAPI、SQLite、pandas、numpy、scikit-learn、xgboost
- 前端：React + Vite + TypeScript、Ant Design、ECharts
- 数据：首次启动自动生成合成演示数据与示例输入 CSV

## 一键启动

在项目根目录运行：

```powershell
.\start.ps1
```

启动后访问：

- 前端：http://127.0.0.1:5173
- 后端 API：http://127.0.0.1:8010/docs

预置账号：

- 学生：`student / 123456`
- 教师：`teacher / 123456`

## 手动启动

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\pip install -r backend\requirements.txt
.\.venv\Scripts\python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8010
```

另开终端：

```powershell
cd frontend
npm install
npm run dev
```

## 云端访问

项目已支持单体部署：前端构建后由 FastAPI 托管，详见 [deploy.md](./deploy.md)。

## CSV 格式

示例文件位于 `sample_data/`。上传 CSV 至少包含：

```csv
cycle,specific_capacity
0,155.20
20,154.10
```

也支持中文表头 `循环次数`、`容量`、`比容量`。解析失败时后端会返回中文错误。

## 功能说明

- 寿命预测：上传 CSV，选择电池类型，填写容量与倍率，得到预测循环寿命、剩余寿命、SOH、相关性与曲线图。
- 历史记录：保存每次预测，可搜索、查看详情、删除。
- 数据库：查看合成或导入的电池数据库条目与曲线。
- 模型展示：查看 XGBoost 模型指标、特征、超参数与训练源码。
- 教师端：模型重训练、种子数据重置、数据库新增/删除、CSV 导入。

## 数据与模型假设

当前 `backend/app/seed.py` 生成的是合成演示数据，用指数衰减叠加轻微噪声模拟 LCO、LFP、LS 三类电池曲线。它适合答辩演示，不代表真实实验结论。

替换为真实数据时，可按以下方式处理：

1. 将真实电池循环曲线整理为 `cycle,specific_capacity` CSV。
2. 使用教师端「CSV 批量导入」写入数据库。
3. 在教师端「模型管理」点击重新训练。

共享数据、历史记录和模型信息均保存在后端 SQLite，不依赖浏览器 localStorage 作为唯一持久层。
