$ErrorActionPreference = "Stop"

Write-Host "Installing backend dependencies..." -ForegroundColor Cyan
$PythonCommand = $null
$Python = Get-Command python -ErrorAction SilentlyContinue
if ($Python -and $Python.Source -notlike "*Microsoft\WindowsApps*") {
  $PythonCommand = "python"
} elseif (Get-Command uv -ErrorAction SilentlyContinue) {
  Write-Host "未找到 PATH 中的真实 python，改用 uv 管理的 Python 3.11。" -ForegroundColor Yellow
  uv venv --python 3.11 .venv
  $PythonCommand = ".\.venv\Scripts\python"
} else {
  throw "未找到可用 Python。请安装 Python 3.10+，或安装 uv 后重试。"
}

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
  & $PythonCommand -m venv .venv
}
if (Get-Command uv -ErrorAction SilentlyContinue) {
  uv pip install --python .\.venv\Scripts\python.exe -r backend\requirements.txt
} else {
  .\.venv\Scripts\python -m ensurepip --upgrade
  .\.venv\Scripts\python -m pip install --upgrade pip
  .\.venv\Scripts\pip install -r backend\requirements.txt
}

Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
Push-Location frontend
npm install
Pop-Location

Write-Host "Starting backend at http://127.0.0.1:8010 ..." -ForegroundColor Green
Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoExit", "-Command", "cd '$PWD'; .\.venv\Scripts\python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8010"

Write-Host "Starting frontend at http://127.0.0.1:5173 ..." -ForegroundColor Green
Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoExit", "-Command", "cd '$PWD\frontend'; npm run dev"

Write-Host "Done. Visit http://127.0.0.1:5173" -ForegroundColor Green
