# US Market Risk Dashboard

监控美股市场系统性风险的静态仪表盘，覆盖6大维度32个指标。通过 GitHub Pages 托管，GitHub Actions 每日自动更新数据。

## Setup

1. Fork 或 clone 本仓库
2. 获取免费 FRED API Key: https://fred.stlouisfed.org/docs/api/api_key.html
3. 在仓库 Settings → Secrets and variables → Actions → New repository secret，添加 `FRED_API_KEY`
4. 在仓库 Settings → Pages → Source: Deploy from branch → `main` / `root`
5. 等待首次 Action 运行（或在 Actions 标签页手动触发）

## Local Development

```bash
# 安装依赖
pip install -r requirements.txt

# 运行数据获取
FRED_API_KEY=your_key python scripts/fetch_data.py

# 打开 index.html 查看仪表盘
open index.html
```

## Disclaimer

本仪表盘仅供学习研究，不构成任何投资建议。
