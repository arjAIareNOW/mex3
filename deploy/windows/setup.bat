@echo off
rem One-time setup on the Windows laptop (needs Python 3.11 or 3.12 from python.org, "Add to PATH" ticked).
cd /d "%~dp0\..\.."
py -3 -m venv .venv || python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
rem app checker (needs Node.js from nodejs.org): a real DOM for testing the apps FDA builds
pushd fda\mcp_servers & call npm install --no-audit --no-fund & popd
if not exist .env (
  (echo GROQ_API_KEY=& echo QWEN_API_KEY=)> .env
  echo Created .env - open it in Notepad and paste your API keys.
)
python -m fda download
python -m fda check
pause
