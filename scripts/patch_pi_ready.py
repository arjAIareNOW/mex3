"""One-off: make the Pi / Windows setup match the current FDA (Pipecat version, new libraries, jsdom, keys)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# requirements
p = ROOT / "requirements.txt"
s = p.read_text(encoding="utf-8")
if "pypdf" not in s:
    s = s.rstrip("\n") + """
# --- tools server: documents, device check, photos for live game design ---
pypdf==6.21.4                # ask_documents: PDF
python-docx==1.2.0           # ask_documents: Word
openpyxl==3.1.5              # ask_documents: Excel
psutil==7.2.2                # device_check
pillow==13.0.0               # photos for tweak_app ("make the pipes a photo of bamboo"), the pipeline diagram
# Node.js (not pip): `node` checks and tests built apps; jsdom: cd fda/mcp_servers && npm install
"""
    p.write_text(s, encoding="utf-8")

# Pi service: the Pipecat + MCP version
p = ROOT / "deploy/pi/fda.service"
s = p.read_text(encoding="utf-8")
s = s.replace("ExecStart=@APP@/.venv/bin/python -m fda --no-browser", "ExecStart=@APP@/.venv/bin/python -m fda.pc")
s = s.replace("Description=FDA full-duplex voice agent", "Description=FDA full-duplex voice agent (Pipecat + MCP)")
p.write_text(s, encoding="utf-8")

ENV_TEMPLATE = ("# FDA keys (never share this file). Leave a line empty if you do not use that service.\\n"
                "QWEN_API_KEY=\\nGROQ_API_KEY=\\nSAMBANOVA_API_KEY=\\nGEMINI_API_KEY=\\n")

# Pi install
p = ROOT / "deploy/pi/install.sh"
s = p.read_text(encoding="utf-8")
s = s.replace("sudo apt-get install -y python3-venv python3-dev libportaudio2 chromium-browser unclutter",
              "sudo apt-get install -y python3-venv python3-dev libportaudio2 chromium-browser unclutter nodejs npm git")
s = s.replace("pip install -r requirements.txt\n", """pip install -r requirements.txt
# app checker: a real DOM (jsdom) for testing the apps and games FDA builds
(cd fda/mcp_servers && npm install --no-audit --no-fund)
""", 1)
s = s.replace('''[ -f .env ] || { echo "QWEN_API_KEY=paste-your-key-here" > .env; chmod 600 .env; echo ">> edit $APP/.env and paste the API key"; }''',
              f'''[ -f .env ] || {{ printf "{ENV_TEMPLATE}" > .env; chmod 600 .env; echo ">> edit $APP/.env and paste your API keys"; }}''')
s = s.replace('''grep -q '^open_browser' fda.toml || echo 'open_browser = false' >> fda.toml
''', '''grep -q '^open_browser' fda.toml || echo 'open_browser = false' >> fda.toml
# offline models (listening, voice, local LLM) for the engines in fda.toml, built for this Pi
python -m fda download
''')
p.write_text(s, encoding="utf-8")

# Windows setup
p = ROOT / "deploy/windows/setup.bat"
s = p.read_text(encoding="utf-8")
if "npm install" not in s:
    s = s.replace("pip install -r requirements.txt\n", """pip install -r requirements.txt
rem app checker (needs Node.js from nodejs.org): a real DOM for testing the apps FDA builds
pushd fda\\mcp_servers & call npm install --no-audit --no-fund & popd
""", 1)
    s = s.replace("""  echo QWEN_API_KEY=paste-your-key-here> .env
  echo Created .env - open it in Notepad and paste your API key.""", """  (echo QWEN_API_KEY=& echo GROQ_API_KEY=& echo SAMBANOVA_API_KEY=& echo GEMINI_API_KEY=)> .env
  echo Created .env - open it in Notepad and paste your API keys.""")
    s = s.replace("python -m fda check", "python -m fda download\npython -m fda check")
p.write_text(s, encoding="utf-8")

# .gitignore: generated things
p = ROOT / ".gitignore"
s = p.read_text(encoding="utf-8")
for line in ("reports/", "fda/mcp_servers/node_modules/"):
    if line not in s:
        s = s.rstrip("\n") + "\n" + line + "\n"
p.write_text(s, encoding="utf-8")
print("ok")
