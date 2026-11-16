cd /c/Users/peter/gwu/spring-2026/ISTM6227/group-project-final/lender_match
deactivate 2>/dev/null; rm -rf .venv

py -3.11 -m venv .venv
# If "py" isn't found, use the full path the old venv used:
# /c/Users/peter/AppData/Local/Programs/Python/Python311/python.exe -m venv .venv

source .venv/Scripts/activate
python --version        # must say 3.11.9

pip install numpy==2.4.3 pandas==3.0.2 pyarrow==23.0.1 joblib==1.5.3 \
            lightgbm==4.6.0 scikit-learn==1.8.0 fastapi==0.135.3 \
            "uvicorn[standard]==0.44.0" pydantic==2.12.5 mcp==1.27.0 httpx==0.28.1
pip install torch --index-url https://download.pytorch.org/whl/cpu
python main.py
