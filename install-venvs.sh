cd lender_match
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
pip install lightgbm==4.6.0 scikit-learn==1.8.0   # the versions the pickled models were saved with
python main.py                                    # check http://localhost:8000/health, then Ctrl+C
