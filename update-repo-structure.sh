#!/usr/bin/env bash
# Moves Loan Arranger into lender_match/. Run in Git Bash with PyCharm closed.
set -euo pipefail
cd /c/Users/peter/gwu/spring-2026/ISTM6227/group-project-final

# 1. Delete the old virtual environment and Python caches (venvs can't be moved; you'll recreate it)
rm -rf .venv __pycache__ mcp_server/__pycache__

# 2. Move the tracked files with git mv so they keep their history.
#    Folders carry their ignored contents along (raw HMDA data, extra model files).
git mv 01_preprocess.py 01b_merge.py 02_train_embeddings.py 03_train_lgbm.py \
       build_county_lookup.py build_lender_meta.py build_lender_names.py \
       compute_fairness.py list_columns.py main.py model_loader.py schemas.py \
       requirements.txt training_log.json training-notes-log.docx \
       GPU-in-action.png preprocess_output.txt \
       data models mcp_server payloads \
       lender_match/

# 3. Move the files that aren't in Git yet
mv model_loader.py.orig lender_match/          # ignored backup; delete when you're satisfied
mkdir -p docs policy/sources
mv ISTM_Initial_Proposal.pdf ISTM_Initial_Proposal_Pipeline_version.docx \
   "Mortgage Readiness Navigator"*.docx docs/
mv Selling-Guide_11-04-26_v3highlight.pdf "Servicing Guide_10-14-26 FINAL.pdf" policy/sources/

# 4. Stage everything and show the five largest staged files
git add -A
git status --short
echo "--- Largest staged files (expect lender_match/models/embedding_weights.pt, about 20 MB, at the bottom):"
git diff --cached --name-only --diff-filter=AR | xargs -d '\n' du -h | sort -h | tail -5
