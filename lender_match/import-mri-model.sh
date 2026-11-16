cd ~/path/to/MRI            # ← change to the MRI project folder
DEST=/mnt/c/Users/peter/gwu/spring-2026/ISTM6227/group-project-final/_mri_import
mkdir -p "$DEST"

# Code, notebooks, configs and model files, skipping environments, caches, Git history and anything over 200 MB
rsync -a --exclude='.venv' --exclude='venv' --exclude='env' --exclude='__pycache__' \
      --exclude='.git' --exclude='.ipynb_checkpoints' --max-size=200m ./ "$DEST/"

# A list of the large files that were skipped, so I know they exist
find . -type f -size +200M -not -path '*/.venv/*' -not -path '*/.git/*' \
     -exec ls -lh {} \; > "$DEST/_skipped_large_files.txt"

# The exact library versions the model was trained with (activate MRI's venv first)
source .venv/bin/activate 2>/dev/null || source venv/bin/activate
python --version > "$DEST/_wsl_python_version.txt"
pip freeze > "$DEST/_wsl_freeze.txt"
