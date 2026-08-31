#!/usr/bin/env bash
# Install + run predec end-to-end on a fresh Linux box.
# Usage:
#   OPENAI_API_KEY=sk-... bash install_and_run.sh
set -euo pipefail

# Sanity
echo "=== Sanity check ==="
python3 --version
git --version

# Create project dir
PROJECT_DIR="/opt/predec"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Clone if not already there
if [ ! -d "$PROJECT_DIR/.git" ]; then
  echo "=== Cloning repo ==="
  git clone https://github.com/ruzkypazzy/predec.git .
else
  echo "=== Repo already exists; pulling latest ==="
  git pull --rebase origin main
fi

# Create venv
echo "=== Creating venv ==="
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip + install
echo "=== Installing package ==="
pip install --upgrade pip --quiet
pip install -e . --quiet

# Sanity: verify the CLI is callable
echo "=== Verifying CLI ==="
predec --help | head -10

# Build the eval set if not present
if [ ! -f eval/eval_pairs.jsonl ]; then
  echo "=== Building eval set ==="
  python3 scripts/build_eval_set.py
fi

# Run the full eval
echo "=== Running eval (agent + baseline) ==="
mkdir -p runs/eval
python3 scripts/run_eval.py --out runs/eval 2>&1 | tee runs/eval/run.log

echo ""
echo "=== DONE ==="
echo "Report:    $PROJECT_DIR/runs/eval/report/report.html"
echo "Results:   $PROJECT_DIR/runs/eval/eval_results.json"
echo "Trajectory: $PROJECT_DIR/runs/eval/report/trajectory.json"
echo ""
echo "Open the HTML report:"
echo "  firefox $PROJECT_DIR/runs/eval/report/report.html"
echo "  # or scp it back to your local machine:"
echo "  scp root@185.2.101.134:$PROJECT_DIR/runs/eval/report/report.html ."
