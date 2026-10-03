#!/bin/bash
set -e

# === CONFIG ===
INPUT_IMAGES="../images/Targets"
OUTPUT_DIR="../images/CLRerNet_Viz"
CLRERNET_REPO="./CLRerNet"
# Point this to where your .pth file is
CHECKPOINT="./clrernet_culane_dla34_ema.pth"
# =============

if [ ! -d "$CLRERNET_REPO/clrer" ]; then
    echo "Error: CLRerNet venv not found."
    exit 1
fi

source "$CLRERNET_REPO/clrer/bin/activate"

python generate_lanes.py \
    --input "$INPUT_IMAGES" \
    --output_dir "$OUTPUT_DIR" \
    --repo_dir "$CLRERNET_REPO" \
    --checkpoint "$CHECKPOINT"
    
echo "Done."