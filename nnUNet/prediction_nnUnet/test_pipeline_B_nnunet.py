# test_pipeline_B_nnunet.py
# -*- coding: utf-8 -*-
from comparative_study_nnunet import (
    CONFIG, load_labels, split_train_test,
    run_pipeline_B, plot_confusion_matrix, plot_roc_pr
)

from pathlib import Path
from datetime import datetime
from comparative_study_nnunet import CONFIG, load_labels, split_train_test
from comparative_study_nnunet import run_pipeline_B
from pathlib import Path
from datetime import datetime
import pandas as pd

if __name__ == "__main__":
    # Load labels
    labels = load_labels(CONFIG["LABELS_CSV"])
    train_df, test_df = split_train_test(labels, CONFIG["TEST_SIZE"], CONFIG["SEED"])

    seg_model = "nnunet"

    # Use the correct nnUNet masks directory from CONFIG
    masks_dir = Path(CONFIG["NNUNET_MASKS_DIR"])

    # Output directory
    out_dir = Path(CONFIG["OUTPUT_DIR"]) / f"testB_{seg_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Test Pipeline B with masks: {seg_model} ===")

    res = run_pipeline_B(train_df, test_df, masks_dir, out_dir, CONFIG)

    print("=== Pipeline B results ===")
    print(res)

    pd.DataFrame([res]).to_csv(out_dir / "summary_B.csv", index=False)

    print(f"Results saved in: {out_dir}")
