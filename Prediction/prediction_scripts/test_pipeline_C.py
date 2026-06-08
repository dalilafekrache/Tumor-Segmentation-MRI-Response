# test_pipeline_C.py — VERSION CLEAN
# -*- coding: utf-8 -*-

from comparative_study import (
    CONFIG, load_labels, split_train_test,
    run_pipeline_C, prepare_masks_dir_for_experiment_direct
)
from pathlib import Path
from datetime import datetime
import pandas as pd

if __name__ == "__main__":
    labels = load_labels(CONFIG["LABELS_CSV"])
    train_df, test_df = split_train_test(labels, CONFIG["TEST_SIZE"], CONFIG["SEED"])

    seg_model = "AttentionUnet"   # Change ici

    masks_dir = prepare_masks_dir_for_experiment_direct(seg_model)

    out_dir = Path(CONFIG["OUTPUT_DIR"]) / f"testC_{seg_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Test Pipeline C avec les masques : {seg_model} ===")

    res = run_pipeline_C(train_df, test_df, CONFIG["IMAGES_DIR"], masks_dir, out_dir, CONFIG)

    print("=== Résultats Pipeline C ===")
    print(res)

    pd.DataFrame([res]).to_csv(out_dir / "summary_C.csv", index=False)

    print(f"Résultats sauvegardés dans : {out_dir}")
