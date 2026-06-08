# test_pipeline_C_nnunet.py
# -*- coding: utf-8 -*-

from comparative_study_nnunet import (
    CONFIG, load_labels, split_train_test, run_pipeline_C
)
from pathlib import Path
from datetime import datetime
import pandas as pd

if __name__ == "__main__":
    # Charger labels
    labels = load_labels(CONFIG["LABELS_CSV"])
    train_df, test_df = split_train_test(labels, CONFIG["TEST_SIZE"], CONFIG["SEED"])

    seg_model = "nnunet"

    # dossier masques nnUNet
    masks_dir = Path(CONFIG["NNUNET_MASKS_DIR"])

    # dossier résultat
    out_dir = Path(CONFIG["OUTPUT_DIR"]) / f"testC_{seg_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Test Pipeline C (deep features + RF) avec masques : {seg_model} ===")

    res = run_pipeline_C(train_df, test_df, CONFIG["IMAGES_DIR"], masks_dir, out_dir, CONFIG)

    print("\n=== Résultats Pipeline C ===")
    print(res)

    pd.DataFrame([res]).to_csv(out_dir / "summary_C.csv", index=False)

    print(f"\nRésultats sauvegardés dans : {out_dir}")
