# test_pipeline_A_nnunet.py
from comparative_study_nnunet import CONFIG, load_labels, split_train_test, run_pipeline_A
from pathlib import Path
from datetime import datetime
import pandas as pd

if __name__ == "__main__":
    # Chargement labels
    labels = load_labels(CONFIG['LABELS_CSV'])

    # Split train / test
    train_df, test_df = split_train_test(labels, CONFIG['TEST_SIZE'], CONFIG['SEED'])

    # Modèle = nnUNet
    seg_model = "nnunet"

    # On prend directement le dossier contenant tes masques nnunet
    masks_dir = Path(CONFIG['MASKS_PRED_ROOT']) / seg_model

    out_dir = Path(CONFIG['OUTPUT_DIR']) / f"testA_{seg_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Test Pipeline A avec les masques {seg_model} ===")
    res = run_pipeline_A(train_df, test_df, CONFIG['IMAGES_DIR'], masks_dir, out_dir, CONFIG)

    print("\n=== Résultats ===")
    print(res)

    pd.DataFrame([res]).to_csv(out_dir / "summary_A.csv", index=False)
    print(f"\nRésultats sauvegardés dans : {out_dir}")
