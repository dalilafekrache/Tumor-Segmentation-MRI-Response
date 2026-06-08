# test_pipeline_B.py
from comparative_study import (
    CONFIG, load_labels, split_train_test,
    run_pipeline_B, prepare_masks_dir_for_experiment_direct
)
from pathlib import Path
from datetime import datetime
import pandas as pd

if __name__ == "__main__":
    # Charger labels et split
    labels = load_labels(CONFIG['LABELS_CSV'])
    train_df, test_df = split_train_test(labels, CONFIG['TEST_SIZE'], CONFIG['SEED'])

    # Choisir le modèle dont on veut tester Pipeline B
    seg_model = "VNet"   # <-- change ici (Medformer, AttentionUnet, ResUnetpp, nnunet, etc.)

    # Récupérer le dossier des masques correspondants
    masks_dir = prepare_masks_dir_for_experiment_direct(seg_model)

    # Dossier de sortie
    out_dir = Path(CONFIG['OUTPUT_DIR']) / f"testB_{seg_model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Test Pipeline B avec les masques : {seg_model} ===")

    # Lancer pipeline B
    res = run_pipeline_B(train_df, test_df, masks_dir, out_dir, CONFIG)

    print("\n=== Résultats Pipeline B ===")
    print(res)

    # Sauvegarde du résumé
    if res is not None:
        pd.DataFrame([res]).to_csv(out_dir / "summary_B.csv", index=False)
        print(f"\nRésultats sauvegardés dans : {out_dir}")

