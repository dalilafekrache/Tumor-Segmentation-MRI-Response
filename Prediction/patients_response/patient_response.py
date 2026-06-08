import os
import shutil
import csv
from datetime import datetime

# ----------------- CONFIG -----------------
images_dir = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\dataset\images"
masks_dir  = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\dataset\masks"

output_images = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\dataset_subset\images"
output_masks  = r"C:\Users\Admin\PycharmProjects\pythonProjetPrediction\dataset_subset\masks"

prefixes_file = "liste_recherche.txt"

# Fichiers de sortie
prefixes_trouves_file    = "prefixes_trouves.txt"
prefixes_non_trouves_file= "prefixes_non_trouves.txt"
summary_csv = "copy_summary.csv"
log_file = "copy_log.txt"

# Options
require_pair = False          # True = ne copier que les fichiers qui ont à la fois image+mask (mêmes noms)
case_insensitive = True       # True pour comparer en minuscules
match_mode = "startswith"     # pour l'instant on utilise startswith (préfixe)
# ------------------------------------------

os.makedirs(output_images, exist_ok=True)
os.makedirs(output_masks, exist_ok=True)

def write_log(line):
    with open(log_file, "a", encoding="utf-8") as L:
        L.write(f"{datetime.now().isoformat()}  {line}\n")

# Nettoyage du log
with open(log_file, "w", encoding="utf-8") as L:
    L.write("=== Copy log ===\n")

# Charger et normaliser les préfixes
with open(prefixes_file, "r", encoding="utf-8") as f:
    raw = [ln.rstrip("\n\r") for ln in f]
prefixes = []
for ln in raw:
    if not ln:
        continue
    p = ln.strip().replace('\ufeff','')  # enlever BOM + espaces
    if p:
        prefixes.append(p)
# supprimer doublons en préservant l'ordre
seen = set()
prefixes = [x for x in prefixes if not (x in seen or seen.add(x))]

write_log(f"Chargé {len(prefixes)} préfixes depuis {prefixes_file}")

# Lister fichiers présents
def list_files(folder):
    try:
        return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    except Exception as e:
        write_log(f"ERREUR en listant {folder}: {e}")
        return []

images_files = list_files(images_dir)
masks_files  = list_files(masks_dir)

write_log(f"{len(images_files)} fichiers trouvés dans images_dir")
write_log(f"{len(masks_files)} fichiers trouvés dans masks_dir")

# Préparer maps lowercase si needed
if case_insensitive:
    images_map = {f.lower(): f for f in images_files}
    masks_map  = {f.lower(): f for f in masks_files}
    images_keys = list(images_map.keys())
    masks_keys  = list(masks_map.keys())
else:
    images_map = {f: f for f in images_files}
    masks_map  = {f: f for f in masks_files}
    images_keys = images_files
    masks_keys  = masks_files

prefixes_trouves = set()
prefixes_non_trouves = set(prefixes)  # initialement tous non trouvés

summary_rows = []

# Pour chaque préfixe, rechercher et copier
for prefix in prefixes:
    key = prefix.lower() if case_insensitive else prefix
    if case_insensitive:
        img_matches = [images_map[k] for k in images_keys if k.startswith(key)]
        mask_matches = [masks_map[k] for k in masks_keys if k.startswith(key)]
    else:
        img_matches = [f for f in images_files if f.startswith(prefix)]
        mask_matches = [f for f in masks_files if f.startswith(prefix)]

    images_found = len(img_matches)
    masks_found = len(mask_matches)
    images_copied = 0
    masks_copied = 0
    pairs_found = 0
    pairs_copied = 0

    if images_found or masks_found:
        prefixes_trouves.add(prefix)
        prefixes_non_trouves.discard(prefix)

        if require_pair:
            # ne garder que les noms communs aux deux dossiers
            pairs = sorted(set(img_matches).intersection(mask_matches))
            pairs_found = len(pairs)
            for fname in pairs:
                try:
                    shutil.copy2(os.path.join(images_dir, fname), os.path.join(output_images, fname))
                    shutil.copy2(os.path.join(masks_dir, fname), os.path.join(output_masks, fname))
                    pairs_copied += 1
                except Exception as e:
                    write_log(f"ERREUR copie paire {fname}: {e}")
        else:
            for fname in img_matches:
                try:
                    shutil.copy2(os.path.join(images_dir, fname), os.path.join(output_images, fname))
                    images_copied += 1
                except Exception as e:
                    write_log(f"ERREUR copie image {fname}: {e}")
            for fname in mask_matches:
                try:
                    shutil.copy2(os.path.join(masks_dir, fname), os.path.join(output_masks, fname))
                    masks_copied += 1
                except Exception as e:
                    write_log(f"ERREUR copie mask {fname}: {e}")

    summary_rows.append({
        "prefix": prefix,
        "images_found": images_found,
        "images_copied": images_copied,
        "masks_found": masks_found,
        "masks_copied": masks_copied,
        "pairs_found": pairs_found,
        "pairs_copied": pairs_copied
    })

# Écrire fichiers prefixes_trouves / non_trouves
with open(prefixes_trouves_file, "w", encoding="utf-8") as f:
    for p in sorted(prefixes_trouves):
        f.write(p + "\n")

with open(prefixes_non_trouves_file, "w", encoding="utf-8") as f:
    for p in sorted(prefixes_non_trouves):
        f.write(p + "\n")

# Écrire summary CSV
with open(summary_csv, "w", newline="", encoding="utf-8") as csvf:
    fieldnames = ["prefix", "images_found", "images_copied", "masks_found", "masks_copied", "pairs_found", "pairs_copied"]
    writer = csv.DictWriter(csvf, fieldnames=fieldnames)
    writer.writeheader()
    for row in summary_rows:
        writer.writerow(row)

# Récapitulatif console
total_images_found = sum(r["images_found"] for r in summary_rows)
total_masks_found  = sum(r["masks_found"] for r in summary_rows)
total_images_copied = sum(r["images_copied"] for r in summary_rows)
total_masks_copied  = sum(r["masks_copied"] for r in summary_rows)

print("=== RÉSUMÉ ===")
print(f"Préfixes demandés        : {len(prefixes)}")
print(f"Préfixes trouvés         : {len(prefixes_trouves)} (-> {prefixes_trouves_file})")
print(f"Préfixes non trouvés     : {len(prefixes_non_trouves)} (-> {prefixes_non_trouves_file})")
print(f"Images trouvées total    : {total_images_found}")
print(f"Masks  trouvés total     : {total_masks_found}")
print(f"Images copiées total     : {total_images_copied}")
print(f"Masks  copiés total      : {total_masks_copied}")
print(f"Résumé détaillé écrit dans: {summary_csv}")
print(f"Log détaillé             : {log_file}")

write_log("Script terminé normalement.")
