import pandas as pd

# === CONFIGURATION ===
# Chemin du fichier des réponses (le fichier filtré que tu as déjà)
fichier_source = "reponses_filtrees.txt"  # <-- change le nom si besoin
fichier_sortie = "labels.csv"              # le fichier final à créer

# === LECTURE DU FICHIER FILTRÉ ===
print(f"📂 Lecture du fichier : {fichier_source}")
df = pd.read_csv(fichier_source, sep="\t")

print("Colonnes détectées :", df.columns.tolist())

# === RENOMMAGE DES COLONNES AU BON FORMAT ===
# On cherche la colonne qui correspond à l’identifiant patient
if "patient" in df.columns:
    df = df.rename(columns={"patient": "patient_id"})
elif "Patient" in df.columns:
    df = df.rename(columns={"Patient": "patient_id"})

# On cherche la colonne correspondant à la réponse (0/1)
if "Treatment response" in df.columns:
    df = df.rename(columns={"Treatment response": "response"})
elif "Treatment_response" in df.columns:
    df = df.rename(columns={"Treatment_response": "response"})
elif "response" not in df.columns:
    raise ValueError("⚠️ Aucune colonne 'response' trouvée dans le fichier source.")

# === VÉRIFICATION DU CONTENU ===
print("\nAperçu des 5 premières lignes :")
print(df.head(), "\n")

# === SAUVEGARDE DU FICHIER FINAL ===
df.to_csv(fichier_sortie, index=False)
print(f"✅ Fichier '{fichier_sortie}' créé avec succès !")
print(f"👉 Colonnes finales : {df.columns.tolist()}")
print(f"📁 Enregistré dans le dossier courant ({fichier_sortie})")
