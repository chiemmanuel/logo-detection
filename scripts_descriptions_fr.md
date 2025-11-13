### Description des Scripts
1) ### link_construct.py

Ce script se connecte à une base de données MySQL pour récupérer les enregistrements médias d’une étude donnée, les filtre par date et génère des liens de téléchargement pour chaque fichier média.
Il distingue les images et les vidéos, en créant automatiquement les dossiers appropriés pour chacune.

# Fonctionnalités principales :

- Interroge la base de données pour toutes les publications associées à une étude et une plage de dates données.
- Génère des liens AWS S3 pour les médias non-Twitter et conserve les URL d’origine pour les médias Twitter.
- Crée des dossiers nommés selon le type de média et l’ID de l’étude (ex : images_53_10, videos_53_10).
- Sauvegarde tous les liens récupérés dans des fichiers texte (aws_links.txt et twitter_links.txt) à l’intérieur de chaque dossier.

# Configuration principale :

study_id : l’ID de l’étude à traiter (par défaut "53").

START_DATE et END_DATE : définissent la période de récupération des médias.

TWITTER_PLATFORM_ID : identifie les médias Twitter pour conserver leurs URLs d’origine.

# Utilisation :

Exécutez le script après avoir défini le bon ID d’étude et la plage de dates.

Il effectuera automatiquement :
La requête de la base de données
La création des dossiers nécessaires
La génération et le stockage des liens AWS et Twitter correspondants

#########################################################################################################################

2) # downloader.py

Ce script gère le téléchargement et le prétraitement des images et vidéos d’une étude donnée.
Il peut être utilisé à la fois pour la préparation de datasets (entraînement YOLO) et pour l’analyse de médias (détection).

# Fonctionnalités principales :

- Lit les URLs générées par link_construct.py depuis aws_links.txt et twitter_links.txt.
- Télécharge toutes les images ou vidéos associées à un study_id spécifique.
- Gère automatiquement les fichiers hébergés sur AWS et Twitter.
- Sauvegarde les fichiers téléchargés dans des dossiers dédiés selon leur usage.
- Journalise les échecs de téléchargement pour réessai ultérieur.
- Peut extraire des frames de courtes vidéos (pour créer des datasets).

# Configuration principale :

study_id : l’ID de l’étude à traiter.

purpose :

  "dataset" → crée des datasets prêts pour YOLO (extrait les frames et gère les longues vidéos).

  "media_analysis" → télécharge les médias sans extraire les frames.

frame_threshold : détermine si une vidéo doit être découpée en frames ou déplacée vers le dossier videos_to_be_cut.

# Aperçu des fonctions :

  download_images_for_study(study_id, purpose="dataset") : télécharge les images pour l’étude spécifiée.

  download_and_process_videos(study_id, purpose="dataset", frame_threshold=150) : télécharge et traite les vidéos.

  retry_failed_downloads(study_id) : relit le journal d’erreurs et retente les téléchargements échoués.

# Utilisation :

Assurez-vous que link_construct.py a été exécuté pour générer les fichiers aws_links.txt et twitter_links.txt.

Ouvrez downloader.py et définissez les variables study_id et purpose.

Exécutez : python downloader.py

Le script créera les dossiers requis, téléchargera les médias et enregistrera les erreurs éventuelles.

############################################################################################################################

3) # image_extraction.py

Ce script offre un contrôle flexible pour extraire des frames de vidéos selon différents modes, stratégies et plages temporelles.
Il est principalement utilisé pour transformer les vidéos téléchargées (via downloader.py) en ensembles d’images représentatives pour les datasets ou l’analyse visuelle.

# Fonctionnalités principales :

Lit un fichier vidéo et récupère ses métadonnées de base : FPS, nombre total de frames et durée.

- Permet plusieurs modes d’extraction :

all → toutes les frames

percentage → un pourcentage des frames

step → une frame toutes les n images

time → une frame toutes les x secondes

Peut se limiter à une plage temporelle spécifique (start_sec, end_sec).

- Deux stratégies de sélection :

equal → frames réparties uniformément

random → frames choisies aléatoirement

- Sauvegarde les frames extraites avec des noms descriptifs.

- Affiche la progression et un résumé des performances.

# Configuration principale :

video_path : chemin complet vers la vidéo à traiter.

output_folder : dossier de destination des frames extraites.

mode : mode d’extraction choisi (all, percentage, step, time).

value : valeur associée au mode (pourcentage, pas, ou intervalle de temps).

strategy : méthode de sélection (equal ou random).

start_sec / end_sec : plage de temps optionnelle.

# Aperçu des fonctions :

get_video_info(video_path) : récupère les infos vidéo.

video_to_frames(...) : gère la logique principale d’extraction.

apply_strategy(frames_list, strategy, n_target) : applique la stratégie de sélection.

extract_frames_standard(...) : sauvegarde les frames extraites sur disque.

# Utilisation :

- Vérifiez que la vidéo cible existe (ex : dans videos_to_be_cut).

- Exécutez : python image_extraction.py

- Suivez les instructions à l’écran pour choisir :

- Le mode d’extraction (pourcentage, step, etc.)

- La plage temporelle éventuelle

- La stratégie (equal ou random)

- Les frames seront automatiquement enregistrées dans le dossier configuré, avec un résumé final.

#############################################################################################################################

4) # image_uploader.py

Ce script automatise le téléversement d’images locales vers un projet Label Studio via son API.
Il gère le téléversement par lots et rafraîchit automatiquement le jeton d’accès.

# Fonctionnalités principales :

- Parcourt un dossier local (ex : dataset généré par downloader.py ou image_extraction.py).
- Identifie tous les fichiers image (.jpg, .jpeg, .png).
- Construit le chemin relatif pour chaque image afin de respecter la configuration de Label Studio.
- Envoie une requête POST à l’API Label Studio pour téléverser chaque image.
- Rafraîchit automatiquement le jeton d’accès via la fonction refresh_access_token().
- Limite le nombre d’images téléversées à max_uploads.
- Journalise le statut des téléversements et affiche les temps de début et de fin.

# Configuration principale :

local_folder : chemin vers le dossier contenant les images.

PROJECT_ID : ID du projet cible dans Label Studio.

max_uploads : nombre maximum d’images à téléverser par exécution.

API_TOKEN : jeton d’accès à l’API Label Studio (géré automatiquement).

API_URL : endpoint pour la création de tâches dans Label Studio.

# Pré-requis :

- Avant l’exécution, définissez ces variables d’environnement dans le terminal :

$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "C:\Users\JuniorCHIEMMANUELNGU\Desktop\test"

- Lancez Label Studio :
label-studio start

- Dans Label Studio, configurez le stockage cible pour pointer vers le dossier contenant les images (ex : "C:\Users\JuniorCHIEMMANUELNGU\Desktop\test\dataset_53").

# Utilisation :

- Définissez les variables d’environnement comme indiqué.

- Assurez-vous que Label Studio est en cours d’exécution et que le projet existe.

- Mettez à jour local_folder et PROJECT_ID dans le script.

- Exécutez : python image_uploader.py

- Le script téléversera les images et affichera la progression dans la console.

- Les échecs éventuels seront visibles dans la sortie console.

########################################################################################################################################

5) # yolo.py

Ce script entraîne un modèle YOLO sur un dataset préparé et l’évalue après l’entraînement.

# Fonctionnalités principales :

- Charge un modèle YOLO (ex : yolo11n.pt) comme base pour le fine-tuning.
- Entraîne le modèle sur un dataset défini dans un fichier data.yaml.
- Gère les paramètres d’entraînement : nombre d’époques, taille des images, batch size, et nom d’expérience.
- Valide le modèle entraîné sur le jeu de validation et affiche les métriques.
- Affiche les timestamps de début et de fin ainsi que la durée totale d’entraînement.

# Configuration principale :

model : chemin vers les poids de base YOLO (.pt).

data.yaml : fichier de configuration du dataset.

epochs : nombre d’époques (ex : 100).

imgsz : taille d’image d’entrée (ex : 640).

batch : taille du batch par itération (ex : 8).

name : nom de l’expérience / dossier de sauvegarde.

# Utilisation :

- Préparez le dataset selon la structure YOLO : dossiers train et val avec annotations.

- Vérifiez que data.yaml pointe vers les bons chemins et noms de classes.

- Définissez le modèle de base : YOLO("yolo11n.pt").

- Exécutez : python yolo.py

Après l’entraînement, les métriques sont affichées et les meilleurs poids sauvegardés dans runs/train/<name>/weights/best.pt.

#############################################################################################################################

6) # image_detection.py

Ce script applique un modèle YOLO entraîné pour détecter les objets dans toutes les images d’un dossier spécifié.
Il sauvegarde à la fois les images annotées et les résultats structurés au format JSON.

# Fonctionnalités principales :

- Charge un modèle YOLO depuis les poids spécifiés (ex : best.pt).
- Parcourt toutes les images (.jpg, .jpeg, .png) d’un dossier cible.
- Exécute la détection sur chaque image.
- Sauvegarde les images annotées dans detections_images.
- Sauvegarde les résultats JSON dans detection_jsons.
- Affiche le temps de traitement par image et les statistiques globales.

# Configuration principale :

images_folder : chemin vers le dossier contenant les images.

model_weights_path : chemin vers les poids YOLO entraînés.

# Utilisation :

- Préparez un dossier contenant les images à traiter.

- Assurez-vous d’avoir les poids du modèle YOLO.

- Mettez à jour images_folder et model_weights_path.

- Exécutez : python image_detection.py

Les images annotées et fichiers JSON seront enregistrés automatiquement dans les dossiers correspondants.

#############################################################################################################################

7) video_analysis_with_tracker.py

Ce script applique un modèle YOLO entraîné pour détecter les objets dans toutes les vidéos d’un dossier spécifié.
Il utilise le suivi SORT pour assurer une cohérence temporelle mais conserve uniquement les boîtes et labels YOLO. Seules les détections au-dessus d’un seuil de confiance sont conservées.

# Fonctionnalités principales :

- Charge un modèle YOLO depuis les poids spécifiés.

- Parcourt toutes les vidéos (.mp4, .mov, .avi) d’un dossier cible.

- Exécute la détection image par image.

- Filtre les détections avec une confiance < 0.35.

- Applique le suivi SORT pour stabiliser les détections dans le temps.

- Sauvegarde les vidéos annotées dans detected_videos_with_tracking_fine_tuneV2.

- Affiche le temps de traitement moyen par frame et par vidéo.

# Configuration principale :

videos_folder : dossier contenant les vidéos à traiter.

model_weights_path : chemin vers les poids YOLO entraînés.

imgsz : taille d’image d’entrée pour YOLO (par défaut 640).

# Utilisation :

- Préparez un dossier contenant les vidéos à analyser.

- Assurez-vous d’avoir les poids YOLO entraînés.

- Mettez à jour videos_folder et model_weights_path.

- Exécutez : python video_analysis_with_tracker.py

Les vidéos annotées seront enregistrées automatiquement dans detected_videos_with_tracking_fine_tuneV2.

Le script affichera un résumé du temps de traitement total et des statistiques par vidéo.