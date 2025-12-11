### Notes importantes

- Avant d’utiliser link construct, assurez-vous toujours que les mois utilisés dans la requête correspondent bien aux noms des dossiers qui seront créés, surtout si la requête s’exécute sur un sous-ensemble précis des données disponibles
    - Par exemple, si vous souhaitez créer les liens pour l’étude 60 mais uniquement pour le mois d’octobre, l’idéal serait de renommer images_{study_id} et videos_{study_id} en images_{study_id}_10 et videos_{study_id}_10, où 10 correspond au mois concerné (si nécessaire)
    - Modifier cela partout dans le fichier link construct via Ctrl + F puis Remplacer
    - Renommer les dossiers n’est pas obligatoire, mais peut éviter des conflits si vous créez plusieurs sous-ensembles pour une même étude

- Après avoir modifié link construct, appliquez les mêmes modifications aux noms des dossiers contenant les liens dans les fonctions du downloader dans downloader.py, car ces fonctions dépendent des noms exacts des dossiers images et videos
    - Là encore, utiliser Ctrl + F puis Remplacer
    - L’idéal est d’appliquer ces changements immédiatement après les modifications dans link construct afin de garantir un pipeline fonctionnel

- Avant de lancer une instance de Label Studio, définissez toujours les variables d’environnement telles qu'indiquées dans scripts_descriptions_fr, sans cela, les images locales dans les projets ne seront pas visibles.

- Avant de lancer Label Studio, assurez-vous que le dossier d’origine depuis lequel vous avez importé les images dans le projet est dans le même état que lors de l’upload
    - Aucun nouveau sous-dossier
    - Uniquement les images
Sinon, les images ne seront pas visibles dans le projet

- Après l’annotation et l’export, créez idéalement un nouveau dossier temporaire contenant:
    - un sous-dossier images avec toutes les images du dossier d’origine utilisé pour l’upload
    - un sous-dossier labels provenant de l’export
- Depuis ce dossier temporaire, lancez create_dataset.py pour générer automatiquement le dataset scindé, sans altérer la structure du dossier d’origine au cas où vous souhaitez retravailler plus tard sur le projet.