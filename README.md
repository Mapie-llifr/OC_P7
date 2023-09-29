# Dashboard Prêt à Dépenser
Interface utilisant Streamlit pour afficher les réponses aux requetes faites vers une application permettant d'utiliser un modèle de prédiction de remboursement d'emprunt.    

Ce programme a été créé dans le cadre de la formation Data Scientist chez OpenClassrooms.     
Il se base sur les données d'une compétition Kaggle : Home Credit.  

## Fichiers
Programme écrit en language python     
Les données : dans Docs_projet7   
- small_model_final.csv : données de moins de 800 variables sur 3000 clients.  
- HomeCredit_columns_description.csv : tableau de définitions des noms de variables dans la colonne 'Row'.  

Autres fichiers :      
- dashboard_streamlit : fichier principal, utilisant la librairie Streamlit pour l'affichage des composants web  

## Requirements
Environnement nécessaire au fonctionnement de l'application :       
- Python, version 3.10  



- matplotlib==3.7.1
- numpy
- pandas==1.5.3
- requests==2.31.0
- streamlit==1.26.0


## Fonctionnement
L'utilisateur saisit l'identification du client, la valeur de 'SK_ID_CURR' dans le tableau de données.     
L'interface envoie des requetes paramétrées avec l'identification du client vers une API qui utilise un modèle de prédiction de la probabilité de non remboursement, préalablement entrainé sur les demandes de prêt de l'ensemble de la clientèle.        
L'interface reçoit la probabilité de non remboursement du client et affiche en pourcentage la probabilité de remboursement.       
L'interface reçoit l'interprétation de la probabilité de non remboursement et affiche si le pret est accepté, risqué ou refusé.       
L'interface reçoit les dix variables ayant le plus influencées la prédiction et affiche un diagramme en barre. Les variables en faveur de l'acceptation du pret sont en haut, celles en faveur du refus sont en bas.      


L'interface affiche aussi les données d'application du client sélectionné.  


L'utilisateur peut saisir une des variables de la demande de pret.      
L'interface affiche alors la distribution de cette variable sur l'ensemble des données, le positionnement du client dans cette distribution ainsi que la description de la variable telle qu'elle apparait dans le tableau de définition des noms de variables.      

## API
L'API est disponible : https://github.com/Mapie-llifr/OpenClassroomsProject7       
utilisable   @          https://pretadepenser-06a8123a2ba8.herokuapp.com/

## Déploiement
Avec Streamlit :      https://pret-a-depenser.streamlit.app/
