# Partie 1 Prise en main de yolo dans le module dnn d’opencv
## Question 1 .1 Quelles sont les classes reconnues par le réseau ? tester sur au 20 images par classes (conserver ces images et les déposer sur le ecampus). Tester les 2 réseaux yolo et tiny-yolo

On retrouve les classes dans le fichier `coco.names`
<details>
    <summary>Liste des classes</summary>
    <pre>
        person 
        bicycle
        car
        motorbike
        aeroplane
        bus
        train
        truck
        boat
        traffic light
        fire hydrant
        stop sign
        parking meter
        bench
        bird
        cat
        dog
        horse
        sheep
        cow
        elephant
        bear
        zebra
        giraffe
        backpack
        umbrella
        handbag
        tie
        suitcase
        frisbee
        skis
        snowboard
        sports ball
        kite
        baseball bat
        baseball glove
        skateboard
        surfboard
        tennis racket
        bottle
        wine glass
        cup
        fork
        knife
        spoon
        bowl
        banana
        apple
        sandwich
        orange
        broccoli
        carrot
        hot dog
        pizza
        donut
        cake
        chair
        sofa
        pottedplant
        bed
        diningtable
        toilet
        tvmonitor
        laptop
        mouse
        remote
        keyboard
        cell phone
        microwave
        oven
        toaster
        sink
        refrigerator
        book
        clock
        vase
        scissors
        teddy bear
        hair drier
        toothbrush
    </pre>
</details>


<details>
    <summary>Détail des images yolo</summary>
    <img src="./img/result1_yolo.png" alt="Résultat de l'image 1">
    <img src="./img/result2_yolo.png" alt="Résultat de l'image 2">
    <img src="./img/result3_yolo.png" alt="Résultat de l'image 3">
    <img src="./img/result4_yolo.png" alt="Résultat de l'image 4">
    <img src="./img/result5_yolo.png" alt="Résultat de l'image 5">
    <img src="./img/result6_yolo.png" alt="Résultat de l'image 6">
    <img src="./img/result7_yolo.png" alt="Résultat de l'image 7">
    <img src="./img/result8_yolo.png" alt="Résultat de l'image 8">
    <img src="./img/result9_yolo.png" alt="Résultat de l'image 9">
    <img src="./img/result10_yolo.png" alt="Résultat de l'image 10">
    <img src="./img/result11_yolo.png" alt="Résultat de l'image 11">
    <img src="./img/result12_yolo.png" alt="Résultat de l'image 12">
    <img src="./img/result13_yolo.png" alt="Résultat de l'image 13">
    <img src="./img/result14_yolo.png" alt="Résultat de l'image 14">
</details>

<details>
    <summary>Détail des images tiny-yolo</summary>
    <img src="./img/result1_tiny-yolo.png" alt="Résultat de l'image 1">
    <img src="./img/result2_tiny-yolo.png" alt="Résultat de l'image 2">
    <img src="./img/result3_tiny-yolo.png" alt="Résultat de l'image 3">
    <img src="./img/result4_tiny-yolo.png" alt="Résultat de l'image 4">
    <img src="./img/result5_tiny-yolo.png" alt="Résultat de l'image 5">
    <img src="./img/result6_tiny-yolo.png" alt="Résultat de l'image 6">
    <img src="./img/result7_tiny-yolo.png" alt="Résultat de l'image 7">
    <img src="./img/result8_tiny-yolo.png" alt="Résultat de l'image 8">
    <img src="./img/result9_tiny-yolo.png" alt="Résultat de l'image 9">
    <img src="./img/result10_tiny-yolo.png" alt="Résultat de l'image 10">
    <img src="./img/result11_tiny-yolo.png" alt="Résultat de l'image 11">
    <img src="./img/result12_tiny-yolo.png" alt="Résultat de l'image 12">
    <img src="./img/result13_tiny-yolo.png" alt="Résultat de l'image 13">
    <img src="./img/result14_tiny-yolo.png" alt="Résultat de l'image 14">
</details>

## Question 1.2 A quoi sert le ou les thresholds ?

Dans notre cas le thresholds sert a définir un seuil. On en définit deux différent :
- Le seuil de confiance : 
- - Valeur de probabilité minimale requise pour afficher la prédiction a l'écran. 
- - Permet d'éviter de polluer avec des prédictions peu fiable
- - Dans notre codé égale a 20%
- Le seuil de suppression non maximal
- - Valeur que l'on applique a l'aglo de supression non maximale entre des predictions sur une même entité
- - Permet de filtrer les prédictions redondante sur un même sujet afin d'en garder la plus fiable
- - Dans notre code définie a 40%

## Question 1.3 Quelles sont les fichiers utilisés liés au réseau de neurones  , que contiennent ils précisément ?

Pour les réseaux `yolov3` et `yolov3-tiny`, on peut distinguer deux types de fichiers pour chacun : 
- weights
- - Contient les poids appris par le modèle yolov3 lors de son entraînement sur un ensemble de données volumineux.
- - On utilise ces poids lors de l'initialisation de l'inférence, ce qui permet de faire des prédictions sur les images
- - Fichier potentiellement assez volumineux pouvant contenir des millions de poids.
- config
- - Contient l'architecture sur réseau
- - On y retrouve 
- - - Des informations directs sur le réseau
- - - Les spécifications des couches du réseaux

## Question 1.4 Quelle est l’architecture du réseau yolov3 et yolov3 tiny  , expliquer les différentes couches du réseaux .  Lesquelles dépendent du nombre de classes ?

YOLOV3 Architecture : 
- Dimension : 608x608 et 3 canaux (RGB)
- Convolution et extraction des caractéristiques
- - Couches de convolution
- - - fonction d'activation `leaky`
- - Suivi de couches `shortcut`
- - - fonction d'activation `linear`
- 80 classes 
- 3 couches `yolo` avec des masques et ancres propre

YOLOV3-TINY 
- Dimension : 416x416 et 3 canaux (RGB)
- Convolution et extraction des caractéristiques
- - Couches de convolution
- - - fonction d'activation `leaky`
- - Suivi de couches `maxpool`
- - - taille 2
- 80 classes 
- 2 couches `yolo` avec des masques et ancres propre

## Question 1.5 Est-ce que vous trouvez d’autres modèles pré entrainés proposés aux téléchargement ? Si oui tester les 
Il semble y en avoir un adapté à l'utilisation d'une webcam  
En branchant une webcam, le code va lire le flux et lui appliquer une detection d'objet en temps réél.  
Ainsi en branche une webcam et en executant le code on voit une fenetre avec le retour de la webcam avec une detection d'objet dessus.

# Partie 2 Interfaçage Python

## Question 2.1 On désire une option pour n'afficher qu'une liste de classe configurée en ligne de commande et mettre en place des  boîtes englobantes avec un effet de transparence ou équivalent pour mettre les objets detectés en surbrillance ou équivalents. Adapter  le code pour travailler sur un  flux webcam ou des images ( En changeant la ligne de commande) .
Voir le fichier `example_official_open_cv_ultalytics_custom`
On lance le fichier en selectionnant les classes en argument : `python3 example_official_open_cv_ultalytics_custom.py --model yolov5nu.onnx --img my_images/img1.jpg --classes person bird`
![alt text](./img/screen8.png)

# Partie 3 Préparation à l’entraînement votre propre classifier

L'idée est maintenant de comprendre comment créer votre propre réseau . L’objectif est de créer votre classifier sur 3 ou 4 classes (à vous de choisir parmi les objets proposés ou autres). Il faut prévoir idéalement dans les 200 images labélisés par classe (ne pas hésiter à mixer les fonds et les éclairages) mais on peut avoir des performances corrects à partir de 40.

## Créer le dataset et le labéliser sur roboflow
![alt text](./img/screen1.png)


## Vous exporterez à la fin le dataset au format yolo et au format tensorflow. Que contienne ses archives ? (vous renderez ces fichiers sur le e-campus)
YOLO v3
```
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="RXLg6HXDr2aPKiDVgGsJ")
project = rf.workspace("robia").project("tp3_ia_girafe_elephant")
dataset = project.version(1).download("yolokeras")

https://app.roboflow.com/ds/L7DPJteV1u?key=CppOvNlDJ8
```

TF 
```
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="RXLg6HXDr2aPKiDVgGsJ")
project = rf.workspace("robia").project("tp3_ia_girafe_elephant")
dataset = project.version(1).download("tensorflow")

https://app.roboflow.com/ds/ghe92TRQH2?key=SjSmVEQs2x
```

Ses archices contiennent 3 dossier principaux : 
- dossier test (images utilisé pour les test)
- dossier train (images utilisé pour l'entrainement)
- dossier valid (images utilisées pour la validation)
- Chaque dossier contient les images correspondantes avec un fichier d'annotation (contient les infos sur les box d'annotation) et un fichier classe (contient les classes annotées)
- Les images sont réparties avec un ratio 70%(train)-20%(valid)-10%(test)
- Fichiers README
## Vous pourrez ensuite lancer l’apprentissage sur roboflow (vous avez 3 credits associés à la création de votre compte). Tester le résultat ? Etes vous satisfait du résultat , expliquer pourquoi et comment vous tester.
C'est fait je l'ai bien lancé mais impossible de finir l'entrainement de modèe (même après 3h)

## Comment peut-on récupérer le réseau généré ?  Comment l’utiliser ? Pourquoi ce choix ?
On aurait pu exécuter directement notre modèle hébergé chez roboflow et l'utiliser sur des images qu'on aurait pu uploader sur le site et le tester donc directment sur le site. Attention roboflow une fois que roboflow a entrainé notre modèle, il est hebergé chez eux. On doit par la suite si on veut bcp l'utiliser payer pour "louer" notre modèle et pouvoir l'utilsier 

## On procédera ensuite à l’export du dataset sous ultralytics puis au training en utilisant le format YOLOv5lu.
![alt text](./img/screen2.png)  
Puis suivre ce qui suit  
Puis on obtient cette courbe représentative du déroulement de l'entrainement  
![alt text](./img/screen3.png)  

## On testera le résultat sur le site , est ce meilleur ou moins bon que le résultat de roboflow ?
![alt text](./img/screen4.png)  
![alt text](./img/screen5.png)  
![alt text](./img/screen6.png)  

## Dans les options de déploiement de ultralytics, on pourra ensuite générer et télécharger le format onnx .
Voir fichier généré 

## Vous pourrez finalement le tester sous opencv directement. en utilisant le code d’exemple fourni sur le e-campus. Vous devrez en particulier trouver une solution pour récupérer le nom de vos classes, expliquer comment ?

Pour que ca fonctionne bien, il a fallu, changer les noms des classes pour que celle ci corresponde au predictions voulues et au model.  
Ainsi il a fallu récupérer le fichier yaml généré lors du train du model ultralytics ou on aurait pu aussi changer les noms de classe en dur.

## Cela est il fonctionnelles, comment procéder à tests ? Etes-vous satisfaits ? 
On obtient le résultat suivant 
![alt text](./img/screen7.png)  
Il suffit d'exécuter le fichier python en ayant changer le noms des classes ainsi que le moteur onnx.  
On voit que ca marche pas super bien mais au moins ca reconnait un peu. Certes, ca reconnait moins que sur ultralytics, surement du a l'outil qui est plus adapté a son propre export que opencv  