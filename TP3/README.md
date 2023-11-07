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