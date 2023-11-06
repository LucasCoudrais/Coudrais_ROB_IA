# 1
## 1.a Expliquer la différence entre la classification d’image, la détection d’image la segmentation d’images
- Classification image 
- - Attribution d'une etiquette à une image en fonction de son contenu selon une règle prédéfinie
- - On s'attend a n'avoir qu'un seul sujet a classifier dans l'image qu'on classifie 
-  Détection d'image
- - Attribution d'étiquette a une partie d'une imageau sein d'une image. 
- - Identification de la position et la nature de plusieurs sujet dans une image
- Segmentation d'image
- - Division d'une image en plusieurs régions de pixels pour les classifier 
- - L'image est divisé en plusieurs régions, tout les pixels de l'image sont attribué a une image de sorte a segmenter toute l'image en plusieurs régions d'identification  

![alt text](./img/screen1.png)

## 1.b Quelles sont les grandes solutions de détection d’objets
On peut séparer les solution de détection d'objet en 2 grandes classes, les one-stage et les two-stage
- Deux étage
- - Proposent d'abord des régions approximatives pour les objets
- - Puis effectuent la classification d'objets et la régression des boîtes englobantes
- - Précision de détection élevée, mais plus lents
- Un étage
- - Prédisent directement les boîtes englobantes sans l'étape de proposition de régions
- - Plus rapides pour les applications en temps réel, mais moins performants pour les objets irréguliers ou de petite taille  

Grands exemples : 
- YOLO (1 étage)
- SSD (1 étage)
- RetinaNet (1 étage)
- R-CNN (2 étages)
- - Faster R-CNN (2 étages)
- - Mask R-CNN (2 étages)
- - Granulated R-CNN (2 étages)  

### On s'interessera plus particulièrement au SSD (single-shot detector)  
Il s'agit d'une approche a prise de vue unique (one-stage), permettant de détecter tous les objets dans une image en une seule passe.  
Le SSD divise l'image en une grille, où chaque cellule de la grille est responsable de détecter les objets dans sa région.  
On peut utiliser une grille 4x4 pour détecter des objets plus petits, une grille 2x2 pour des objets de taille moyenne, et une grille 1x1 pour des objets qui couvrent l'ensemble de l'image.

# 2 
On s’intéresse à l’exemple suivant  
https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/hub/tutorials/tf2_object_detection.ipynb

## 2.a Quelles sont les classes reconnues par le réseau ?
Pour le réseau `CenterNet HourGlass104 Keypoints 512x512`, nous observons les classes suivantes dans notre résultat :
- person
- - On voit meme un squelette de la personne si assez de détail
- kite
- surfboard
Par ailleurs on peut voir sur la dac de ce modèle qu'il reconnaitra toute les classe du dataset COCO 2017
La liste des classes de ce dataset se trouve dans le fichier `coco-2017-labels.txt`

## 2.b Quelle partie du code correspond au chargement du modèle de réseau.Quelles sont les modèles proposés
Sélection du modèle :  
```
#@title Model Selection { display-mode: "form", run: "auto" }
model_display_name = 'CenterNet HourGlass104 Keypoints 512x512' # @param ['CenterNet HourGlass104 512x512','CenterNet HourGlass104 Keypoints 512x512','CenterNet HourGlass104 1024x1024','CenterNet HourGlass104 Keypoints 1024x1024','CenterNet Resnet50 V1 FPN 512x512','CenterNet Resnet50 V1 FPN Keypoints 512x512','CenterNet Resnet101 V1 FPN 512x512','CenterNet Resnet50 V2 512x512','CenterNet Resnet50 V2 Keypoints 512x512','EfficientDet D0 512x512','EfficientDet D1 640x640','EfficientDet D2 768x768','EfficientDet D3 896x896','EfficientDet D4 1024x1024','EfficientDet D5 1280x1280','EfficientDet D6 1280x1280','EfficientDet D7 1536x1536','SSD MobileNet v2 320x320','SSD MobileNet V1 FPN 640x640','SSD MobileNet V2 FPNLite 320x320','SSD MobileNet V2 FPNLite 640x640','SSD ResNet50 V1 FPN 640x640 (RetinaNet50)','SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)','SSD ResNet101 V1 FPN 640x640 (RetinaNet101)','SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)','SSD ResNet152 V1 FPN 640x640 (RetinaNet152)','SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)','Faster R-CNN ResNet50 V1 640x640','Faster R-CNN ResNet50 V1 1024x1024','Faster R-CNN ResNet50 V1 800x1333','Faster R-CNN ResNet101 V1 640x640','Faster R-CNN ResNet101 V1 1024x1024','Faster R-CNN ResNet101 V1 800x1333','Faster R-CNN ResNet152 V1 640x640','Faster R-CNN ResNet152 V1 1024x1024','Faster R-CNN ResNet152 V1 800x1333','Faster R-CNN Inception ResNet V2 640x640','Faster R-CNN Inception ResNet V2 1024x1024','Mask R-CNN Inception ResNet V2 1024x1024']
model_handle = ALL_MODELS[model_display_name]

print('Selected model:'+ model_display_name)
print('Model Handle at TensorFlow Hub: {}'.format(model_handle))
```

Chargement du modèle (hub correspond a la librarie `tensorflow_hub`) : 
```
print('loading model...')
hub_model = hub.load(model_handle)
print('model loaded!')
```

On retrouve les modèles proposée dans le code de sélection du modèle. Ils sont dans le tableau de paramètre de la variable `model_display_name`  

## 2.c Quelles sont les structures des modèles de réseaux sous jacents ?
Voici la structure de l'ensemble des modèles de la liste du dessus : 
- CenterNet
- - HourGlass104 
- - - Keypoints
- - - - 512x512
- - - - 1024x1024
- - - 512x512
- - - 1024x1024
- - Resnet50 ou Resnet101
- - - V1 FPN ou V2 
- - - - Keypoints
- - - - - 512x512
- - - - 512x512
- EfficientDet
- - D0 ... D7
- - - 512x512 ... 1536x1536
- SSD
- - MobileNet
- - - V1 ou V2
- - - FPN ou FPNLite
- - - 320x320 ou 640x640
- - ResNet50 ou ResNet101 ou ResNet152
- - - V1 FPN 640x640 ou 1024x1024
- Faster R-CNN
- - ResNet50 ou ResNet101 ou ResNet152
- - - V1
- - - - 640x640 ou 1024x1024 ou 800x1333
- - Inception ResNet
- - - V2 640x640 ou 1024x1024
- Mask R-CNN 
- - Inception ResNet V2 1024x1024

## 2 .d Tester sur une douzaine d’images de votre choix (Essayer sur des images contenant le plus de classes possibles reconnus) et faites un tableau comparatif

<details>
    <summary>Détail des tests</summary>
    <img src="./img/img1_result.png" alt="Résultat de l'image 1">
    <img src="./img/img2_result.png" alt="Résultat de l'image 2">
    <img src="./img/img3_result.png" alt="Résultat de l'image 3">
    <img src="./img/img4_result.png" alt="Résultat de l'image 4">
    <img src="./img/img5_result.png" alt="Résultat de l'image 5">
    <img src="./img/img6_result.png" alt="Résultat de l'image 6">
    <img src="./img/img7_result.png" alt="Résultat de l'image 7">
    <img src="./img/img8_result.png" alt="Résultat de l'image 8">
    <img src="./img/img9_result.png" alt="Résultat de l'image 9">
    <img src="./img/img10_result.png" alt="Résultat de l'image 10">
    <img src="./img/img11_result.png" alt="Résultat de l'image 11">
    <img src="./img/img12_result.png" alt="Résultat de l'image 12">
</details>

| Image| Type| Nombre reconnaissance| Reconnaissance fausse| Reconnaissance "vide"|
|------|------------|-------|------------|--|
| 1| Vie a la campagne| 25| 5| 4| 
| 2| Ville| 26| 3| 0| 
| 3| Moment convivial| 21| 1| 2| 
| 4| Liste animal| 6| 3| 0| 
| 5| Plateau aliment| 8| 2| 0| 
| 6| Liste transport| 17| 0| 0| 
| 7| Nature morte| 10| 4| 0| 
| 8| Page e-commerce vetement| 8| 0| 0| 
| 9| Equipement de sport| 1| 0| 0| 
| 10| Industrie robot| 4| 4| 0| 
| 11| Robot| 4| 2| 0| 
| 10| Intérieur maison| 12| 0| 1| 


# 3
## 3.a A quoi sert Tensorflow Hub, et y a t il des solutions équivalentes ?
TensorFlow Hub est un depôt de modèles de machine learning entrainés que l'on peut optimiser et déployer rapidement et n'importe ou. Cet outil met a disposition un large choix de modèle entre autre adapté pour de la detection d'objet, exploitable avec TensorFlow.

Voici une liste de solution semblables à TensoFlow
- PyTorch Hub
- Model Zoo de Apache MXNet
- OpenAI GPT Models (outils GPT/BERT)
- Hugging Face Transformers (outils GPT/BERT)
- Scikit-learn
- Keras

## 3.b Combien trouve t’on sur tensorflow hub de réseaux de detection d’objets ?
On se base sur cette doc qui semble reprendre la liste que l'on avait dans le collab. Elle rescence tout les model de detection d'objet TF2 entrainée sur le dataset COCO 2017.
https://tfhub.dev/tensorflow/collections/object_detection/1  
Il semble y avoir 39 réseaux de detection d'objet.

## 3.c Quelles sont les architectures de ces réseaux ?
Architecture des reseaux du collab deja décrites au dessus. La liste des reseaux du collab est la même que celle de la question précédente (2c).
En se basant sur la doc on peut voir les suivants : 
- CenterNet 
- Faster R-CNN
- EfficientDet
- SSD Mobilenet V2
- RetinaNet
- Mask R-CNN
- SSD Mobilenet V1

## 3.d Quelles sont les classes reconnues ?
Dans la doc, nous pouvons voir que tout les modeles sont entrainé sur le dataset `COCO 2017`. Ainsi tout nos modèle reconnaitront les classes de ce dataset. 
- https://cocodataset.org/#detection-2017
- https://cocodataset.org/#keypoints-2017
- https://cocodataset.org/#stuff-2017
On comprend par exemple que ce dataset propose 3 taches, la detection, le keypoint (on comprend qu'il s'agit du squelette des personnes) et le stuff.
https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/  
Il semble possible de récupérr l'ensemble des liste avec un code python.  
Voici la liste des classes du dataset COCO 2017
https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt

## 3.e Y a-t-il des exemples pour gérer une phase d’apprentissage ?

- Collecte de données : La première étape consiste à collecter des données de haute qualité pour votre tâche d'apprentissage. Plus les données sont pertinentes et diversifiées, meilleures seront les performances de votre modèle.

- Prétraitement des données : Avant de passer les données à un modèle d'apprentissage, vous devrez souvent les prétraiter. Cela peut inclure la normalisation, la réduction de dimensionnalité, la gestion des valeurs manquantes et d'autres transformations de données.

- Choix du modèle : Sélectionnez un modèle approprié pour votre tâche. Il peut s'agir de réseaux de neurones convolutifs (CNN) pour la vision par ordinateur, de réseaux de neurones récurrents (RNN) pour le traitement du langage naturel, ou d'autres architectures en fonction de votre problème.

- Définition de l'architecture du modèle : Concevez l'architecture de votre modèle en spécifiant le nombre de couches, les fonctions d'activation, les hyperparamètres, etc.

- Division des données : Séparez vos données en ensembles d'entraînement, de validation et de test. L'ensemble de validation est utilisé pour régler les hyperparamètres, tandis que l'ensemble de test est utilisé pour évaluer les performances du modèle.

- Entraînement du modèle : Utilisez l'ensemble d'entraînement pour ajuster les poids et les biais du modèle en utilisant un algorithme d'optimisation, tel que la descente de gradient. Surveillez les métriques de performance pendant l'entraînement.

- Validation et ajustement des hyperparamètres : Utilisez l'ensemble de validation pour ajuster les hyperparamètres du modèle, tels que le taux d'apprentissage, le nombre de couches, la taille du lot (batch size), etc.

# Sources 
- ChatGPT
- https://intelligence-artificielle.com/classification-d-image-guide-complet/
- https://larevueia.fr/quest-ce-que-la-segmentation-dimages/
- https://www.augmentedstartups.com/blog/mastering-image-classification-techniques-enhancing-accuracy-and-efficiency
- https://viso.ai/deep-learning/object-detection/
- https://www.tensorflow.org/hub/tutorials/tf2_object_detection?hl=fr#more_models
- https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/