# 1
## 1.1 Expliquez les points communs et différences entre Keras, Tensorflow, pytorch et Yolo Darknet
| Outil| Description|Mot clé|Point Commun| Différence|
|------|------------|-------|------------|-----------|
| Keras| Bibliothèque d'apprentissage profond à haut niveau qui permet d'interagir avec les algorithmes de réseaux de neurones profonds et d'apprentissage automatique|Simple - Flexible - Powerful - Interaction - Ergonomie - Modularité| Intégré à TensorFlow| Interface haut niveau|
| TensorFlow| Bibliothèque open source d'apprentissage automatique développée par Google|Open source - Interface - MainStream - Machine learning - Static| Backend intégré pour Keras, agit au même niveau que PyTorch| Utilise un graphique statique, différent de PyTorch|
| PyTorch| Bibliothèque open source python d'apprentissage automatique développée par Meta qui s'appuie sur Torch|Machine learning - Deep learning - Tensor - Numpy - Gradient| Agit au même niveau que TensorFlow | Utilise un graphique dynamique, différent de TensorFlow|
| YOLO Darknet | Framework pour la détection d'objets en temps réel sur des images|Détection object - Traitement image - Marginal - Easy to set up| - | Spécialisé dans la détection d'objets. Un réseau de neuronne pour une image entière|

## 1.2 On explicitera en particulier, ce qui est « statique » vs « dynamique »
- Graphique statique 
- - Structure prédéfinie, présente l'ensemble des opération que le modèle effectuera
- - Le modèle suit strictement la structure prédéfinie lors de son exécution
- - Par exemple TensorFlow, vous déclarez les opérations à l'avance, puis les exécutez dans une session TensorFlow. Ce graphique est fixe et ne change pas pendant l'exécution du modèle.
- Graphique dynamique
- - Flexible dans la construction du modèle. Opérations définies à mesure que le code est exécuté
- - Possibilité d'ajouter, modifier ou supprimer dynamiquement des opérations en fonction des besoin pendant l'exécution du code
- - Par exemple PyTorch, le graphe est construit dynamiquement à mesure que vous effectuez des opérations. Cela facilite le débogage, la recherche et la construction de modèles plus flexibles.

## 1.3 Le langage support utilisé
- Keras
- - API Python
- TensorFlow
- - Python 
- - C++
- - Java
- - ...
- PyTorch
- - Python 
- - API native Python
- YOLO Darknet
- - C 
- - CUDA (pour accéleration GPU)

Globalement Python est principalement utilisé mais beaucoup d'autres possibilité nativement ou en utilisant des franmewors, ou interfaces de binding

## 1.4 Comment est décrit le réseau de neurones  (donner des exemples)
Un réseaux est décrit selon ces spécifications : 
- Manière dont les neurones sont connectés
- Fonctions d'activation utilisées 
- Definition du comprtement du réseau
Exemples de description d'un réseau de neurones entrée de 100 neurones, deux couches cachées 64 neurones avec fonction ReLU, et sortie de 10 neurones avec fonction d'activation softmax.
- Keras (utilisation TensorFlow) en Python
```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=100))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```
- TensorFlow (utilisation Keras) en Python
```
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```
- PyTorch en Python
```
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleMLP()
```
- YOLO Darknet
```
[net]
batch=64
subdivisions=16
height=416
width=416
channels=3

[convolutional]
size=3
stride=1
pad=1
filters=64
activation=leaky
```
# 2
https://www.tensorflow.org/tutorials/keras/classification </br>
Test du code fournit sous google Collab
![alt text](./screen1.png)
![alt text](./screen2.png)
## 2.1 Donner la définition d’un neurone au sens informatique
Un neuronne est une unité dans un réseau de neuronne. Chaque neuronne est conçu pour simuler le comportement des neuronnes biologiques du cerveau humain. Il s'appuie en général sur des données d'entrée, leur appliquent un traitement pour générer une sortie. Chaque neuronne peut etre connecter a une autre neuronne ou une source extérieur pour former un réseau de neuronne.
- Entrée
- - Reception valeur d'entrée venant d'autre neuronne ou de source extérieur
- - Pondérés par des poids définissant leur importance.
- Traitement 
- - Somme des valeurs d'entré en prenant en compte leur poid
- - Soumission des sommes pondérée a une fonction d'activation non linéaire qui permet d'aprendre des relations complexes.
- Sortie
- - Résultat de la fonction d'activation a la somme pondéré. 
- - Transmise a d'autre neuronne ou couche suivante du réseau ou source extérieur.
![alt text](./screen3.png)

## 2.2 En quoi consiste la notion de couche dans les réseaux de neurones 
- Un réseau de neuronne est consititué de plusieur couche.
- Chaque couche possède plusieur neuronne qui sont au même niveau. 
- Chaque neuronne d'une même couche est situé au meme niveau dans le réseau
- - Ils recoivent des données de la couche précédente 
- - Ils envoient leur données traitées a la couche suivante 
- - Les neuronne d'une même couche ne communiquent pas entre eux 

Il existe 3 type de couche 
- Input
- - Une seule couche par réseaux, c'est la première du réseau 
- - Les neuronne recoivent d'une source externe et envoient a la couche caché suivante 
- - Nombre de neuronne depend du cadre d'utilsation du réseau
- Cachées
- - Nombre potentiellement infini par réseaux, apres la couche d'input et avant cette d'output
- - A partir de 2 couches on peut parler de réseau profond => deep learning
- - Chaque couche peut potentiellement contenir une infinité de neuronne.
- - Chaque neuronne recoit de l'input ou d'une couche caché et renvoi a une couche caché ou l'output
- - Applciation d'un traitement dans le transit des données au sein de cette couche
- Output
- - Une seule couche par réseaux, c'est la dernière du réseau 
- - Les neuronne recoivent d'un neuronne chaché et envoient la source externe 
- - Nombre de neuronne depend du cadre d'utilsation du réseau

![alt text](./screen4.png)

## 2.3 En quoi consiste une couche dans un réseau neurone
Réponse au dessus 

## 2.4 Quelles sont les différentes  couches généralement associés dans les réseaux de neurones (faire d’abord une  réponse rapide que vous compléterez jusqu au dernier TP au fur et à mesure ….)
Réponse au dessus

## 2.5 Expliquer en détail le réseau sous-jacent de l’exemple
```
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
```
- Dans l'exemple on construit un réseaux de neuronne séquentiel constitué de 3 couches
- La première est de type `Flatten` 
- - Elle transforme le format des images d'un tableau bidimensionnel (de 28 par 28 pixels) en un tableau unidimensionnel (de 28 * 28 = 784 pixels). 
- - Cette couche permet de désempiler les rangées de pixels de l'image et de les aligner. Cette couche n'a pas de paramètres à apprendre ; elle ne fait que reformater les données.
- Une fois les pixels aplatis, nous avons deux couches de type `Dense` ce type a des paramètre appris pendant l'entrainement, la formation du réseau
- - La première couche Dense comporte 128 nœuds (ou neurones) avec une fonction d'activation `relu`
- - La deuxième (et dernière) couche renvoie un tableau de logits d'une longueur de 10. 
- - - Chaque nœud contient un score qui indique que l'image actuelle appartient à l'une des 10 classes.

## 2.6 Est ce habituel ? Pouvait on faire plus simple ?
Ca semble assez habituel. Je ne sais pas si on pouvait faire plus simple.  
Le format semble assez simple avec seulement une couche d'apprentissage.  
Il semble adaptéau problème en question :  couche d'applatissage des images, et couche de sortie qui renvoie les score par classe  
On a l'air d'enchainer des couches assez simple

## 2.7 Quel type de modifications simplistes peut être faite sur le réseau et en particulier  sur les hyper paramètres. 
Les hyperparamètres ne sont pas appris à partir des données, mais sont définis avant le processus d'entraînement.  
Ils contrôlent divers aspects de la formation et de l'architecture du réseau, influençant ainsi la manière dont le modèle apprend et généralise à partir des données.  
Les hyperparamètres sont essentiels pour ajuster et optimiser le modèle.
Ils sont renseigné au moment de la compilation du modèle 
```
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```
- Optimiseur
- - C'est la façon dont le modèle est mis à jour en fonction des données qu'il voit et de sa fonction de perte.
- Fonction de perte 
- - Elle mesure la précision du modèle pendant la formation. 
- - Vous souhaitez minimiser cette fonction pour "orienter" le modèle dans la bonne direction.
- Métriques
- - Utilisées pour surveiller les étapes de formation et de test. 
- - L'exemple suivant utilise la précision, c'est-à-dire la fraction des images correctement classées.

Dans notre cas, les hyper paramètre semblent complexe a changer, on pourrait deja simplement essayer de changer le modèle du réseau en changeant les couches.  
La couche d'entrée et de sortie etant contraintes par le cadre et le contexte on peut essayer de changer la couche du milieu en changeant la fonction d'activation ou le nombre de noeud, ou bien en rajoutant des couches de ce type.

## 2.8 Relancer l’apprentissage en changeant ses paramètres, mettre en place un tableau montrant l’influence de ses paramètres.
Test avec les params suivants
```
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
```
``
``
```
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])
```
``
``



## 2.9 Quelle est la fonction d’activation associée au modèle, tester avec d’autres fonctions d’activation. Quelles sont les performances obtenues.
C'est la focntion relu qui est définie dans la couche du milieu au moment de la construction des couches du modèle.


# Sources
- ChatGPT  
- https://pjreddie.com/darknet/yolo/  
- https://pytorch.org/ 
- https://www.tensorflow.org/?hl=fr 
- https://keras.io/about/ 
- https://www.geeksforgeeks.org/dynamic-vs-static-computational-graphs-pytorch-and-tensorflow/ 
- https://fr.wikipedia.org/wiki/R%C3%A9seau_de_neurones_artificiels 
- https://fr.blog.businessdecision.com/tutoriel-machine-learning-comprendre-ce-quest-un-reseau-de-neurones-et-en-creer-un/ 
- https://www.tensorflow.org/api_docs/python/tf/keras/optimizers 
- https://www.tensorflow.org/api_docs/python/tf/keras/losses
- https://www.tensorflow.org/api_docs/python/tf/keras/metrics
-  
-  
-  
-  
-  
-  
-  
-  