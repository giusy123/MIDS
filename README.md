# Machine learning for Intrusion Detection System (MIDS)

Il repository contiene l'implementazione di una Deep Neural Networks (DNN) per classificare esempi di attacchi da comportamenti normali

## Requisiti del codice

Il codice realizzato è stato testato sulla libreria **python3.6+** libs.

Packages necessari sono:
* [Tensorflow 1.13](https://www.tensorflow.org/) 
* [Keras 2.2.4](https://github.com/keras-team/keras) 
* [Matplotlib 3.0.3](https://matplotlib.org/)
* [Numpy 1.15.4](https://www.numpy.org/)
* [Pandas 0.24.2](https://pandas.pydata.org/)
* [Scikit-learn](https://scikit-learn.org/stable/)
* [Seaborn 0.9.0](https://seaborn.pydata.org/)

## Data
Il dataset usato per gli esperimenti è accessibile da [__NSL-KDD__](https://www.unb.ca/cic/datasets/nsl.html). 
Il dataset originale (con classificazione a 5 classi) è stato trasformato in un dataset binario con due classificazioni: "_attack_, _normal_" (_oneCls files) and then the  feature  selection  stage  is  performed  by  retain  the  10top-ranked  features  according  to  __Information  Gain(IG)__ .
Inoltre, delle 41 features originali il dataset contenuto nella cartella dataset del progetto sono state selezionate 10 features, che dventano 89 dopo le fasi di preprocessing

## Descrizione del codice
Lo script contiene il codice utile per:
1. Fase di preprocessing: 
  * Trasforma da categoriche a numeriche le categorie target del dataset: attacco=0; normale=1
  * One-hot encoding per trasforma le features categoriche 
  * Uso dela libreria Standard Scale
2. Crea un autoencoder con 60-30-10 neuroni rispettivamente per la parte di encoder e 30-60 per la parte di decoder e lo addestra sul dataset di training

![Layers autoencoder model](https://github.com/giusy123/MIDS/blob/master/autoencoder.png)

3. Salva i pesi dell'autoencoder precedentemente appreso. I primi tre livelli della parte relativa all'encoder con i pesi fissati diventano i  primi due livelli, a cui è aggiunto un ultimo livello con fuzione __softmax__ di un modello che classifica attacchi da non attacchi.

![Layers classification model](https://github.com/giusy123/MIDS/blob/master/classifier.png)

4. Il modello è poi usato con funzione di predizione sul testing set per valutarne l'accuratezza del modello

## Come usare lo script
Lo script richiede in input:

* La cartella in cui si trova il dataset, usare dataset (la cartella dataset è già fornita nel repository)
* Il nome del dataset (sugg. KDDTrain.csv è il dataset fornito con il codice)
* Il path della cartella dove salvare i plot (la cartella plot è già fornita con il repository)
* La percentuale in cui splittare il dataset tra training set e testing set (il codice accetta in input un valore tra 0.1 e 0.5)


