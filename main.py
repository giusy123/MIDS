from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from sklearn import preprocessing
from keras.utils import np_utils
from keras import callbacks
from DNN_Models import DNN as Models
from keras import optimizers
from keras.models import Model
from Plot import Plot
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from keras.utils.vis_utils import plot_model

def mapLabel(df):
    # creating labelEncoder
    le = preprocessing.LabelEncoder()
    # Converting string labels into numbers
    colClassification = df.columns[-1]
    le.fit(df[colClassification])
    print(list(le.classes_))
    df[colClassification]=le.transform(df[colClassification])
    print("Dataset after mapping categorical classification to numeric")
    print(df.head(5))



def preprocessingDS(df):
    # number of distinct classification target (es normal, attack, dos, probe, U2R, R2L)
    distinctLabelsTrain = df[df.columns[-1]].unique().tolist()
    distinctLabels = sorted(distinctLabelsTrain)
    print("Classification label: ", distinctLabels)
    mapLabel(df)
    #get all columns
    cols=df.columns
    #get numerical columns
    num_cols = df._get_numeric_data().columns
    #get categorical columns
    cat_cols=set(cols) - set(num_cols)
    print("Categorical columns: ", cat_cols)
    df = pd.get_dummies(df, columns=cat_cols)
    return df


def getXY(train, test, target):
    # remove label from dataset to create Y ds
    train_Y=train[target]
    test_Y=test[target]
    print("test y s", test_Y.shape)
    # remove label from dataset
    train_X = train.drop(target, axis=1)
    train_X=train_X.values
    test_X = test.drop(target, axis=1)
    test_X=test_X.values

    return train_X, train_Y, test_X, test_Y





def split_dataset():
    try:
        sizesplit = float(input("Insert split size  (example 0.1):"))
        if  not(0.1 <= sizesplit <= 0.50) :
            raise ValueError()
        return sizesplit

    except ValueError:
        print("Size of split must be a number between 0.1 and 0.5")
    # Prompt user again to input age
    return split_dataset()

def scaler(df, listContent):
    scaler = StandardScaler()
    df[listContent]=scaler.fit_transform(df[listContent].values)
    return df





def main():
 pd.set_option('display.expand_frame_repr', False)
 pathFolder=input("Insert dataset path folder  (tips: dataset):")
 pathDataset=input("Insert dataset path folder  (tips: KDD99.csv):")
 pathPlot=input("Insert plot path folder  (tips: plot):")
 df=pd.read_csv(os.path.join(pathFolder, pathDataset), delimiter=",")
 print("Dataset shape: ", df.shape)
 print("Dataset before preprocessing: ")
 print(df.head(5))

 #Show distinct classification target
 distinctLabels = df[df.columns[-1]].unique().tolist()
 N_CLASSES=len(distinctLabels)

 print("Start preprocessing step")
 numericColumn=df.select_dtypes(include=[np.number]).columns.tolist() #retrieve all numerical columns for standard scaler
 classificationCol=df.columns[-1] #name of target column
 print(classificationCol)

 #preprocessing: map target from categorical to numeric and one-hot encoding at categorical columns
 df = preprocessingDS(df)
 print("Dataset after one-hot encoding:")
 print(df.head(5))

 #preprocessing: standar scaler
 df=scaler(df,numericColumn)

 #Split function on train and testing set
 sizesplit=split_dataset()
 pl=Plot(pathPlot)




 train, test = train_test_split(df, test_size=sizesplit)

 print("Train shape after split: ", train.shape)
 print("Test shape after split: ", test.shape)
 pl.plotStatistics(train,test,classificationCol)

 train_X, train_Y, test_X, test_Y=getXY(train,test,classificationCol)
 # convert class vectors to binary class matrices
 train_Y2 = np_utils.to_categorical(train_Y, N_CLASSES)


 callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)]

 m = Models(N_CLASSES)

 VALIDATION_SPLIT=0.1
 print('Model with autoencoder+softmax with fixed encoder weights')
 # parametri per autoencoder
 p1 = {
     'first_layer': 60,
     'second_layer': 30,
     'third_layer': 10,
     'batch_size': 64,
     'epochs': 150,
     'optimizer': optimizers.Adam,
     'kernel_initializer': 'glorot_uniform',
     'losses': 'mse',
     'first_activation': 'tanh',
     'second_activation': 'tanh',
     'third_activation': 'tanh'}

 autoencoder = m.deepAutoEncoder(train_X, p1)
 autoencoder.summary()

 #get encoder for feature extraction
 encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder3').output)
 encoder.summary()

 history2 = autoencoder.fit(train_X, train_X,
                            validation_split=VALIDATION_SPLIT,
                            batch_size=p1['batch_size'],
                            epochs=p1['epochs'], shuffle=False,
                            callbacks=callbacks_list,
                            verbose=1)


 pl.printPlotLoss(history2, 'autoencoder')
 plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True, show_layer_names=True)


 '''
 Save weigths from autoencoder model
 Weights are fixed in the classifier model
 '''
 weights = []
 i = 0
 for layer in autoencoder.layers:
     weights.append(layer.get_weights())

 # parameters for final model
 p2 = {
     'batch_size': 256,
     'epochs': 100,
     'optimizer': optimizers.Adam,
     'kernel_initializer': 'glorot_uniform',
     'losses': 'binary_crossentropy',
     'first_activation': 'tanh',
     'second_activation': 'tanh',
     'third_activation': 'relu'}



 model = m.MLP_WeightFixed(encoder, train_X, p2)

 history3 = model.fit(train_X, train_Y2,
                      validation_split=VALIDATION_SPLIT,
                      batch_size=p1['batch_size'],
                      epochs=p1['epochs'], shuffle=False,
                      callbacks=callbacks_list,
                      verbose=1)


 pl.printPlotAccuracy(history3, 'finalModel1')
 pl.printPlotLoss(history2, 'finalModel1')
 model.save('modelfixedW.h5')
 plot_model(model, to_file='classifier.png', show_shapes=True, show_layer_names=True)

 predictions = model.predict(test_X)

 # Predicting the Test set results
 y_pred = np.argmax(predictions, axis=1)
 cm = confusion_matrix(test_Y, y_pred)
 acc = accuracy_score(test_Y, y_pred, normalize=True)
 LABELS=["Attacks","Normal"]
 print("Confusion matrix on test set")
 print(cm)
 print("Accuracy model on test set: "+ str(acc))
 plt.figure(figsize=(12, 12))
 sns.heatmap(cm, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
 plt.title("Confusion matrix on test set")
 plt.ylabel('True class')
 plt.xlabel('Predicted class')
 plt.savefig(os.path.join(pathPlot,"confusion matrix"))
 plt.show()
 plt.close()


if __name__ == "__main__":
 main()