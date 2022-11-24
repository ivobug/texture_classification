# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 01:57:06 2022

@author: Ivan
"""

import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from sklearn.svm import LinearSVC
import mahotas
import pandas as pd
import random
from sklearn import metrics

path = "brodatz"

readImages=os.listdir(path)
#we will use dataframe to store our data, dataframe will have 3 columns
#(id, stored image, actual label, and calculated haralick label)
df=pd.DataFrame()
#We need iterator to set id in dataframe
ite=0
for img in readImages:
    #load image to cv2
    image= cv2.imread(os.path.join(path,img),0)
    height, width= image.shape
    # set image name as a label name
    label=img.split('.')[0]
    for i in range(4):
        heightDiv4= int(height/4)
        widthDiv4= int(width/4)
        #next for loop we use to divide each image on 16 (4x4) smaller images to expand the dataset
        for j in range(4):
            #croped image
            crop_img=image[heightDiv4*i: heightDiv4*i+heightDiv4, widthDiv4*j: widthDiv4*j+widthDiv4]
            #calculate haralick features
            features = mahotas.features.haralick(crop_img).mean(axis=0)
            #define new roe for dataframe
            new_row = {'id':ite,'image':crop_img, 'label':label, 'features':features}
            df = df.append(new_row, ignore_index=True)
            ite+=1

#split data on 75% for train and 25% for test
train, test = train_test_split(df)

#define Linear Support Vector Classification model
model = LinearSVC(C=2)
# fit model with real labels 
model.fit(list( train['features']), list(train['label']))

#implementing visualize one random sample function
def visualizeRandom():
    #put all 'id'-s from test dataframe to list
    ids=list(test['id'])
    #get one random data from test dataframe
    random_data=random.choice(ids)
    #get features from dataframe by provided random id
    features = test.set_index('id').loc[random_data, 'features']
    #predict our random sample
    pred = model.predict(features.reshape(1, -1))[0]
    print(f'Real: {test["label"][random_data]}, predicted: {pred}')
    #plot
    plt.imshow(test['image'][random_data], cmap='gray')

#!!!!!! uncomment RUN this to visualize random data
#visualizeRandom()

#define actual and predicted arrays where we gonna append real and predicted labels
actual=[]
predicted=[]

for data in test['id']:
    #features are alredy stored to dataframe so we are gonna stoore them to varibale to predict them
    features= test.set_index('id').loc[data, 'features']
    pred = model.predict(features.reshape(1, -1))[0]
    predicted.append(pred)
    #store actual label to actual array
    actual_label=test.set_index('id').loc[data, 'label']
    actual.append(actual_label)

#implement and visualize confusion matrix
confusion_matrix = metrics.confusion_matrix(actual, predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()

score= model.score(list( train['features']), list(train['label'])) 
print(f'Mean accuracy of model is {score}')
