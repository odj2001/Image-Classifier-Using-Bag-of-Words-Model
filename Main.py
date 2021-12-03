import cv2 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

from sklearn.cluster import SpectralClustering
from sklearn.cluster import SpectralBiclustering

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn


import os.path

from sklearn import metrics


#%matplotlib inline

# read images
img1 = cv2.imread('Data/Training/cars/0006.jpg')
img2 = cv2.imread('Data/Training/cars/0007.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#sift
sift = cv2.SIFT_create()




combinedDescriptors = []

#appending each descriptor onto a single list so it can be fit to the spectral function

for i in range(80):
    
    
    if os.path.isfile(f'Data/Training/airplanes/00{i+1:02d}.jpg') == True:

        img = cv2.imread(f'Data/Training/airplanes/00{i+1:02d}.jpg') 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = sift.detectAndCompute(img,None)  
        #print(descriptors.shape[0])
    
        for x in range(descriptors.shape[0]):
            combinedDescriptors.append(descriptors[x])

    if os.path.isfile(f'Data/Training/cars/00{i+1:02d}.jpg') == True:
    
        img = cv2.imread(f'Data/Training/cars/00{i+1:02d}.jpg') 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = sift.detectAndCompute(img,None)  
        for x in range(descriptors.shape[0]):
            combinedDescriptors.append(descriptors[x])
            
             
    if os.path.isfile(f'Data/Training/dog/00{i+1:02d}.jpg') == True: 
        
        img = cv2.imread(f'Data/Training/dog/00{i+1:02d}.jpg') 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = sift.detectAndCompute(img,None)  
        for x in range(descriptors.shape[0]):
            combinedDescriptors.append(descriptors[x])
            
    if os.path.isfile(f'Data/Training/faces/00{i+1:02d}.jpg') == True:
    
        img = cv2.imread(f'Data/Training/faces/00{i+1:02d}.jpg') 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img,None)  
        for x in range(descriptors.shape[0]):
            combinedDescriptors.append(descriptors[x])
            
    if os.path.isfile(f'Data/Training/keyboard/00{i+1:02d}.jpg') == True:
    
        img = cv2.imread(f'Data/Training/keyboard/00{i+1:02d}.jpg') 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img,None)  
        for x in range(descriptors.shape[0]):
            combinedDescriptors.append(descriptors[x])


spectral = MiniBatchKMeans(init="k-means++",n_clusters=500,batch_size=5000, random_state=0).fit(combinedDescriptors)

#spectral = SpectralBiclustering(n_clusters=50,random_state=0).fit(combinedDescriptors)




#Outputs the histogram




bar_width = 1. # set this to whatever you want
positions = np.arange(spectral.cluster_centers_.shape[0])

trainFeature = []
trainLabel = []

testFeature = []
testLabel = []



for i in range(80):
    
    
    airplaneHistData = [0]*spectral.cluster_centers_.shape[0]
    carHistData = [0]*spectral.cluster_centers_.shape[0]
    dogHistData = [0]*spectral.cluster_centers_.shape[0]
    facesHistData = [0]*spectral.cluster_centers_.shape[0]
    keyboardHistData = [0]*spectral.cluster_centers_.shape[0]
    
    
    
    testAirplaneHistData = [0]*spectral.cluster_centers_.shape[0]
    testCarHistData = [0]*spectral.cluster_centers_.shape[0]
    testDogHistData = [0]*spectral.cluster_centers_.shape[0]
    testFacesHistData = [0]*spectral.cluster_centers_.shape[0]
    testKeyboardHistData = [0]*spectral.cluster_centers_.shape[0]
    
    #creating the training feature and training labels

    if os.path.isfile(f'Data/Training/airplanes/00{i+1:02d}.jpg') == True:

        img = cv2.imread(f'Data/Training/airplanes/00{i+1:02d}.jpg') 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img,None)
        
        for x in range(descriptors.shape[0]):
            minDistance=100000
            currentWord=0
            for y in range (spectral.cluster_centers_.shape[0]):
                currentDistance = np.linalg.norm(descriptors[x]-spectral.cluster_centers_[y])
        

                if minDistance > currentDistance:
                    minDistance = currentDistance
                    currentWord = y
                    
            airplaneHistData[currentWord]+=1
        
        norm = np.linalg.norm(airplaneHistData)
        
        airplaneHistData = airplaneHistData/norm
        
        trainFeature.append(airplaneHistData)
        trainLabel.append("Airplanes")


        plt.bar(positions, airplaneHistData, bar_width)
        plt.savefig(f'Output/airplanes/Histogram_00{i+1:02d}.jpg',bbox_inches='tight')
        plt.clf()


    if os.path.isfile(f'Data/Training/cars/00{i+1:02d}.jpg') == True:
    
        img = cv2.imread(f'Data/Training/cars/00{i+1:02d}.jpg') 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img,None)
        
        
        for x in range(descriptors.shape[0]):
            minDistance=100000
            currentWord=0
            for y in range (spectral.cluster_centers_.shape[0]):
                currentDistance = np.linalg.norm(descriptors[x]-spectral.cluster_centers_[y])
                
                
                

                if minDistance > currentDistance:
                    minDistance = currentDistance
                    currentWord = y
                    
            carHistData[currentWord]+=1
            
            
        norm = np.linalg.norm(carHistData)
        
        carHistData = carHistData/norm
        
        trainFeature.append(carHistData)
        trainLabel.append("Car")
        

        plt.bar(positions, carHistData, bar_width)
        plt.savefig(f'Output/cars/Histogram_00{i+1:02d}.jpg',bbox_inches='tight')
        plt.clf()


             
    if os.path.isfile(f'Data/Training/dog/00{i+1:02d}.jpg') == True: 
        
        img = cv2.imread(f'Data/Training/dog/00{i+1:02d}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img,None)
        

        for x in range(descriptors.shape[0]):
            minDistance=100000
            currentWord=0
            for y in range (spectral.cluster_centers_.shape[0]):
                currentDistance = np.linalg.norm(descriptors[x]-spectral.cluster_centers_[y])
        
                if minDistance > currentDistance:
                    minDistance = currentDistance
                    currentWord = y
                    
            dogHistData[currentWord]+=1
            
            
        norm = np.linalg.norm(dogHistData)
        
        dogHistData = dogHistData/norm
        trainFeature.append(dogHistData)
        trainLabel.append("Dog")
        
        
        plt.bar(positions, dogHistData, bar_width)
        plt.savefig(f'Output/dog/Histogram_00{i+1:02d}.jpg',bbox_inches='tight')
        plt.clf()

            
    if os.path.isfile(f'Data/Training/faces/00{i+1:02d}.jpg') == True:
    
        img = cv2.imread(f'Data/Training/faces/00{i+1:02d}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img,None)
        

        for x in range(descriptors.shape[0]):
            minDistance=100000
            currentWord=0
            for y in range (spectral.cluster_centers_.shape[0]):
                currentDistance = np.linalg.norm(descriptors[x]-spectral.cluster_centers_[y])
        

                if minDistance > currentDistance:
                    minDistance = currentDistance
                    currentWord = y
                    
            facesHistData[currentWord]+=1
            
            
        norm = np.linalg.norm(facesHistData)
        
        facesHistData = facesHistData/norm
        trainFeature.append(facesHistData)
        trainLabel.append("Faces")
        
        plt.bar(positions, facesHistData, bar_width)
        plt.savefig(f'Output/faces/Histogram_00{i+1:02d}.jpg',bbox_inches='tight')
        plt.clf()
            
    if os.path.isfile(f'Data/Training/keyboard/00{i+1:02d}.jpg') == True:
    
        img = cv2.imread(f'Data/Training/keyboard/00{i+1:02d}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img,None)
        

        for x in range(descriptors.shape[0]):
            minDistance=100000
            currentWord=0
            for y in range (spectral.cluster_centers_.shape[0]):
                currentDistance = np.linalg.norm(descriptors[x]-spectral.cluster_centers_[y])
        

                if minDistance > currentDistance:
                    minDistance = currentDistance
                    currentWord = y
                    
            keyboardHistData[currentWord]+=1
            
            
            
        norm = np.linalg.norm(keyboardHistData)
        
        keyboardHistData = keyboardHistData/norm
        
        trainFeature.append(keyboardHistData)
        trainLabel.append("Keyboard")
        
        plt.bar(positions, keyboardHistData, bar_width)
        plt.savefig(f'Output/keyboard/Histogram_00{i+1:02d}.jpg',bbox_inches='tight')
        plt.clf()
            
        
        
    #creating the testing feature and testing labels
     
    if os.path.isfile(f'Data/Test/airplanes/00{i+1:02d}.jpg') == True:

        img = cv2.imread(f'Data/Test/airplanes/00{i+1:02d}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img,None)
        
        for x in range(descriptors.shape[0]):
            minDistance=100000
            currentWord=0
            for y in range (spectral.cluster_centers_.shape[0]):
                currentDistance = np.linalg.norm(descriptors[x]-spectral.cluster_centers_[y])


                if minDistance > currentDistance:
                    minDistance = currentDistance
                    currentWord = y
                    
            testAirplaneHistData[currentWord]+=1
        
        norm = np.linalg.norm(testAirplaneHistData)
        
        testAirplaneHistData = testAirplaneHistData/norm
        
        testFeature.append(testAirplaneHistData)
        testLabel.append("Airplanes")



    if os.path.isfile(f'Data/Test/cars/00{i+1:02d}.jpg') == True:
    
        img = cv2.imread(f'Data/Test/cars/00{i+1:02d}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img,None)
        
        
        for x in range(descriptors.shape[0]):
            minDistance=100000
            currentWord=0
            for y in range (spectral.cluster_centers_.shape[0]):
                currentDistance = np.linalg.norm(descriptors[x]-spectral.cluster_centers_[y])

                if minDistance > currentDistance:
                    minDistance = currentDistance
                    currentWord = y
                    
            testCarHistData[currentWord]+=1
            
            
        norm = np.linalg.norm(testCarHistData)
        
        testCarHistData = testCarHistData/norm
        
        testFeature.append(testCarHistData)
        testLabel.append("Car")
             
    if os.path.isfile(f'Data/Test/dog/00{i+1:02d}.jpg') == True: 
        
        img = cv2.imread(f'Data/Test/dog/00{i+1:02d}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img,None)
        

        for x in range(descriptors.shape[0]):
            minDistance=100000
            currentWord=0
            for y in range (spectral.cluster_centers_.shape[0]):
                currentDistance = np.linalg.norm(descriptors[x]-spectral.cluster_centers_[y])
        
                if minDistance > currentDistance:
                    minDistance = currentDistance
                    currentWord = y
                    
            testDogHistData[currentWord]+=1
            
            
        norm = np.linalg.norm(testDogHistData)
        
        testDogHistData = testDogHistData/norm
        testFeature.append(testDogHistData)
        testLabel.append("Dog")
            
    if os.path.isfile(f'Data/Test/faces/00{i+1:02d}.jpg') == True:
    
        img = cv2.imread(f'Data/Test/faces/00{i+1:02d}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img,None)
        

        for x in range(descriptors.shape[0]):
            minDistance=100000
            currentWord=0
            for y in range (spectral.cluster_centers_.shape[0]):
                currentDistance = np.linalg.norm(descriptors[x]-spectral.cluster_centers_[y])
        

                if minDistance > currentDistance:
                    minDistance = currentDistance
                    currentWord = y
                    
            testFacesHistData[currentWord]+=1
            
            
        norm = np.linalg.norm(testFacesHistData)
        
        testFacesHistData = testFacesHistData/norm
        testFeature.append(testFacesHistData)
        testLabel.append("Faces")
            
    if os.path.isfile(f'Data/Test/keyboard/00{i+1:02d}.jpg') == True:
    
        img = cv2.imread(f'Data/Test/keyboard/00{i+1:02d}.jpg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(img,None)
        

        for x in range(descriptors.shape[0]):
            minDistance=100000
            currentWord=0
            for y in range (spectral.cluster_centers_.shape[0]):
                currentDistance = np.linalg.norm(descriptors[x]-spectral.cluster_centers_[y])
        

                if minDistance > currentDistance:
                    minDistance = currentDistance
                    currentWord = y
                    
            testKeyboardHistData[currentWord]+=1
            
            
            
        norm = np.linalg.norm(testKeyboardHistData)
        
        testKeyboardHistData = testKeyboardHistData/norm
        
        testFeature.append(testKeyboardHistData)
        testLabel.append("Keyboard")
     
     

        
model = KNeighborsClassifier(n_neighbors=20)
# Train the model using the training sets
model.fit(trainFeature,trainLabel)





predicted= model.predict(testFeature) 

target_names = ["Airplanes", "Car", "Dog", "Faces", "Keyboard"]


confusionMatrix = confusion_matrix(testLabel, predicted)

accuracy = confusionMatrix.diagonal()/confusionMatrix.sum(axis=1)


print("Airplanes Accuracy: ", accuracy[0])
print("Car Accuracy: ", accuracy[1])
print("Dog Accuracy: ", accuracy[2])
print("Faces Accuracy: ", accuracy[3])
print("Keyboard Accuracy: ", accuracy[4])

df_cm = pd.DataFrame(confusionMatrix, target_names, target_names)
# plt.figure(figsize=(10,7))
sn.set(font_scale=1.4) # for label size
sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
plt.xlabel('Prediction', fontsize=16)
plt.ylabel('Actual', fontsize=16)
plt.savefig('Output/Confusion_Matrix.png',bbox_inches='tight')












