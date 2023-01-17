import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

k = 4


def dist(p, c):
    k,_ = c.shape
    #print(k)
    dists = []
    for i in range(k):
        #print (c[i])
        dists.append(np.linalg.norm(p - c[i], axis = 1))
    return np.array(dists).T




img = cv.imread('Q4image.png')
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
pixels = rgb_img.reshape((-1, 3))
pixels = np.float32(pixels)
w, h = pixels.shape[0], pixels.shape[1]
no_of_iterations = 10


    
idx = np.random.choice(len(pixels), k, replace = False)
centroids = pixels[idx, :] #Step 1
     
#finding the distance between centroids and all the data points
distances = dist(pixels, centroids) #Step 2
     
#Centroid with the minimum Distance
points = np.array([np.argmin(i) for i in distances]) #Step 3
     
    #Repeating the above steps for a defined number of iterations
    #Step 4
for _ in range(no_of_iterations): 
    centroids = []
    for idx in range(k):
        #Updating Centroids by taking mean of Cluster it belongs to
        temp_cent = pixels[points==idx].mean(axis=0) 
        centroids.append(temp_cent)
 
    centroids = np.vstack(centroids) #Updated Centroids 
        
    distances = dist(pixels, centroids)
    points = np.array([np.argmin(i) for i in distances])
         

centroids = np.uint8(np.round(centroids))
segmented_image = centroids[points]
segmented_image = segmented_image.reshape(img.shape)

plt.imshow(segmented_image)
plt.show()
