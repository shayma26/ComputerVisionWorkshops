#mount google drive to get access to train data and test data
from google.colab import drive
drive.mount('/content/drive')

cd /content/drive/MyDrive/caltech_dataset

from google.colab.patches import cv2_imshow
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC            
from skimage import exposure
from skimage import feature
from imutils import paths
import imutils
from sklearn.decomposition import PCA


# Function to compute histogram from LBP features
def compute_lbp(image, eps=1e-7,numPoints=24,radius=8):
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, numPoints,
			radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, numPoints + 3),
			range=(0, numPoints + 2))
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		# return the histogram of Local Binary Patterns
		return hist



# Feature extraction from the training data with the corresponding labels

print("[INFO] Extracting different features:  Raw Pixel values from image/Color Histogram/HOG/LBP")
train_data_hog=[]
#TODO : Define  a vector to stock labels
train_labels = []

train_data = []
train_data_lbp = []
train_data_hsv = []

for imagePath in paths.list_images('train_set'):
  #TODO: Retrieve the name of the class from each image path 
  label = imagePath.split('/')[1]
  print(label)
  image = cv2.imread(imagePath)
  #TODO: Display the image
  image = cv2.resize(image,(200,100))
  cv2_imshow(image)
  v=image.flatten()

  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
  hist = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256]).flatten()

  (H, hogImage) = feature.hog(gray_image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2),transform_sqrt=True, block_norm="L1",visualize=True)
  hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
  hogImage = hogImage.astype("uint8")
  #TODO: Display HOG image
  cv2_imshow(hogImage)
  
  lbpImage = compute_lbp(gray_image)

  train_data_hog.append(H)
  train_data_lbp.append(lbpImage)
  train_data_hsv.append(hist)
  train_data.append(v)

  #TODO: update labels vector
  train_labels.append(label)


#TODO: Print the size of each feature vector 
print(len(train_data) == len(train_labels))

# Feature extraction from the testing data with the corresponding labels

print("[INFO] Extracting different features:  Raw Pixel values from image/Color Histogram/HOG/LBP")
test_data_hog=[]
#TODO : Define  a vector to stock labels
test_labels = []

test_data = []
test_data_lbp = []
test_data_hsv = []

for imagePath in paths.list_images('test_set'):
  #TODO: Retrieve the name of the class from each image path 
  label = imagePath.split('/')[1]
  print(label)
  image = cv2.imread(imagePath)
  #TODO: Display the image
  image = cv2.resize(image,(200,100))
  cv2_imshow(image)
  v=image.flatten()

  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
  hist = cv2.calcHist([hsv_image], [0, 1], None, [180, 256], [0, 180, 0, 256]).flatten()

  (H, hogImage) = feature.hog(gray_image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2),transform_sqrt=True, block_norm="L1",visualize=True)
  hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
  hogImage = hogImage.astype("uint8")
  #TODO: Display HOG image
  cv2_imshow(hogImage)
  
  lbpImage = compute_lbp(gray_image)

  test_data_hog.append(H)
  test_data_lbp.append(lbpImage)
  test_data_hsv.append(hist)
  test_data.append(v)

  #TODO: update labels vector
  test_labels.append(label)


# SVM and KNN models

model1 = KNeighborsClassifier(n_neighbors=1)
model2 = SVC(kernel='linear')

# Train SVM and KNN models and predict the results, evaluate them as well!

# Train and predict based on model 1, evalute your results as well!
model1.fit(train_data, train_labels)
predicted_classes1 = model1.predict(test_data)
#TODO: evaluate the obtained results by comparing the predicted classes to the ground truth
accurcy = sum(test_labels==predicted_classes1)/len(test_labels)
print(accurcy)

# Train and predict based on model 2, evalute your results as well!
model2.fit(train_data, train_labels)
predicted_classes2 = model2.predict(test_data)
#TODO: evaluate the obtained results by comparing the predicted classes to the ground truth
accurcy = sum(test_labels==predicted_classes2)/len(test_labels)
print(accurcy)

# Train and predict based on model 1, evalute your results as well!
model1.fit(train_data_hog, train_labels)
predicted_classes1 = model1.predict(test_data_hog)
#TODO: evaluate the obtained results by comparing the predicted classes to the ground truth
accurcy = sum(test_labels==predicted_classes1)/len(test_labels)
print(accurcy)

# Train and predict based on model 2, evalute your results as well!
model2.fit(train_data_hog, train_labels)
predicted_classes2 = model2.predict(test_data_hog)
#TODO: evaluate the obtained results by comparing the predicted classes to the ground truth
accurcy = sum(test_labels==predicted_classes2)/len(test_labels)
print(accurcy)

# Train and predict based on model 1, evalute your results as well!
model1.fit(train_data_lbp, train_labels)
predicted_classes1 = model1.predict(test_data_lbp)
#TODO: evaluate the obtained results by comparing the predicted classes to the ground truth
accurcy = sum(test_labels==predicted_classes1)/len(test_labels)
print(accurcy)

# Train and predict based on model 2, evalute your results as well!
model2.fit(train_data_lbp, train_labels)
predicted_classes2 = model2.predict(test_data_lbp)
#TODO: evaluate the obtained results by comparing the predicted classes to the ground truth
accurcy = sum(test_labels==predicted_classes2)/len(test_labels)
print(accurcy)

# Train and predict based on model 1, evalute your results as well!
model1.fit(train_data_hsv, train_labels)
predicted_classes1 = model1.predict(test_data_hsv)
#TODO: evaluate the obtained results by comparing the predicted classes to the ground truth
accurcy = sum(test_labels==predicted_classes1)/len(test_labels)
print(accurcy)

# Train and predict based on model 2, evalute your results as well!
model2.fit(train_data_hsv, train_labels)
predicted_classes2 = model2.predict(test_data_hsv)
#TODO: evaluate the obtained results by comparing the predicted classes to the ground truth
accurcy = sum(test_labels==predicted_classes2)/len(test_labels)
print(accurcy)
