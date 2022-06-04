# feature-extraction-with-dimensionality-reduction
# Import Package
import common packages
- import **numpy as np**
- import **pandas as pd**
- import **matplotlin.pyplot as plt**
- from **sklearn.model_selection** import **train_test_split**
- from **sklearn.decomposition** import **PCA, NMF**

# Import Dataset
The data used is life faces in the wild data, namely the faces of people who are known to many people. its contents are world figures, or movie actors, etc. 

Data in the form of images that have been flattened. The data consists of 3023 faces, and the features are 5656 which is the pixel size of the image. The image is 87 x 65 pixels.

# Dataset Splitting
Split data into X, and y.

X = all columns except the target column. Which is converted into numpy

y = 'name' column as target. Which is converted into numpy

test_size = 0.2 (which means 80% for train, and 20% for test). And stratified so that the test data is representative.

# Visualize Data
Sample data.
![image](https://user-images.githubusercontent.com/86812576/171996592-23892dbc-36ee-490e-92f7-e0974c7154a9.png)

# Decide n_components using Cumulative Explained Variance
I will try feature extraction from image. the principle is the same after the image is flattened, it will be decomposed, which was originally 3023 x 5656 into just a few features and got its principle component (PCA).

# Feature Extraction with Dimensionality Reduction
After doing cumulative explained variance, I chose to use 250 components. It can be seen in the previous plot, that 250 components can still store around 95% of information, and add whiten (scaling).

At first there is x_train data, the size is 2418 faces with 5655 pixels, after reducing x_train_pca the number of faces remains the same 2418 but now the feature has been compressed to a size of 250.

# Visualize PCA components
What are the features of the PCA components? I will try to visualized the PCA component. PCA components have such a hierarchy, where the first component is the most important component, and so on up to 250 components.

Plot of the first ten components. 

![image](https://user-images.githubusercontent.com/86812576/171997276-be1c16c4-acd5-4d95-a546-ce9e41c6642f.png)
