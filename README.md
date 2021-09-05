# Transfer-Learning-for-fabric-pattern-classification
Keras library which is built on TensorFlow is used to design and modify the networks along with other python libraries like sklearn, Pillow, NumPy and matplotlib are used for data preprocessing, image manipulation, array processing and visualization respectively. 
## Dataset 
Dataset for this research work is scrapped from internet using python beautiful soup library. It has a total of more than 5000 images, covering seven categories of fabric designs. It is further divided into test and train using sklearn preprocessing library. For each image, ten more samples were generated using data augmentation technique by using Keras data generator. 

<img src='https://i.pinimg.com/564x/6a/9c/db/6a9cdb44ca1a102d7dcf6d5f21d243f0.jpg'>

## Data preprocessing

The major preprocessing steps consist of image resizing and standardization. With the constrain of input image size, it is mandatory to adjust the input image size to uniform size. Pillow and OpenCV libraries are used to load and preprocess the images. For making the convergence faster while training the network, images were standardized by dividing each pixel values by 255.
