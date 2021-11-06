# Fabric pattern classification using Transfer Learning
Keras library which is built on TensorFlow is used to design and modify the networks along with other python libraries like sklearn, Pillow, NumPy and matplotlib for data preprocessing, image manipulation, array processing and visualization respectively. 
## Dataset 
Dataset for this research work is scrapped from internet using python beautiful soup library. It has a total of more than 5000 images, covering seven categories of fabric designs. It is further divided into test and train using sklearn preprocessing library. For each image, ten more samples were generated using data augmentation technique by using Keras data generator. 
You can download the dataset from Dataset link: https://drive.google.com/drive/folders/1p-ZeZaP3C8t2NIfdk7gpJYrd9czULufZ?usp=sharing.

<img src='https://i.pinimg.com/564x/6a/9c/db/6a9cdb44ca1a102d7dcf6d5f21d243f0.jpg'>

## Data preprocessing

The major preprocessing steps consist of image resizing and standardization. With the constrain of input image size, it is mandatory to adjust the input image size to uniform size. Pillow and OpenCV libraries are used to load and preprocess the images. For making the convergence faster while training the network, images were standardized by dividing each pixel values by 255.


## Results Obtained
<img src='https://64.media.tumblr.com/d1421b7d1f051dae2063b1de7f720b84/ce9d2e6270bf5b7f-5b/s640x960/b060aac3667980ad70ea530f8453dcde4823007c.png'>
