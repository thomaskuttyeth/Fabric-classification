from augmentation import ImageAugmentation 
from dataloader import Dataloader 

# calling the loader 
main_loader = Dataloader() 


# if there is any test folder inside the Images directory - skip 
main_loader.make_test(0.15) 

# getting counts of images in both test and train directories 
counts = main_loader.get_counts() 

# setting up the train_data, train_labels and test_data, test_labels 
main_loader.load_('Images/train',224)
main_loader.load_('Images/test',224) 
main_loader.load_('augmented_images',224) 

train_data,train_labels = main_loader.train_data, main_loader.train_labels 
test_data, test_labels = main_loader.test_data, main_loader.test_labels 
aug_data, aug_labels = main_loader.augmented_data, main_loader.augmented_labels 

# merging train and augmented data together to form final train and test 
final_train_data = train_data+aug_data 
final_train_labels = train_labels+aug_labels 

# model calling 