# Mask-Detection

> CNN using Tensorflow and OpenCV to detect whether a mask is being worn




Who Is This Project For?
-----------------------
* You have an intermediate-advanced understanding of Python
* You have an interest in computer vision and have completed "Hello World" projects like [mnist](https://www.tensorflow.org/datasets/catalog/mnist) 
* You have neural network experience and want to dabble in OpenCV development
* You are interested in contributing to relevant, data science impacting our current world



Why This Project?
-----------------------
* The goals of your project
* Your though process and methods in creating the project
[Prajna Bhandary](https://github.com/prajnasb/observations)



Usage
-----------------------
* Download the data source from [here](https://github.com/prajnasb/observations/tree/master/experiements/data)
* Prajna's project comes in two sub-folders with and without masks, so merge the two into a singular "Mask Images" folder 
* Follow the steps listed in mask_algorithm.py (resize to 100x100, divide by 225, convert to greyscale, split into 80-20 train test split using sklearn.model_selection.train_test_split, save each data source as data.npy and target.npy respectively)
* Follow the steps listed in mask_algorithm.py
* To save time, save the Tensorflow weights so the model will not need to be trained each time the script is run (~10 minutes)
* The webcam output gets saved as alex_mask.api. You can change the name if your name is not also Alex :)



Extending this
-------------------------
* Larger training set 
* More types of masks included 
