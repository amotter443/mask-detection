# Mask-Detection
> CNN using Tensorflow and OpenCV to detect whether a mask is being worn

This project was came from a recent focus on developing skills in computer vision/OpenCV development. Having seen a version of this created by [Prajna Bhandary](https://github.com/prajnasb/observations), I knew other data scientists were working in the space and had similar successes in implementing solutions. I included several unique variations, including tuning the model using images of reusable masks, strengthening the CNN architecture, and streamlining the code for more efficient runtime. The final result, with an over 99.5% accuracy, was able to identify not only whether individuals were wearing masks but also if those masks were being worn correctly. The final demo of my code can be viewed [here.](https://www.linkedin.com/feed/update/urn:li:activity:6709477923985907712/)


Who Is This Project For?
-----------------------
* You have an intermediate to advanced understanding of Python
* You have an interest in computer vision and have completed "Hello World" projects like [mnist](https://www.tensorflow.org/datasets/catalog/mnist) 
* You have neural network experience and want to dabble in OpenCV development
* You are interested in contributing to relevant, data science impacting our current world



Usage
-----------------------
* Download the data source from [here](https://github.com/prajnasb/observations/tree/master/experiements/data) and save folder as "Mask Images" 
* Follow the steps listed in image_preprocess.py
* Follow the steps listed in mask_algorithm.py 
* An optional step in mask_algorithm.py is saving and loading the trained Tensorflow models. By saving the final trained weights, the ~10 minute runtime to compile and fit the model each time the script is run can be eliminated
* The webcam output is saved as mask_vid.api



Extending this
-------------------------
* If this model were to be productionized for real-world applications, it could benefit from a more diverse set of images to be trained on
* Additionally, reusable and customized masks were tested against the model in Shaq_test.py and other testing experiments. Interspersing more images like these into the training set could help it better identify non-medical masks
* Training the model on crowds of people would help better assess how well the model could be applied in realistic scenarios
* Creation of a multiclass classification version of the model that could classify images into: yes mask worn and worn correctly, yes mask worn but not correctly, or no mask present. Additional classes could identify what type of mask is being worn, as certain reusable masks like [neck gaiters](https://medical.mit.edu/covid-19-updates/2020/08/how-do-i-choose-cloth-face-mask) can be less effective than other cloth face masks
