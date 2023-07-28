# Rotational-Rectification-Network

Contents
cv_MNIST.ipynb: Jupyter Notebook containing the Python code for loading the MNIST dataset, applying rotation augmentation, and training different models for evaluation.
Caltech Dataset.ipynb: Python script for preprocessing the custom dataset and applying rotation augmentation.

Requirements
Python 3.x
TensorFlow 2.x
Keras
NumPy
PIL (Python Imaging Library)
Matplotlib
OpenCV (cv2)

Note: You can generate your own Caltech dataset with the below steps, or you can download the dataset from the following link: https://drive.google.com/file/d/1lTo4Y4rDe8PNxlC1Sk43rmDTs4YiSKP7/view?usp=sharing

Instruction to Generate Caltech Dataset 
Organize the dataset into video sequences with corresponding image and annotation folders. The dataset folder structure should look like this:
https://drive.google.com/file/d/1lVRbAKqRe4KRcLqvQi2FPRdfGfBykAqb/view?usp=sharing

In the CV Caltech 2.ipynb file, update the source and destination directory path in the “source_dir” and “destination_dir” variables.
In the third cell updated the path in the following variables “training_Path_To_Rotate”, ”testing_Path_To_Rotate”, ”rotated_training_path”, ”rotated_testing_path”,  ”rotated_train_json_angle_path” and “rotated_test_json_angle_path.”

Instruction To Generate the Annotations
We have used caltech_pedestrian_extractor, available at https://github.com/dbcollection/caltech_pedestrian_extractor
Download the converter.py file and run the code with the below command
python converter.py -data_path <path_to_data> -save_path <path_to_extract>


Instructions For cv_MNIST
Make sure you have all the required libraries installed. If not, you can install them using pip install <library_name>.
Execute the cv_MNIST.ipynb Jupyter Notebook cell by cell to run the code and observe the results.
The notebook will display visualizations of the MNIST dataset, rotated images, and the loss graph for each model.
You can modify the model architectures or hyperparameters to experiment with different configurations.
