# Computer-Vision

## Summary
This repository contains [Bradley Reardon](https://github.com/breardon7), [Divya Parmar](https://github.com/dparmar16) and [Haruna Salim](https://github.com/BABAYEGAR)'s final project for George Washington University's DATS 6203:Machine Learning II course.
Our project objective was to a convolutinal neural network to a real world problem. For our project we applied preprocessing techniques and a convolutinal neural network to classify cars.

## Data
[Data Source 1](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)   

## Folders
* Code contains all of our code and data used in the project.
* Proposal contains the proposal for the project.
* Presentation contains a PowerPoint presentation of our project.
* Report contains a report of our findings from this project.

## Code Execution

1. To successfully execute the code, make sure you have following libraries installed on your python interpreter enviroment:

* sklearn 
* opencv-python
* opencv-contrib-python
* tensoflow 
* keras
* seaborn 
* pandas
* matplotlib

2. After downloading the dataset, copy the images from the car_test into the Dataset/Test folder and the images from the car_train into the Dataset/Train folder.

3. To cut data preprocessing time, you can access the numpy files [here](https://drive.google.com/file/d/1UIDvnY5WKOtZEBPg_BxjFJlT3mzN127d/view?usp=sharing). Simple copy the train_data.npy and test_data.npy into the DataStorage folder in the Code folder.If you also want to reprocess the data, just execute the DataGenerator.py file again.

3. When you run the main.py file. Set either pretrained or custom to true or false or both. 

3. After proprocessing, training, the model will be saved to the SavedModel folder and be used for prediction on the test set.
