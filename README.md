# ECE1512-2023F-ProjectRepo-YufanWang
Description: The GitHub is used to display all the coding I used in ECE1512 Project A assignment. Task_1_complete.py includes the Teacher Model Test Accuracy, Student Model Test Accuracy and its plot under different temperatures. Also it has the result of Student Model Test Accuracy without KD. Task_1_12.py prints the outcomes of including Teacher Assistant in the training. The outcomes include Teacher Model Test Accuracy, Teacher Assistant Model Test Accuracy, and Student Model Test Accuracy. Task_1_12_Compared.py prints the outcomes of Teacher Model Test Accuracy and Student Model Test Accuracy to compare the data with the method introducing TA. Task_2.py prints the results of Teacher Model Test Accuracy ,Student Model Test Accuracy, and its plot with different temperatures. Task_2_12.py introduces the TA to compare the results with the original one. Task_2_without KD.py generates the results of Teacher Model Test Accuracy and Student Model Tesst Accuracy without KD.


Requirement:

In Task_1 codes (Task_1_complete.py, Task_1_12_Compared.py, Task_1_12.py). They need to import the TensorFlow library, TensorFlow's Keras API, MNIST dataset from TensorFlow's built-in datasets, Adam, and pyplot module from Matplotlib. Ensure that TensorFlow and Matplotlib are installed in your Python environment. These imports allow for building neural network models, accessing the MNIST dataset, optimizing models during training, and visualizing data and results.

      pip install tensorflow numpy pandas scikit-learn

      
In Task_2 codes (Task_2.py, Task_2_12.py, Task_2_without KD.py). Ensure TensorFlow, NumPy, Pandas, and scikit-learn are installed in the Python environment. Notably, running the codes needs to set the path to the required MHIST dataset which I uploaded to my desktop and google drive. You have to get the dataset first, and change the path according to your condition.

      pip install tensorflow scikit-learn pandas numpy matplotlib
