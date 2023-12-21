For our data, we made two versions. 
One version is the original data where we split up the data into two areas. This version of data is stored in the folder named data(orginal) 
The other version involves exchanging the class names 'fire' and 'not fire' for 0 and 1, making it easier to work with for the logistic regression model. This version is stored in the folder named data(recoded)

In this repository, we mainly used three files for our code and graphs. 
The first file is timeseries_test.py, where you will find the code that generated the plots for our time series test. This file uses the datasets "Both.csv" under the folder "data(recoded)" 
The second file is a Jupyter notebook called 'Multicolinearity and logistic regression on single predictors.ipynb' 
This notebook contains the code for our multicollinearity tests and VIF values tests. The notebook also includes correlation tests for our individual predictors. To run this code, you need two datasets: "Bejaia.csv" and "Sidi.csv" under the folder "data(original)".
The final file we use is logregres.py, which contains all the code to generate and create all the plots used in the slides for the logistic regression model.This code file uses the datasets "Bejaia.csv" and "Sidi_BEL-abbes.csv" under the folder "data(recoded)".

The timeseries_test.py and the logregres.py can be executed from the command line to get the results, and the notebook needs to be executed itself. 
For logregres.py, you will need to comment out the tests you don't want to run or want to run in the main function to get the desired outcome.

In another folder you can find the trial figures and tests we ran but not used for the presentation.
