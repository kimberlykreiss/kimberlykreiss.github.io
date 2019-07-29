**Who will default on a loan?**

For my final project for my Machine Learning I class, my group and I used data from a microlender called Home Credit (from a Kaggle competition) to predict whether a client would default on a loan. Home Credit operates in several counties including countries in Europe, Asia, and some parts of the United States. It focuses on lending to those that are unbanked and provides access to credit for people with little to no credit. 

One challenge with this problem is that the classes are extremely unbalanced. A dataset with two balanced classes would be evenly divided between the two classes. Instead, our two classes (1-default, 0-not default) are split 92% and 8%, respectively. This is a common problem in machine learning, and we should exercise a degree of caution when interpreting accuracy statistics. For example, if we wrote a model that accurately predicted *every* instance of a client not defaulting on a loan, but could not predict a *single* instance of a client defaulting, we would still get a model with 92% accuracy. 

Our final report write up can be found [here](final_group_report.pdf). Our code was written in python, and can be found in [my github repository](https://github.com/kimberlykreiss/GWU-Machine-Learning-I/blob/master/Final-Project-Group5-master%205/Code/group5_code.py)
