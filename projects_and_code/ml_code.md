### Machine Learning I Project

**Who will default on a loan?**

For my final project for my Machine Learning I class, my group and I used data from a microlender called Home Credit (from a Kaggle competition) to predict whether a client would default on a loan. Home Credit operates in several counties including countries in Europe, Asia, and some parts of the United States. It focuses on lending to those that are unbanked and provides access to credit for people with little to no credit. 

This document will cover some of the challenges of this particular problem and will show how some different machine learning algorithms work on this problem. 

**The data** 

The dataset includes several features including loan type, demographic information about the client, annual income of the client, and the outcome--whether the client defaulted or not. Below is code to import numpy and pandas and read in the data. 

```
import numpy as np
import pandas as pd

df = pd.read_csv("application_train.csv)
```
A data dictionary was provided with the code, so I want to give some summary statistics for demographic information about the clients in the dataset. I need to a do a little bit of cleaning first since the age variable (DAYS_BIRTH) is given as the number of days ago from the date of application that the respondent was born. This is a negative number, so I just divide by 365 and change the sign to get the number of years. I also initialize a vector of demographic variable I'm interested in to throw in a loop and see the percentage of the sample for each. 

```
df['AGE'] = -df['DAYS_BIRTH']/365
demographic_variables = ['CODE_GENDER', 'AGE','NAME_EDUCATION_TYPE', 
                         'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS',
                         'NAME_HOUSING_TYPE']
                         

for i in demographic_variables:
    if i != 'AGE':
        print(i)
        print(df[i].value_counts(normalize=True)*100)
        print('\n')
        
    elif i == 'AGE':
        df.sort_values(by=[i])
        print(i)
        print(df[i].value_counts(normalize=True, bins = 5).sort_index()*100)
        print('\n')
```

The output from this code is below: 
```
CODE_GENDER
F      65.834393
M      34.164306
XNA     0.001301
Name: CODE_GENDER, dtype: float64


AGE
(20.468, 30.238]    15.220919
(30.238, 39.959]    26.076466
(39.959, 49.679]    24.279131
(49.679, 59.4]      21.525734
(59.4, 69.121]      12.897750
Name: AGE, dtype: float64


NAME_EDUCATION_TYPE
Secondary / secondary special    71.018923
Higher education                 24.344820
Incomplete higher                 3.341994
Lower secondary                   1.240931
Academic degree                   0.053331
Name: NAME_EDUCATION_TYPE, dtype: float64


NAME_FAMILY_STATUS
Married                 63.878040
Single / not married    14.778008
Civil marriage           9.682580
Separated                6.429038
Widow                    5.231683
Unknown                  0.000650
Name: NAME_FAMILY_STATUS, dtype: float64


NAME_HOUSING_TYPE
House / apartment      88.734387
With parents            4.825844
Municipal apartment     3.636618
Rented apartment        1.587260
Office apartment        0.851026
Co-op apartment         0.364865
Name: NAME_HOUSING_TYPE, dtype: float64
```
Interestingly, roughly 2/3 of the sample identifies as female. Most live in a house or apartment and around a quarter of the sample has higher education. Almost 2/3 of the sample is married and most live in a house or apartment. Next, I'm interested in seeing what type of loans clients typically get and how common it is to default. 

```
print('Types of loans')
print(df['NAME_CONTRACT_TYPE'].value_counts(normalize=True).sort_index()*100)
print('\n')
print('Default rate')
print(df['TARGET'].value_counts(normalize=True).sort_index()*100)
```    
Again, the output is below:
```
Types of loans
Cash loans         90.478715
Revolving loans     9.521285
Name: NAME_CONTRACT_TYPE, dtype: float64


Default rate
0    91.927118
1     8.072882
Name: TARGET, dtype: float64
```
Over 90% of the loans are cash loans, and the default rate is hovering around 8%. I did not include information about the balance of the loan, since the loans in this dataset happen in several different countries. Thus, several different currencies are captured in this variable, and without normalizing to one currency in particular, a comparison of dollar amounts of these loans doesn't tell us much information. 

One challenge with this dataset is that the classes are extremely unbalanced. A dataset with two balanced classes would be evenly divided between the two classes. Instead, our two classes (1-default, 0-not default) are split 92% and 8%, respectively. This is a common problem in machine learning, and we should exercise a degree of caution when interpreting accuracy statistics. For example, if we wrote a model that accurately predicted *every* instance of a client not defaulting on a loan, but could not predict a *single* instance of a client defaulting, we would still get a model with 92% accuracy. 