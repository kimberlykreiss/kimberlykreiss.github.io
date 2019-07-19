### Who will default on a loan? 

For my final project for my Machine Learning I class, my group and I used data from a microlender called Home Credit (from a Kaggle competition) to predict whether a client would default on a loan. Home Credit operates in several counties including countries in Europe, Asia, and some parts of the United States. It focuses on lending to those that are unbanked and provides access to credit for people with little to no credit. 

This document will cover some of the challenges of this particular problem and will show how some different machine learning algorithms work on this problem. 

### The data 
The dataset includes several features including loan type, demographic information about the client, annual income of the client, and the outcome--whether the client defaulted or not. Below are some summary statistics of the dataset: 

```
import numpy as np
import pandas as pd

df = pd.read_csv("application_train.csv)
```