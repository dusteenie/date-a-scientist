# Name: Sarah Stepak
# Date: 04.08.2019
# Program: capstone.py
# Purpose: Completes the capstone project!

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Imports classification methods
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# Imports regression methods
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# Imports the data
df = pd.read_csv("profiles.csv")



# ----    ----    ----    ----    ----    ----    ----    ----    ----    ----

# Question 1: Can you predict somebody's age,
#             based on their body type, diet, and overall health?

# ----    ----    ----    ----    ----    ----    ----    ----    ----    ----

# Assigning numeric values to health
# physical health factors
body_type_mapping = {"overweight":0.1666,
                     "a little extra":0.3333,
                     "thin":0.5,
                     "average":0.6666,
                     "athletic":0.8333,
                     "fit":1}

diet_mapping = { "strictly anything":0.1111,
                 "mostly anything":0.2222,
                 "anything":0.3333,
                 "mostly vegan":0.4444,
                 "vegan":0.5555,
                 "strictly vegan":0.6666,
                 "mostly vegetarian":0.7777,
                 "vegetarian":0.8888,
                 "strictly vegetarian":1}

drink_mapping = {"desperately":0.1666,
                 "very often":0.3333,
                 "often":0.5,
                 "socially":0.6666,
                 "rarely":0.8333,
                 "not at all":1}

height_mapping = {"1.0":1,"3.0":1,"4.0":1,
                  "6.0":0.8945,"8.0":0.8945,"9.0":0.8945,
                  "26.0":0.7892,
                  "36.0":0.6839,"37.0":0.6839,
                  "42.0":0.6313,"43.0":0.6313,
                  "47.0":0.5787,"48.0":0.5787,"49.0":0.5787,
                  "50.0":0.5261,"51.0":0.5261,"52.0":0.5261,"53.0":0.5261,"54.0":0.5261,
                  "55.0":0.4735,"56.0":0.4735,"57.0":0.4735,"58.0":0.4735,"59.0":0.4735,
                  "60.0":0.4209,"61.0":0.4209,"62.0":0.4209,"63.0":0.4209,"64.0":0.4209,
                  "65.0":0.3683,"66.0":0.3683,"67.0":0.3683,"68.0":0.3683,"69.0":0.3683,
                  "70.0":0.3157,"71.0":0.3157,"72.0":0.3157,"73.0":0.3157,"74.0":0.3157,
                  "75.0":0.2631,"76.0":0.2631,"77.0":0.2631,"78.0":0.2631,"79.0":0.2631,
                  "80.0":0.2105,"81.0":0.2105,"82.0":0.2105,"83.0":0.2105,"84.0":0.2105,
                  "85.0":0.1579,"86.0":0.1579,"87.0":0.1579,"88.0":0.1579,"89.0":0.1579,
                  "90.0":0.1053,"91.0":0.1053,"92.0":0.1053,"93.0":0.1053,"94.0":0.1053,
                  "95.0":0.0526}

smoke_mapping = {"yes":0.1666,
                 "when drinking":0.3333,
                 "sometimes":0.5,
                 "trying to quit":0.6666,
                 "no":0.8333,
                 "never":1}

drugs_mapping = {"often":0.3333,
                 "sometimes":0.6666,
                 "never":1}


#mental health factors
child_mapping = {"doesn&rsquo;t want kids":0.1666,
                 "doesn&rsquo;t have kids":0.3333,
                 "doesn&rsquo;t have kids, but might want them":0.5,
                 "doesn&rsquo;t have kids, but wants them":0.6666,
                 "has a kid":0.8333,
                 "has kids":1}

education_mapping = {"graduated from high school":0.125,
                     "working on space camp":0.25,
                     "graduated from space camp":0.375,
                     "working on two-year college":0.5,
                     "working on college/university":0.75,
                     "graduated from college/university":0.75,
                     "working on masters program":0.875,
                     "graduated from masters program":8}

pets_mapping = {"likes cats":0.1429, "likes dogs":0.1429,
                "likes dogs and dislikes cats":0.2858,
                "dislikes dogs and likes cats":0.2858,
                "likes dogs and likes cats":0.4287,
                "has cats":0.5713, "has dogs":0.5713,
                "has dogs and dislikes cats":0.7142,
                "dislikes dogs and has cats":0.7142,
                "has dogs and likes cats":0.8571,
                "likes dogs and has cats":0.8571,
                "has dogs and has cats":1}

income_mapping = {"20000":0.0833,
                  "30000":0.1666,
                  "40000":0.2499,
                  "60000":0.3332,
                  "50000":0.4165,
                  "70000":0.4998,
                  "80000":0.5831,
                  "100000":0.6664,
                  "150000":0.7497,
                  "250000":0.833,
                  "500000":0.9163,
                  "1000000":1}

profession_mapping = {"unemployed":0.125,
                      "rather not say":0.25,
                      "other":0.375,
                      "student":0.5,
                      "transportation":0.625,
                      "science / tech / engineering":0.75,
                      "computer / hardware / software":0.75,
                      "artistic / musical / writer":0.75,
                      "sales / marketing / biz dev":0.75,
                      "medicine / health":0.75,
                      "education / academia":0.75,
                      "executive / management":0.75,
                      "banking / financial / real estate":0.75,
                      "entertainment / media":0.75,
                      "law / legal services":0.75,
                      "hospitality / travel":0.75,
                      "construction / craftsmanship":0.75,
                      "clerical / administrative":0.75,
                      "political / government":0.75,
                      "military":0.875,
                      "retired":1}


# Data mapping
df["body_type_data"] = df.body_type.map(body_type_mapping)
df["child_data"] = df.offspring.map(child_mapping)
df["diet_data"] = df.diet.map(diet_mapping)
df["drinks_data"] = df.drinks.map(drink_mapping)
df["drugs_data"] = df.drugs.map(drugs_mapping)
df["education_data"] = df.education.map(education_mapping)
df["height_data"] = df.height.map(height_mapping)
df["income_data"] = df.income.map(income_mapping)
df["job_data"] = df.job.map(profession_mapping)
df["pet_data"] = df.pets.map(pets_mapping)
df["smokes_data"] = df.smokes.map(smoke_mapping)


# Splitting the data
graph_scale = df[["body_type_data","diet_data","drinks_data","drugs_data","height_data","smokes_data","pet_data","job_data","income_data","education_data","child_data"]]
graph_scale = graph_scale.replace(np.nan, 0,regex=True)
x=graph_scale.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
graph_scale = pd.DataFrame(x_scaled, columns=graph_scale.columns)

x = graph_scale
y = df[["age"]]
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.8,
                                                    test_size = 0.2,
                                                    random_state=100)


# Ridge Regression methods
rr = Ridge(alpha=0.0001,normalize=True)
rr.fit(x_train, y_train)
y_predict = rr.predict(x_test)
plt.scatter(y_test, y_predict, alpha=0.4)

# Plots the data
plt.xlabel("Age")
plt.ylabel("Predicted Age")
plt.title("Actual vs Predicted Age, Based on Overall Health Factors")
plt.show()

# Accuracy and Precision
print(plt.score(x_train,y_train))
print(plt.score(x_test,y_test))

#Multi-Linear Regression
#mlr = LinearRegression()
#mlr.fit(x_train, y_train)
#y_predict = mlr.predict(x_test)

#Modeling the Data
#model=mlr.fit(x_train, y_train)
#y_predict = mlr.predict(x_test)
#plt.scatter(y_test, y_predict, alpha=0.4)


# ----    ----    ----    ----    ----    ----    ----    ----    ----    ----

# Question 2: Can you predict somebody's sex,
#             based on their sign and their income level?

# ----    ----    ----    ----    ----    ----    ----    ----    ----    ----

sex_mapping = {"m":0, "f":1}

sign_mapping = {"capricorn and it&rsquo;s fun to think about":1,
                "capricorn but it doesn&rsquo;t matter":1,
                "capricorn":1,
                "capricorn and it matters a lot":1,
                "aquarius and it&rsquo;s fun to think about":2,
                "aquarius but it doesn&rsquo;t matter":2,
                "aquarius":2,
                "aquarius and it matters a lot":2,
                "pisces and it&rsquo;s fun to think about":3,
                "pisces but it doesn&rsquo;t matter":3,
                "pisces":3,
                "pisces and it matters a lot":3,
                "aries and it&rsquo;s fun to think about":4,
                "aries but it doesn&rsquo;t matter":4,
                "aries":4,
                "aries and it matters a lot":4,
                "taurus and it&rsquo;s fun to think about":5,
                "taurus but it doesn&rsquo;t matter":5,
                "taurus":5,
                "taurus and it matters a lot":5,
                "gemini and it&rsquo;s fun to think about":6,
                "gemini but it doesn&rsquo;t matter":6,
                "gemini":6,
                "gemini and it matters a lot":6,
                "cancer and it&rsquo;s fun to think about":7,
                "cancer but it doesn&rsquo;t matter":7,
                "cancer":7,
                "cancer and it matters a lot":7,
                "leo and it&rsquo;s fun to think about":8,
                "leo but it doesn&rsquo;t matter":8,
                "leo":8,
                "leo and it matters a lot":8,
                "virgo and it&rsquo;s fun to think about":9,
                "virgo but it doesn&rsquo;t matter":9,
                "virgo":9,
                "virgo and it matters a lot":9,
                "libra and it&rsquo;s fun to think about":10,
                "libra but it doesn&rsquo;t matter":10,
                "libra":10,
                "libra and it matters a lot":10,
                "scorpio and it&rsquo;s fun to think about":11,
                "scorpio but it doesn&rsquo;t matter":11,
                "scorpio":11,
                "scorpio and it matters a lot":11,
                "sagittarius and it&rsquo;s fun to think about":12,
                "sagittarius but it doesn&rsquo;t matter":12,
                "sagittarius":12,
                "sagittarius and it matters a lot":12}

#Maps the data]
df["gender_data"] = df.sex.map(sex_mapping)
df["sign_data"] = df.sign.map(sign_mapping)

# Splitting the data
graph_scale = df[["body_type_data","sign_data"]]
graph_scale = graph_scale.replace(np.nan, 0,regex=True)
x=graph_scale.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
graph_scale = pd.DataFrame(x_scaled, columns=graph_scale.columns)
x = graph_scale
y = df[["gender_data"]]
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size = 0.8,
                                                    test_size = 0.2,
                                                    random_state=100)
# Support Vector
#classifier = SVC(kernel = 'linear')
#classifier.fit(x_train,y_train)
#y_predict = classifier.predict(x_test)

# K-Neighbors
classifier = KNeighborsClassifier(n_neighbors = 80)
classifier.fit(x_train,y_train)
y_predict = classifier.predict(x_test)

# Plots the data
plt.xlabel("Gender")
plt.ylabel("Predicted Gender")
plt.title("Actual vs Predicted Gender")
plt.plot(y_test,y_predict)
plt.show()

# Accuracy, Precision, Recall, F1
print(accuracy_score(y_test,y_predict))
print(recall_score(y_test,y_predict))
print(precision_score(y_test,y_predict))
print(f1_score(y_test,y_predict))
