# %% [markdown]
# ### Part One

# %% [markdown]
# #### Q1

# %%
# Loading all just to be safe
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import altair as alt
import streamlit as st 

# Read data
s = pd.read_csv("social_media_usage.csv")

# Dimensions
# s.shape

# %% [markdown]
# ***

# %% [markdown]
# #### Q2

# %%
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# toy_df = pd.DataFrame({'col1': [0, 1, 3], 'col2': [1, 1, 0]})

# toy_df_clean = toy_df.applymap(clean_sm)
# toy_df_clean


# %% [markdown]
# ***

# %% [markdown]
# #### Q3

# %%
# Apply clean_sm on columns
ss = pd.DataFrame({
    "sm_li": clean_sm(s["web1h"]),
    "income": np.where(s["income"] <= 9, s["income"], np.nan),
    "education": np.where(s["educ2"] <= 8, s["educ2"], np.nan),
    "parent": clean_sm(s["par"]),
    "married": clean_sm(s["marital"]),
    "female": np.where(s["gender"] == 2, 1, 0),
    "age": np.where(s["age"] <= 97, s["age"], np.nan)
}).dropna()

ss.head(20)


# # %%
# ss_visuals = ss.groupby(["age", "education"], as_index=False)["sm_li"].mean()

# ss_chart = alt.Chart(ss_visuals).mark_circle().encode(
#     x="age",
#     y="sm_li",
#     color="education:N"
# ).properties(
#     title="LinkedIn Usage by Age and Education Level",
#     width=600,
#     height=400
# )

# ss_chart

# %% [markdown]
# Our exploratory analysis finds that when looking into LinkedIn Usage by Age and Education levels, there are trends that are found. We can see that people who are younger are more likely to use LinkedIn than those who are older. Another group who is more likely to use LinkedIn is those who are more highly educated.

# %% [markdown]
# ***

# %% [markdown]
# #### Q4

# %%
y = ss["sm_li"]
X = ss.drop(columns=["sm_li"])

print(X.shape, y.shape)

# %% [markdown]
# ***

# %% [markdown]
# #### Q5

# %%
# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X.values,
                                                    y,
                                                    stratify=y,
                                                    test_size=0.2,
                                                    random_state=987)

# %% [markdown]
# X_train is our training feature set that is 80% of x, y_train is the target vector of our training set, X_test is our testing feature set that is 20% of x, y_test is the target vector of our testing set. These objects are used in machine learning to break up our data to see trends and coorelations. By breaking them up into different sets, we can see two different sizes of samples and see how that might effect our data. We can also run tests on both sets and see if they have similar results.

# %% [markdown]
# ***

# %% [markdown]
# #### Q6

# %%
# Regression Model and Class Weight fit
lr = LogisticRegression(class_weight="balanced", random_state=987)

# Fit Logistic Model to training data
lr.fit(X_train, y_train)

# %% [markdown]
# ***

# %% [markdown]
# #### Q7

# %%
# Predictions based on testing data
y_pred = lr.predict(X_test)

pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted : No", "Predicted : Yes"],
            index=["Actual : No","Actual : Yes"]).style.background_gradient(cmap="PiYG")

# %% [markdown]
# From the Confusion Matrix we can see that our model correctly saw 111 non-LinkedIn users as non-users, and 63 LinkedIn users as LinkedIn users. The model incorrectly saw 57 non-LinkedIn users as LinkedIn users, and 21 LinkedIn users as non-LinkedIn users. 

# %%
# Get other metrics with classification_report
print(classification_report(y_test, y_pred))

# %% [markdown]
# We can see that accuracy for the model is 69%

# %% [markdown]
# ***

# %% [markdown]
# #### Q8

# %%
pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted : Non-LinkedIn User", "Predicted : LinkedIn User"],
            index=["Actual : Non-LinkedIn User","Actual : LinkedIn User"]).style.background_gradient(cmap="PiYG")

# %% [markdown]
# ***

# %% [markdown]
# #### Q9

# %%
# # precision: TP/(TP+FP)
# percision = 63/(63+57)
# percision

# %% [markdown]
# This metric depicts when our modek predicts if a user is a LinkedIn user and is correct 53% of the time. An actual example of when this would be the preffered metric of evaluation is CAPTCHAs. CAPTCHAS are those website systems that make you solve some sort of puzzle to ensure that you are human. In this case, percision would used as a metric to confirm that robots are actual robots.

# %%
# # recall: TP/(TP+FN)
# recall = 63/(63+21)
# recall

# %% [markdown]
# This metric depicts that our model correctly predicts 75% of actual LinkedIn users. An actual example of when this would be the preffered metric of evaluation is during TSA Screening. In this case, TSA screenings are used to flag items that are potentially dangerous which can be determined by the metric of recall. Some of these items can be dangerous but also some of them may not be at all and just trigger the system. 

# %%
# f1_score = 2 * (percision * recall) / (percision + recall)
# f1_score = round(f1_score,2)
# f1_score

# %% [markdown]
# Our F1 Score comes in at 62% depicting how recall and precision interact in our model under one variable going off of each other. An actual example of when this would be the preffered metric of evaluation is when your bank may not allow you to make a purchase because they believe it is fraud. F1 Score would be the metric in this case because the bank is seeing the transaction as possibly false. If this is the case, they are blocking a fraud charge, but if not you can just confirm the purchase. 

# %%
# Get other metrics with classification_report
print(classification_report(y_test, y_pred))

# %% [markdown]
# Accuracy comes in at 0.69 based on our model and the testing data. We see that percision corretly predicts 84% of non-LinkedIn users and 53% of LinkedIn users. Based on Recall score, we can see 66% correctly sees the actual non-LinkedIn users and 75% of actual LinkedIn users. 

# %% [markdown]
# ***

# %% [markdown]
# #### Q10

# %%
# New data for predictions
newdata = pd.DataFrame({
    "income": [8, 8],
    "education": [7, 7],
    "parent": [0, 0],
    "married": [1, 1],
    "female" : [1, 1],
    "age" : [42,82]
})


# %%
# Use model to make predictions
newdata["prediction"] = lr.predict(newdata)

# %%
# New data for features: income, education, parent, married, female, age=42
person = [8, 7, 0, 1, 1, 42]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

# %%
# Print predicted class and probability
print(f"Predicted class: {predicted_class[0]}") # 0=non-LinkedIn User, 1=LinkedIn User
print(f"Probability that this person is a LinkedIn User: {probs[0][1]}")

# %% [markdown]
# Based on all the requirements and age equaling 42, the probability this person uses LinkedIn is 73%

# %%
# New data for predictions
newdata = pd.DataFrame({
    "income": [8, 8],
    "education": [7, 7],
    "parent": [0, 0],
    "married": [1, 1],
    "female" : [1, 1],
    "age" : [42,82]
})

# %%
# New data for features: income, education, parent, married, female, age=82
person = [8, 7, 0, 1, 1, 82]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

# %%
# Print predicted class and probability
print(f"Predicted class: {predicted_class[0]}") # 0=non-LinkedIn User, 1=LinkedIn User
print(f"Probability that this person is a LinkedIn User: {probs[0][1]}")

# %% [markdown]
# Based on all the requirements and age equaling 82, the probability this person uses LinkedIn is 47%

# %% [markdown]
# ***

# %% [markdown]
# ### Part Two

# %%

# Title
st.title("LinkedIn User Prediction")

# Education Input
education = st.selectbox("Education level", options=["High School Diploma", "College Degree", "Graduate Degree"])
if education == "High School Diploma":
    education_label = "High School Diploma"
    education = 1
elif education == "College Degree":
    education_label = "College Degree"
    education = 2
else:
    education_label = "Graduate Degree"
    education = 3

# Income Input
income = st.slider(label="Income (1=low, 9=high)", min_value=1, max_value=9, value=5)
if income <= 3:
    income_label = "Low Income"
elif 4 <= income <= 6:
    income_label = "Middle Income"
else:
    income_label = "High Income"

# Age Input
age = st.slider(label="Age", min_value=18, max_value=97, value=35)
if age < 30:
    age_label = "Young-Age"
elif 30 <= age <= 50:
    age_label = "Mid-Age"
else:
    age_label = "Senior-Age"


# Parents Input
parent = st.selectbox("Parent Status", options=["No", "Yes"])
if parent == "No":
    parent_label = "Non-Parent"
    parent = 0
else:
    parent_label = "Parent"
    parent = 1

# Married Input
married = st.selectbox("Marital Status", options=["Not Married", "Married"])
if married == "Not Married":
    marital_label = "Not Married"
    married = 0
else:
    marital_label = "Married"
    married = 1

# Gender Input
female = st.selectbox("Gender", options=["Male", "Female"])
if female == "Male":
    gender_label = "Male"
    female = 0
else:
    gender_label = "Female"
    female = 1

# Display Inputs
st.write(f"Income level: {income_label}")
st.write(f"Education level: {education_label}")
st.write(f"Age group: {age_label}")
st.write(f"Parent status: {parent_label}")
st.write(f"Marital status: {marital_label}")
st.write(f"Gender: {gender_label}")

# Input DataFrame
input_data = pd.DataFrame({
    "education": [education],
    "income": [income],
    "age": [age],
    "parent": [parent],
    "married": [married],
    "female": [female]
})

# Reorder columns to match training data
input_data = input_data[["income", "education", "parent", "married", "female", "age"]]

# Prediction
if st.button("Predict LinkedIn Usage"):
        prediction = lr.predict(input_data)[0]
        probability = lr.predict_proba(input_data)[0][1]

        # Show Prediction
        if prediction == 1:
            st.success("This person is predicted to be a LinkedIn user.")
        else:
            st.error("This person is predicted to NOT be a LinkedIn user.")
        st.write(f"Probability of using LinkedIn: {probability :.2%}")