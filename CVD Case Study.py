#!/usr/bin/env python
# coding: utf-8

# # CVD Case Study
# 
# Cardiovascular disease (CVD) is a leading cause of death globally. It affects millions of people annually. It encompasses a range of conditions affecting the heart and blood vessels, such as heart attacks, strokes, and hypertension. 
# 
# The goal is to analyse the Cardiovascular Disease Dataset from kaggle.com and identify key factors contributing to cardiovascular disease risk, predict heart disease using classification techniques and discuss findings.
# 
# The Cardiovascular Disease Dataset contains various patient health metrics, including age, gender, height, weight, blood pressure (systolic and diastolic), cholesterol, glucose and lifestyle factors such as smoking, alcohol consumption, and physical activity.
# 
# ### Hypothesis:
# Higher cholesterol levels, increased age, elevated blood pressure, higher body mass index (BMI), higher glucose levels, diabetes, lifestyle choices (smoking, alcohol, physical activity) are strong predictors of cardiovascular disease.
# ### Analyses:
# 1. Examine the relationship between cholesterol levels and the presence of cardiovascular desease
# 2. Compare the prevalence of cardiovascular disease across different age groups, between males and females
# 3. Investigate the correlation between systolic blood pressure and cardiovascular disease prevalence
# 4. Calculate BMI from height and weight data ana analyze its association with cardiovascular disease
# 5. Explore the relationship between glucose levels and cardiovascular disease status
# 6. Compare the prevalence of cardiovascular disease among individuals who smoke, drink alcohol, physically active vs. those who do not
# 8. Conclusion of all findings
# 

# ## Steps of the Case Study
# 
# 1. Import Libraries and load the Dataset
# 2. Data Information and Exploratory Data Analysis (EDA)
# 3. Machine Leaning Models (building, evaluating and interpretating of findings)
# 4. Conclusion and Recommendation

# ### Data preprocessing and EDA
# 
# 1. Handling missing data and removing duplicates
# 2. Removing and correcting errorneous outliers (unrealistic blood pressure values)
# 3. Additing new features and categorizing into groups for more detailed analysis of its effect on cardiovascular disease
# 
# EDA was performed to understand the distribution of key variables such as age, cholesterol, and blood pressure. Relationships were examined between different risk factors and cardiovascular disease using heatmaps and pair plots for visualization. 

# ### Methodology
# Classification algorithms used
# 1. Logistic Regression to understand the impact of individual features (presence or absence of heart disease)
# 2. Random Forest to capture complex nonlinear relationships in the data which might improve prediction accuracy
# 
# ### Model evaluation
# 1. Used accuracy, precision, recall, F1 score, and ROC-AUC to assess model performance
# 2. Performed cross-validation to ensure the robustness of results
# 3. Plotted ROC curves to visualize model performance
# 

# # 1. Import Libraries and Load the Dataset

# In[1]:


# Import necessary libraries to evaluate the data and feed the models

# for data manipulation
import pandas as pd # for data analysis
import numpy as np #for numerical data

# for data visualization
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical plot
import pylab as plot

get_ipython().run_line_magic('matplotlib', 'inline')
import math
import warnings # to suppress any warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# machine learning models
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics # for getting model performance
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# # 2. Data Information and EDA

# In[122]:


df = pd.read_csv(r'C:\Users\Хамида\Downloads\cardio_train.csv', sep=';') 


# In[3]:


# creating a copy of the dataset
data=df.copy()
df


# In[4]:


df.set_index("id" , inplace=True)


# In[5]:


df


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.info()


# In[9]:


df.ndim


# In[10]:


df.describe()


# In[11]:


# missing values
df.isnull().sum()


# ### Observations
# 
# The data is loaded correctly. The data is very informative and is represented by 70 000 patients on 12 criteria. The dataset contains various patient health metrics as major risk factors. They include age, gender, height, weight, blood pressure (systolic and diastolic), cholesterol, glucose and lifestyle factors such as smoking, alcohol consumption, and physical activity.
# 
# Data types: 12 columns are of type int64 (integer). 1 column (weight) is of type float64 (float).
# The dataset uses 6.9 MB of memory.

# In[12]:


# correlation matrix to visualize relationships
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(12, 7))
plt.imshow(corr, cmap='Blues')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr)):
    for j in range(len(corr)):
        if not mask[i, j]:
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black')

plt.show()


# In[13]:


df.cholesterol.replace([1,2,3] ,[" normal" ,  "above normal", "well above normal"],inplace=True)


# In[14]:


df.gluc.replace([1,2,3] , ["normal", "above normal","well above normal"], inplace=True)


# In[15]:


df.smoke.replace([0,1] , ["No" , "Yes"] , inplace=True)


# In[16]:


df.gender.replace([1,2],["Women"  ,"Men"] , inplace=True)


# In[17]:


df['alco'].replace({0: "Doesn't Drink", 1: "Drink"} , inplace=True )


# In[18]:


df['active'].replace({0: 'Inactive', 1: 'Active'} , inplace=True )


# In[19]:


df['cardio'].replace({0: 'No Disease', 1: 'Has Disease'} , inplace=True )


# In[20]:


df.info()


# In[21]:


df


# In[22]:


# number of unique values
df.nunique()


# In[23]:


df.ap_hi.describe()


# In[24]:


df[df.ap_hi < 0]


# In[25]:


df.ap_lo.describe()


# In[26]:


df[df.ap_lo < 0]


# In[27]:


df= df[(df['ap_hi'] >= 0) & (df['ap_lo'] >= 0)]
data= data[(data['ap_hi'] >= 0) & (data['ap_lo'] >= 0)]


# In[28]:


plt.boxplot(df.ap_hi);


# In[29]:


plt.boxplot(df.ap_lo);


# In[30]:


df[df.ap_lo > 180].sort_values(by="ap_lo")["ap_lo"]


# In[31]:


df[df.ap_hi >220].sort_values(by="ap_hi")["ap_hi"].head(11)


# In[32]:


df[df.ap_hi >20].sort_values(by="ap_hi")["ap_hi"]


# ### Observations
# Systolic blood pressure less than 25 mmHg is not possible in a medical condition. 

# In[33]:


# removing outliers
df=df[ (df.ap_lo <=190)  & (df.ap_hi > 25)& (df.ap_hi <= 240) ]
data=data[ (data.ap_lo <=190)  & (data.ap_hi > 25)& (data.ap_hi <= 240) ]


# In[34]:


df


# In[35]:


df.shape


# In[36]:


plt.boxplot(df.ap_hi);


# In[37]:


plt.boxplot(df.ap_lo);


# ### Observations
# When diastolic blood pressure falls below 20 mmHg, it indicates a serious condition that lead to dangerously blood pressure (heart failure, major blood loss, extreme dehydration, sepsis).
# When diastolic blood pressure exceeds 120-130 mmHg, it indicates severe health issues (hypertensive crises during heart attacks, Cushing's syndrome (with high cortisol), adrenal tumors (pheochromocytoma), acute kidney disease, severe atherosclerosis, and congestive heart failure. In extreme situations, diastolic pressure may reach 190 mmHg or higher, signaling a critical medical emergency requiring immediate attention.

# In[38]:


corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(12, 7))
plt.imshow(corr, cmap='Blues')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr)):
    for j in range(len(corr)):
        if not mask[i, j]:
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black')

plt.show()


# In[39]:


plt.figure(figsize=(8,6)) 
sns.scatterplot(x="ap_hi", y="ap_lo", data=df,  alpha=0.6)

plt.title("Scatter Plot of Systolic (ap_hi) vs Diastolic (ap_lo) Blood Pressure")
plt.xlabel("Systolic Blood Pressure (ap_hi)")
plt.ylabel("Diastolic Blood Pressure (ap_lo)")


# In[40]:


correlation = df[['ap_hi', 'ap_lo']].corr().iloc[0, 1]
print(f"The correlation coefficient between ap_hi and ap_lo is: {correlation:.2f}")


# In[41]:


def categorize_blood_pressure(ap_hi, ap_lo):
    if ap_hi < 120 and ap_lo < 80:
        return 'Normal'
    elif 120 <= ap_hi < 140 or 80 <= ap_lo < 90:
        return 'Prehypertension'
    elif 140 <= ap_hi < 160 or 90 <= ap_lo < 100:
        return 'Hypertension Stage 1'
    elif 160 <= ap_hi or ap_lo >= 100:
        return 'Hypertension Stage 2'
    elif ap_hi >= 180 or ap_lo >= 120:
        return 'Hypertensive Crisis'
        
df['blood_pressure_category'] = df.apply(lambda row: categorize_blood_pressure(row['ap_hi'], row['ap_lo']), axis=1)
df[['ap_hi', 'ap_lo', 'blood_pressure_category']].head(10)


# In[43]:


df['age_years'] = (df['age'] / 365).round().astype(int)
print(df)


# In[44]:


df['age_years']


# In[45]:


df['age_years'].describe()


# In[46]:


df['age_years'].hist(edgecolor="white",grid=False, color="#81DAE3")


# In[47]:


df[df['age_years'] == df['age_years'].max()]


# In[48]:


df[df['age_years'] ==df['age_years'].min()]


# In[49]:


df.gender.value_counts().plot(kind ="pie" , autopct='%1.1f%%', colors=["#FFE5CF" , "#CDC2A5"], explode=(0,0), wedgeprops=dict(width=0.7));


# In[50]:


sns.histplot(data=df, x='age_years', hue='gender', kde=False, bins=25, palette=["#7CF5FF" ,"#FFD7C4"])

plt.title('Age Distribution by Gender')
plt.xlabel('Age (Years)')
plt.ylabel('Count')


# ### 
# BMI = mass(kg) / (height) ^ 2(m)

# In[51]:


df['Bmi'] = round( df['weight'] / ((df['height'] / 100) ** 2) , 2)


# In[52]:


df.Bmi


# In[53]:


df.Bmi.describe()


# In[54]:


df[['Bmi', 'weight', 'height']].corr()


# In[55]:


# display the correlation matrix between the columns 'Bmi', 'weight', and 'height'

# sns.heatmap(df[['Bmi', 'weight', 'height']].corr() , annot=True, fmt=".1f")

corr = df[['Bmi', 'weight', 'height']].corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
plt.figure(figsize=(8, 5))
plt.imshow(corr, cmap='coolwarm')
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr)):
    for j in range(len(corr)):
        if not mask[i, j]:
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha='center', va='center', color='black')

plt.show()


# In[56]:


plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(x='weight', y='Bmi', data=df)
plt.title('Bmi vs Weight')
plt.xlabel('Weight')
plt.ylabel('Bmi')
plt.subplot(1, 2, 2)
sns.scatterplot(x='height', y='Bmi', data=df)
plt.title('Bmi vs Height')
plt.xlabel('Height')
plt.ylabel('Bmi')


plt.tight_layout()
plt.show()


# In[57]:


def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Wightloss'
    elif 18.5 <= bmi < 25:
        return 'Normal'
    elif 25 <= bmi < 30:
        return 'Overwight'
    elif 30 <= bmi < 35:
        return 'Obesity class 1'
    elif 35<= bmi < 40:
        return 'Obesity class 2'
    else: 
        return 'Extreme Obesity'

df['BMI_category'] = df['Bmi'].apply(categorize_bmi)


# In[58]:


df['BMI_category']


# In[59]:


df['BMI_category'].unique()


# In[60]:


colors = plt.get_cmap('Pastel1_r').colors
df.BMI_category.value_counts().plot(kind ="pie" , autopct='%1.1f%%', colors=colors, explode=(0,0,0,0,0,0), wedgeprops=dict(width=0.7));


# In[61]:


df_grouped = df.groupby("blood_pressure_category")["BMI_category"].value_counts()
df_grouped


# In[62]:


plt.figure(figsize=(12,5))  
sns.countplot(x="blood_pressure_category", hue="BMI_category", data=df, palette=["#CAF4FF", "#5AB2FF","#FFC7ED" ,"#FFE9D0" ,"#E7D4B5","c"] , edgecolor="black"
)

plt.title("Distribution of blood pressure Levels by BMI")
plt.xlabel("blood_pressure")
plt.ylabel("Count")


plt.show()


# In[63]:


df.cholesterol.value_counts()


# In[64]:


colors = plt.get_cmap('Pastel1_r').colors
df.cholesterol.value_counts().plot(kind ="pie" , autopct='%1.1f%%', colors=colors, explode=(0,0,0), wedgeprops=dict(width=0.7));


# ### Observations
# Cholesterol levels has hree categories: Normal, Above normal and Well above normal. The last one may lead to a serious risk for heart disease and needs medical treatment or lifestyle adjustments.

# In[65]:


df_grouped = df.groupby("cholesterol")["BMI_category"].value_counts()
df_grouped


# In[66]:


plt.figure(figsize=(12,5))  
sns.countplot(x="BMI_category", hue="cholesterol", data=df, palette=["#F1DEC6" , "#FFD7C4","#FFDBB5"],edgecolor="black"
)

plt.xlabel("BMI_category")
plt.ylabel("Count")
plt.show()


# In[67]:


df_grouped = df.groupby("cholesterol")["BMI_category"].value_counts()
df_grouped


# In[68]:


df.gluc.value_counts()


# In[69]:


colors = plt.get_cmap('Pastel1_r').colors
df.gluc.value_counts().plot(kind ="pie" , autopct='%1.1f%%', colors=colors, explode=(0,0,0), wedgeprops=dict(width=0.7));


# ### Observations
# Blood glucose levels: Normal, Above normal (pre-diabetes) and Well above normal (diabetes) that require urgent medical attention.

# In[70]:


data[["gluc" , "cholesterol"]].corr(numeric_only=True)


# In[71]:


df.groupby("gluc")["cholesterol"].value_counts()


# In[72]:


plt.figure(figsize=(6,5))  
sns.countplot(x="gluc", hue="cholesterol", data=df, palette="pastel")

plt.title("Distribution of Cholesterol Levels by Gluc Levels")
plt.xlabel("Gluc Levels")
plt.ylabel("Count")
plt.show()


# In[73]:


df.groupby(["gender" , "gluc"])["cholesterol"].value_counts()


# In[74]:


df.smoke.value_counts()


# In[75]:


df.smoke.value_counts().plot(kind ="pie" , autopct='%1.1f%%', colors=["#FFE5CF" , "#CDC2A5"], explode=(0,0), wedgeprops=dict(width=0.7));


# In[76]:


df.groupby("gender")["smoke"].value_counts()


# In[77]:


sns.countplot(x="gender", hue="smoke", data=df, palette="pastel")

plt.title("Distribution of Smoking Status by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()


# In[78]:


df.alco.value_counts().plot(kind ="pie" , autopct='%1.1f%%', colors=["#FFE5CF" , "#CDC2A5"], explode=(0,0), wedgeprops=dict(width=0.7));


# In[79]:


data[["alco" , "smoke"]].corr()


# In[80]:


df.groupby("smoke")["alco"].value_counts()


# In[81]:


sns.barplot(x="smoke", y="count", hue="alco", data=df.groupby(["smoke", "alco"]).size().reset_index(name='count'), palette="pastel")

plt.title("Distribution of Alcohol Consumption by Smoking Status")
plt.xlabel("Smoking Status")
plt.ylabel("Count");


# In[82]:


smoke_drink = df[ (df["smoke"]=="Yes") &  (df["alco"]=="Drink") ]
smoke_drink


# In[83]:


df.active.value_counts()


# In[84]:


df.active.value_counts().plot(kind ="pie" , autopct='%1.1f%%', colors=["#FFE5CF" , "#CDC2A5"], explode=(0,0), wedgeprops=dict(width=0.7));


# In[85]:


df.cardio.value_counts()


# In[86]:


df.cardio.value_counts().plot(kind ="pie" , autopct='%1.1f%%', colors=["#FFE5CF" , "#CDC2A5"], explode=(0,0), wedgeprops=dict(width=0.7));


# In[87]:


sns.scatterplot(y='Bmi', x='age_years', hue='cardio', data=df, palette='Set1')

plt.title('Age vs BMI Colored by Cardio Disease')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.legend(title='Cardio Disease')
plt.show()


# In[88]:


plt.figure(figsize=(6,5))
axis=sns.countplot(data=df, x='gender', hue='cardio', palette='Set1' , alpha=0.8)
axis.bar_label(axis.containers[0]);
axis.bar_label(axis.containers[1]);
plt.title('Heart Disease by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[89]:


plt.figure(figsize=(10,6))
sns.histplot(data=df, x='age_years', hue='cardio', multiple='stack', kde=False, palette='Set1')
plt.title('Distribution of Heart Disease by Age')
plt.xlabel('Age (years)')
plt.ylabel('Count')
plt.show()


# In[90]:


round( pd.crosstab(df['BMI_category'], df['cardio'], normalize='index') * 100 , 2)


# In[91]:


plt.figure(figsize=(10,6))
axis=sns.countplot(data=df, x='BMI_category', hue='cardio', palette=["#7CF5FF" ,"#FF8C9E"])
axis.bar_label(axis.containers[0])
axis.bar_label(axis.containers[1])
plt.title('Heart Disease by BMI Categories')
plt.xlabel('BMI Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[92]:


round( pd.crosstab(df['blood_pressure_category'], df['cardio'], normalize='index') * 100 , 2)


# In[93]:


plt.figure(figsize=(10,6))
axis=sns.countplot(data=df, x='blood_pressure_category', hue='cardio', palette=["#7CF5FF" ,"#FF8C9E"])
axis.bar_label(axis.containers[0])
axis.bar_label(axis.containers[1])
plt.title('Heart Disease by blood_pressure ')
plt.xlabel('blood_pressure')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[94]:


round( pd.crosstab(df['cholesterol'], df['cardio'], normalize='index') * 100 , 2)


# In[95]:


plt.figure(figsize=(8,5))
axis= sns.countplot(data=df, x='cholesterol', hue='cardio', palette=["#E7D4B5" ,"#FEECE2"])
axis.bar_label(axis.containers[0])
axis.bar_label(axis.containers[1])

plt.title('Heart Disease by Cholesterol Levels')
plt.xlabel('Cholesterol Levels')
plt.ylabel('Count')
plt.show()


# In[96]:


round( pd.crosstab(df['gluc'], df['cardio'], normalize='index') * 100 , 2)


# In[97]:


plt.figure(figsize=(8,5))
axis= sns.countplot(data=df, x='gluc', hue='cardio', palette=["#E7D4B5" ,"#FEECE2"])
axis.bar_label(axis.containers[0])
axis.bar_label(axis.containers[1])
plt.title('Heart Disease by gluc');


# In[98]:


df_grouped = df.groupby(["cardio", "cholesterol"])["gluc"].value_counts().reset_index(name='count')
df_grouped.set_index(['cardio' , "cholesterol","gluc"], inplace=True)

df_grouped


# In[99]:


round( pd.crosstab(df['smoke'], df['cardio'], normalize='index') * 100 , 2)


# In[100]:


plt.figure(figsize=(8,5))
axis= sns.countplot(data=df, x='smoke', hue='cardio', palette=["#E7D4B5" ,"#FEECE2"])
axis.bar_label(axis.containers[0]);
axis.bar_label(axis.containers[1]);

plt.title('Heart Disease by smoke')
plt.xlabel('smoke ')
plt.ylabel('Count')
plt.show()


# In[101]:


df_grouped = df.groupby(["cardio", "smoke"])["alco"].value_counts().reset_index(name='count')
df_grouped.set_index(['cardio', 'smoke',"alco"], inplace=True)

df_grouped


# In[102]:


round( pd.crosstab(df['active'], df['cardio'], normalize='index') * 100 , 2)


# In[103]:


plt.figure(figsize=(8,5))
axis= sns.countplot(data=df, x='active', hue='cardio', palette=["#FFDBB5", "#FFBF78"])
axis.bar_label(axis.containers[0]);
axis.bar_label(axis.containers[1]);

plt.title('Heart Disease by active')
plt.xlabel('smoke ')
plt.ylabel('Count')
plt.show()


# # 3. Machine Learning 
# 
# Classification techniques were used: Logistic Regression, Random Forest. 
# For evaluation metrics were used: used accuracy, precision, recall, F1 score, and ROC-AUC to assess model performance.
# Performed cross-validation to ensure the robustness of your results.
# Plotted ROC curves to visualize model performance.

# In[118]:


print(df.columns)


# In[124]:


# Prepare the data by selecting features and target variable
# Assuming 'cardio' is the target variable and other columns are features
X = df.drop('cardio', axis=1)  # Features
y = df['cardio']                # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[125]:


# Train the Models, both the Logistic Regression and Random Forest models.

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[126]:


# Evaluate the Models using various metrics

# Make predictions
log_predictions = log_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Evaluate Logistic Regression
print("Logistic Regression Evaluation:")
print("Accuracy:", accuracy_score(y_test, log_predictions))
print("Precision:", precision_score(y_test, log_predictions))
print("Recall:", recall_score(y_test, log_predictions))
print("F1 Score:", f1_score(y_test, log_predictions))

# Evaluate Random Forest
print("\nRandom Forest Evaluation:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print("Precision:", precision_score(y_test, rf_predictions))
print("Recall:", recall_score(y_test, rf_predictions))
print("F1 Score:", f1_score(y_test, rf_predictions))


# ### Observations
# The Random Forest model and the Logistic Regression model show reasonable performance. It might be beneficial to explore hyperparameter tuning for both models or to try additional models, such as Support Vector Machines or XGBoost, to see if further improvements can be made.

# In[127]:


# Ensure robust of the results

# Cross-Validation for Logistic Regression
log_cv_scores = cross_val_score(log_model, X, y, cv=5, scoring='accuracy')
print("\nLogistic Regression Cross-Validation Accuracy:", log_cv_scores.mean())

# Cross-Validation for Random Forest
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
print("Random Forest Cross-Validation Accuracy:", rf_cv_scores.mean())


# ### Observations
# Logistic Regression is 70% that is positive indicator of the model’s robustness. The Random Forest model had a lower average accuracy of 67.2% during cross-validation. It might be overfitting on the training data, as it performed better in the training set but did not generalize as effectively during cross-validation.

# In[128]:


# Plot the ROC curves for both models to visualize their performance

# ROC Curve for Logistic Regression
log_prob = log_model.predict_proba(X_test)[:, 1]
fpr_log, tpr_log, _ = roc_curve(y_test, log_prob)
roc_auc_log = auc(fpr_log, tpr_log)

# ROC Curve for Random Forest
rf_prob = rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Plotting the ROC Curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_log, tpr_log, color='blue', label='Logistic Regression (AUC = {:.2f})'.format(roc_auc_log))
plt.plot(fpr_rf, tpr_rf, color='green', label='Random Forest (AUC = {:.2f})'.format(roc_auc_rf))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


# ### Observations
# The model performs slightly better in predicting class 1 (cardiovascular disease) with higher precision but slightly lower recall. Overall, the accuracy is 73%, and the performance is balanced across both classes.
# 

# In[131]:


# feature importance from Random Forest
importances = rf_model.feature_importances_
feature_names = X_train.columns

# plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title('Feature Importance - Random Forest')
plt.show()


# The model shows a balanced performance.

# ### Results
# The Logistic Regression model achieved an accuracy of 73%, with an AUC score of 0.77. The Random Forest model performed similarly, with an accuracy of 72% and an AUC score of 0.78. Feature importance from the Random Forest model indicated that age, systolic blood pressure (ap_hi), and cholesterol levels were the most important predictors of cardiovascular disease.
# 
# 
# Model Performance:
# Logistic Regression achieved an accuracy of 73% with a precision of 0.70 for patients without cardiovascular disease and 0.77 for patients with it.
# 
# Random Forest achieved an accuracy of 72%, with slightly lower recall values but a more balanced performance overall.
# 
# AUC Scores:
# Logistic Regression: 0.778
# Random Forest: 0.998 (showing near-perfect classification performance).
# Feature Importance:
# Random Forest highlighted the most influential features for predicting cardiovascular disease:
# Systolic and diastolic blood pressure (ap_hi, ap_lo)
# Cholesterol levels
# Age
# BMI (derived feature)

# # 4. Final conclusion and recommendation
# In summary, this Case Study identified the risk factors for cardiovascular disease. It is showed that high blood pressure and high BMI are linked to a higher risk of heart disease. Smoking and lack of exercise also contribute to the risk.
# Based on the analysis, to prevent cardiovascular disease, following measures are suggested: eat a healthy diet low in saturated fats, salt, and sugar; maintain a healthy weight; stay physically active; quit smoking, alcohol, and drugs; manage stress to protect your heart; monitor and control blood pressure; have regular check-ups with a cardiologist; keep cholesterol levels in check; monitor blood sugar, especially after 40; and take blood thinners if prescribed by a doctor. 
# 

# In[ ]:




