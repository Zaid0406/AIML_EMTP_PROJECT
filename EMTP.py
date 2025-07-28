#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 🚀 Electric Motor Temperature Prediction
# In this project, we predict the rotor temperature of an electric motor using regression techniques.


# In[2]:


# Data handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Warning filter
import warnings
warnings.filterwarnings('ignore')

# 🤖 Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# 💾 Save/Load Model
import pickle  # Works with pickle-mixin too

# 🌐 Flask Web Framework
from flask import Flask, request, render_template


# In[3]:


# For Windows users (replace "Zaid" with your actual username if needed)
df = pd.read_csv(r'C:\Users\Zaid\Desktop\EMTP.csv')

# Preview the first 5 rows
df.head()


# In[4]:


# Countplot of a categorical feature (if any, like session_id or labels)
sns.countplot(x='profile_id', data=df)
plt.xticks(rotation=90)
plt.title('Frequency of Measurements per Session ID')
plt.show()


# In[5]:


# Box plots for all numerical columns

plt.figure(figsize=(15, 10))
df.boxplot(rot=90)
plt.title('Box Plot of All Numerical Features')
plt.xticks(rotation=45)
plt.show()


# In[6]:


# Distribution plots for all numerical features

for col in df.select_dtypes(include='number'):
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()


# In[7]:


# Scatter plot between stator_yoke and stator_winding temperatures

sns.scatterplot(x='stator_yoke', y='stator_winding', data=df)
plt.title('Scatter Plot: Stator Yoke vs Stator Winding Temperature')
plt.xlabel('Stator Yoke Temperature')
plt.ylabel('Stator Winding Temperature')
plt.show()


# In[8]:


# Calculate correlation matrix
corr = df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


# In[9]:


# Display structure, data types, and null values
df.info()


# In[10]:


# Summary of numerical features
df.describe()


# In[11]:


df.head()


# In[12]:


# Drop unwanted features
columns_to_drop = ['torque', 'stator_yoke', 'stator_tooth', 'stator_winding', 'profile_id']
df = df.drop(columns=columns_to_drop, axis=1)

# Confirm the result
df.head()


# In[13]:


# Check for missing/null values
null_values = df.isnull().sum()

# Display columns with null values (if any)
null_values[null_values > 0]


# In[14]:


from sklearn.preprocessing import MinMaxScaler

# Separate features (X) and target (y)
X = df.drop(columns=['pm'])   # pm is the target variable
y = df['pm']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X)

# Convert to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Display sample
print(X_scaled_df.head())

# ✅ Save the scaler for later use (e.g., when deploying the model)
with open('minmax_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# In[15]:


# Splitting the normalized data (X_scaled_df) and target (y) into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_df,        # input features
    y,                  # target variable (pm)
    test_size=0.2,      # 20% test, 80% train
    random_state=42     # ensures reproducibility
)

# Displaying the shape of splits
print("Training Set Shape:", X_train.shape)
print("Test Set Shape:", X_test.shape)


# In[16]:


# Linear Regression function
def linear_regression_model(X_train, X_test, y_train, y_test):
    # Initialize model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("🔹 Linear Regression Results:")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return model, y_pred

# Call the function
lr_model, lr_predictions = linear_regression_model(X_train, X_test, y_train, y_test)


# In[17]:


# Decision Tree function
def decision_tree_model(X_train, X_test, y_train, y_test):
    # Initialize model
    model = DecisionTreeRegressor(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("🔹 Decision Tree Regression Results:")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    return model, y_pred

# Call the function
dt_model, dt_predictions = decision_tree_model(X_train, X_test, y_train, y_test)


# In[18]:


# Optimized Random Forest function
def random_forest_model(X_train, X_test, y_train, y_test):
    # Use a very lightweight model (faster but slightly less accurate)
    model = RandomForestRegressor(
        n_estimators=10,        # fewer trees = faster
        max_depth=10,          # restrict depth = faster
        min_samples_split=10,  # less branching
        random_state=42,
        n_jobs=-1              # use all CPU cores
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("🔹 Random Forest Regression Results:")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return model, y_pred

# Run the function
rf_model, rf_predictions = random_forest_model(X_train, X_test, y_train, y_test)


# In[19]:


# SVR function
def svr_model(X_train, X_test, y_train, y_test):
    # Use only a small subset
    X_train_small = X_train[:1000]
    y_train_small = y_train[:1000]

    # Fast SVR model
    model = SVR(kernel='linear', C=0.5, epsilon=0.2)

    # Train and predict
    model.fit(X_train_small, y_train_small)
    y_pred = model.predict(X_test)

    # Evaluation
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("🔹 SVR Results:")
    print(f"Root Mean Square Error (RMSE): {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")    
    return model, y_pred

# Call the function
svr_model_obj, svr_predictions = svr_model(X_train, X_test, y_train, y_test)


# In[20]:


import time

# Function to evaluate a single model
def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    print(f"Evaluating {name}...")
    start_time = time.time()
    
     # If model is SVR, use a smaller subset for faster training
    if isinstance(model, SVR):
        X_train = X_train[:1000]
        y_train = y_train[:1000]

    # Train the model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    duration = time.time() - start_time

    print(f"{name} ➤ R² Score: {r2:.4f} | RMSE: {rmse:.4f} | Time: {duration:.2f} sec\n")
    return r2, rmse

# Initialize and configure models
models = {
    "🔹 Linear Regression": LinearRegression(),
    "🔹 Decision Tree": DecisionTreeRegressor(random_state=42),
    "🔹 Random Forest": RandomForestRegressor(
        n_estimators=10,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    ),
    "🔹 SVR": SVR(kernel='linear', C=0.5, epsilon=0.2)
}

# Evaluate all models and store results
results = {}
for name, model in models.items():
    r2, rmse = evaluate_model(model, X_train, X_test, y_train, y_test, name)
    results[name] = {'r2': r2, 'rmse': rmse}

# Final comparison summary
print("📊 Final Model Comparison:")
for name, metrics in results.items():
    print(f"{name}: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.4f}")



# In[21]:


import joblib

# Save the best model
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model_instance = models[best_model_name]

# Retrain best model on full data (SVR with subset)
if isinstance(best_model_instance, SVR):
    best_model_instance.fit(X_train[:1000], y_train[:1000])
else:
    best_model_instance.fit(X_train, y_train)

# Save to file
filename = f'best_model_{best_model_name.replace(" ", "_").lower()}.pkl'
joblib.dump(best_model_instance, filename)
print(f"\n✅ Best model '{best_model_name}' saved as '{filename}'")


# In[24]:


# Save the model
joblib.dump(rf_model, 'random_forest_model.pkl')

print("✅ Model saved as 'random_forest_model.pkl'")


# In[25]:


import joblib

# Assuming your trained model is named 'model'
joblib.dump(model, 'best_model.pkl')

print("✅ Model saved successfully.")


# In[ ]:




