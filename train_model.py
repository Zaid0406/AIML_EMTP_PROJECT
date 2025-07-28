from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

# Load dataset
df = pd.read_csv("your_dataset.csv")

# Features and target
X = df[["feature1", "feature2", "feature3"]]
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "best_model.pkl")
print("✅ Model saved as best_model.pkl")
