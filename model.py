import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
data = pd.read_csv("social_ads.csv")

print("=== Head ===")
print(data.head())
print("\n=== Info ===")
print(data.info())
print("\n=== Describe ===")
print(data.describe())
print("\n=== Null Values ===")
print(data.isnull().sum())

# ─────────────────────────────────────────────
# 2. EDA (Exploratory Data Analysis)
# ─────────────────────────────────────────────
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.countplot(x="Purchased", data=data)
plt.title("Purchase Distribution")

plt.subplot(1, 3, 2)
sns.boxplot(y=data['Age'])
plt.title("Age Outliers")

plt.subplot(1, 3, 3)
sns.boxplot(y=data['EstimatedSalary'])
plt.title("Salary Outliers")

plt.tight_layout()
plt.show()

sns.scatterplot(x="Age", y="EstimatedSalary", hue="Purchased", data=data)
plt.title("Age vs Salary by Purchase")
plt.show()

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
le = LabelEncoder()
data["Gender"] = le.fit_transform(data["Gender"])

# Print encoding so app.py stays in sync
print("\n=== Gender Encoding ===")
for cls, enc in zip(le.classes_, le.transform(le.classes_)):
    print(f"  {cls} -> {enc}")

X = data[['Gender', 'Age', 'EstimatedSalary']]
y = data['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# 4. TRAIN MODEL
# ─────────────────────────────────────────────
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────
y_pred = model.predict(X_test)

print("\n=== Accuracy ===")
print(f"  Test Accuracy : {accuracy_score(y_test, y_pred):.4f}")

# Cross-validation for a more reliable estimate
cv_scores = cross_val_score(model, np.vstack([X_train, X_test]),
                             np.concatenate([y_train, y_test]),
                             cv=5, scoring='accuracy')
print(f"  CV Mean       : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────────
# 6. SAVE MODEL
# ─────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("\nModel saved to model.pkl")
