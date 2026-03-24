
#source code –
# ------------------------------------------------------------
# ROAD ACCIDENT ANALYSIS & ACCIDENT SEVERITY PREDICTION PROJECT
# FULL COMBINED CODE (READY FOR GOOGLE COLAB)
# ------------------------------------------------------------
# STEP 0: IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# STEP 1: LOAD YOUR DATASET
df = pd.read_csv("traffic_accident_dataset.csv")
print("Dataset Loaded Successfully!")
print(df.head())

# STEP 2: BASIC INFORMATION
print("\nShape of Dataset:", df.shape)
print("\nColumn Names:\n", df.columns)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isna().sum())

# STEP 3: HANDLE MISSING VALUES
for col in df.columns:
    if df[col].dtype == "object":   # text type
        df[col] = df[col].fillna("Unknown")
    else:                           # number type
        df[col] = df[col].fillna(0)

print("\nMissing values after cleaning:\n", df.isna().sum())

# STEP 4: DROP UNNEEDED COLUMNS
if "crash_date" in df.columns:
    df = df.drop(columns=["crash_date"])

# STEP 5: VISUALIZATIONS (EDA)
plt.figure(figsize=(10,4))
sns.countplot(x="most_severe_injury", data=df)
plt.title("Most Severe Injury Count")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,4))
sns.countplot(x="weather_condition", data=df)
plt.title("Accidents vs Weather Condition")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,4))
sns.histplot(df["crash_hour"], bins=24)
plt.title("Accidents by Hour of Day")
plt.xlabel("Hour")
plt.show()

# STEP 6: LABEL ENCODING FOR CATEGORICAL (TEXT) DATA
from sklearn.preprocessing import LabelEncoder

df_ml = df.copy()
cat_columns = df_ml.select_dtypes(include=["object"]).columns

label_encoders = {}

for col in cat_columns:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])
    label_encoders[col] = le

print("\nAfter Label Encoding:")
print(df_ml.head())

# STEP 7: SELECT TARGET AND FEATURES
target_column = "most_severe_injury"

X = df_ml.drop(columns=[target_column])
y = df_ml[target_column]

print("\nX shape:", X.shape)
print("Y shape:", y.shape)

# STEP 8: TRAIN-TEST SPLIT
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# STEP 9: TRAIN RANDOM FOREST MODEL
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
print("\nModel Training Complete!")

# STEP 10: MODEL PREDICTIONS & PERFORMANCE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# STEP 11: SAVE MODEL
import joblib
joblib.dump(model, "road_accident_severity_model.pkl")
print("\nModel saved as road_accident_severity_model.pkl")
