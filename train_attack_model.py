import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("D:\SentinelNet\data\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")

# Target column (change if needed)
y = df["Label"]   # e.g., BENIGN, DoS, Probe, etc.
X = df.drop("Label", axis=1)

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save encoder
joblib.dump(le, "models/label_encoder.pkl")

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight="balanced")
model.fit(X, y_encoded)

# Save model
joblib.dump(model, "models/attack_classifier.pkl")

print("✅ Attack classifier trained")