import pandas as pd
from sklearn.model_selection import train_test_split  # lightweight model
from sklearn.naive_bayes import MultinomialNB
import pickle

print("Loading dataset...")

# LOAD CSV
df = pd.read_csv("Disease and symptoms dataset.csv")
print("Dataset loaded successfully!")

# CLEAN DATA
df = df.fillna(0)
df.columns = [col.strip().lower() for col in df.columns]

print("Cleaning done!")

# target + features
y = df["diseases"]
X = df.drop("diseases", axis=1)

# SPLIT
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# TRAIN FAST MODEL
print("Training lightweight AI model...")

model = MultinomialNB()
model.fit(X_train, y_train)

print("Training completed!")

# ACCURACY
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")

# SAVE MODEL
pickle.dump(model, open("disease_model.pkl", "wb"))
pickle.dump(X.columns.tolist(), open("symptom_list.pkl", "wb"))

print("\nModel saved successfully!")