import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV we just created
df = pd.read_csv("architectural_styles.csv")

# Stratified split to maintain class balance
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["architectural_style"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["architectural_style"], random_state=42)

# Save splits
train_df.to_csv("train_data.csv", index=False)
val_df.to_csv("val_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print(f"Dataset split successfully!\nTrain: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
