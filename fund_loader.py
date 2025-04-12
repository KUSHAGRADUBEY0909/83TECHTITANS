import pandas as pd

def load_and_clean_data():
    print("🔄 Loading data...")
    df = pd.read_json("all_funds_combined.json", lines=True)  # Ensure this file exists

    # Fill missing values and normalize text
    df.fillna("", inplace=True)
    df = df.applymap(lambda x: str(x).strip().lower() if isinstance(x, str) else x)

    print("✅ Data loaded and preprocessed")
    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    df.to_pickle("clean_funds.pkl")  # Save cleaned data
    print("📂 Cleaned data saved as clean_funds.pkl!")