import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    df['ph'] = df['ph'].fillna(df['ph'].mean())
    df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean())
    df['Trihalomethanes'] = df['Trihalomethanes'].fillna(
        df['Trihalomethanes'].mean()
    )

    features_to_scale = df.drop('Potability', axis=1).columns

    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    for col in features_to_scale:
        upper = df[col].mean() + 3 * df[col].std()
        lower = df[col].mean() - 3 * df[col].std()
        df[col] = df[col].clip(lower, upper)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_csv = os.path.join(BASE_DIR, "water_potability_raw.csv")
    output_csv = os.path.join(
        BASE_DIR,
        "preprocessing",
        "water_potability_preprocessing.csv"
    )

    preprocess_data(input_csv, output_csv)
