import pandas as pd
import os
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path: str, output_path: str):

    # 1. Load Dataset
    df = pd.read_csv(input_path)

    # 2. Handling Missing Values
    df['ph'] = df['ph'].fillna(df['ph'].mean())
    df['Sulfate'] = df['Sulfate'].fillna(df['Sulfate'].mean())
    df['Trihalomethanes'] = df['Trihalomethanes'].fillna(
        df['Trihalomethanes'].mean()
    )

    # 3. Feature Scaling
    features_to_scale = df.drop('Potability', axis=1).columns

    scaler = StandardScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    # 4. Outlier Handling (Clipping)
    for col in features_to_scale:
        upper_limit = df[col].mean() + 3 * df[col].std()
        lower_limit = df[col].mean() - 3 * df[col].std()
        df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)


    # 5. Save Preprocessed Dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    preprocess_data(
        input_path="Eksperimen_SML_Ahmad-Alfim-Majid-Roifi\water_potability_raw.csv",
        output_path="Eksperimen_SML_Ahmad-Alfim-Majid-Roifi\preprocessing\water_potability_preprocessing"
    )
