import pandas as pd

def extract_parameters_from_csv(file_path: str):
    df = pd.read_csv(file_path)

    required_columns = ['Ref Temp', 'a', 'b', 'c', 'd']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("CSV file must contain columns: 'Ref Temp', 'a', 'b', 'c', 'd'")
    
    parameters = df[required_columns].values.tolist()
    return parameters

def extract_dataframe_from_csv(file_path: str):
    df = pd.read_csv(file_path)
    return df