# Data_prep.py
from sklearn.datasets import load_breast_cancer
import pandas as pd

def fetch_data():
    # Load the breast cancer dataset from sklearn
    data = load_breast_cancer(as_frame=True)
    
    # Convert the data to a pandas dataframe
    df = data.frame
    
    # Optionally save the dataset to a CSV or Parquet file
    df.to_csv('breast_cancer_data.csv', index=False)
    
    # Print a message indicating that the data has been fetched and saved
    print("Data fetched and saved to breast_cancer_data.csv")
    return df
