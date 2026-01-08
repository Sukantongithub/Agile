
import pandas as pd
import os

csv_path = r"d:\Agile\ocr_project\document_classification_training_data.csv"
if not os.path.exists(csv_path):
    exit(1)

df = pd.read_csv(csv_path)
print(df['label'].value_counts())
