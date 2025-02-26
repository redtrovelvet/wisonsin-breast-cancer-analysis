import pandas as pd
import numpy as np

splits = {'train': 'train.csv', 'test': 'test.csv'}
df = pd.read_csv("hf://datasets/wwydmanski/wisconsin-breast-cancer/" + splits["train"])