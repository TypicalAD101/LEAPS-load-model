import pandas as pd
import datetime
import random

date_range = pd.date_range(start='2024-01-01', periods= 365 * 24 * 4, freq='15T')

value = [random.randint(240, 1000) for step in date_range]

df = pd.DataFrame({
    'Date': date_range,
    'crit_load': value}
)

df.to_csv("Load_Profile.csv", index=False)
