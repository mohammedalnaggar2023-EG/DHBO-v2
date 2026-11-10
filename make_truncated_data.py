"""Build the earlier-period dataset: JHU file truncated at 31 December 2021."""
import os, pandas as pd
src = os.environ.get('DHBO_DATA', 'data/confirmed_global.csv')
df = pd.read_csv(src); meta = list(df.columns[:4])
keep = [c for c in df.columns[4:] if pd.to_datetime(c, format='%m/%d/%y') <= pd.Timestamp('2021-12-31')]
os.makedirs('data', exist_ok=True)
df[meta + keep].to_csv('data/confirmed_to_2021-12-31.csv', index=False)
print('kept', len(keep), 'days, last', keep[-1])
