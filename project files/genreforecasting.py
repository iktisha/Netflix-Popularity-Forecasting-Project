import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('netflix_titles.csv')
df['main_genre'] = df['listed_in'].apply(lambda x: x.split(',')[0].strip())
df = df[df['release_year'] > 2000]

top_genres = df['main_genre'].value_counts().head(5).index.tolist()

forecast_results = []

for genre in top_genres:
    genre_df = df[df['main_genre'] == genre]
    yearly_counts = genre_df.groupby('release_year').size().reset_index(name='count')

    X = yearly_counts[['release_year']]
    y = yearly_counts['count']

    model = LinearRegression()
    model.fit(X, y)

    future_years = pd.DataFrame({'release_year': np.arange(2021, 2026)})
    future_years['count'] = model.predict(future_years)

    for _, row in future_years.iterrows():
        forecast_results.append({
            'release_year': int(row['release_year']),
            'main_genre': genre,
            'predicted_count': max(0, int(row['count']))
        })

forecast_df = pd.DataFrame(forecast_results)

historical_df = df.groupby(['release_year', 'main_genre']).size().reset_index(name='predicted_count')
historical_df = historical_df[historical_df['main_genre'].isin(top_genres)]

#for bi
combined_df = pd.concat([historical_df, forecast_df], ignore_index=True)
combined_df = combined_df.sort_values(['main_genre', 'release_year'])

combined_df.to_csv('linear_genre_forecasts.csv', index=False)
print("Forecast CSV saved as linear_genre_forecasts.csv")


#python visualization
top_genres = combined_df['main_genre'].unique()

plt.figure(figsize=(12, 8))

for genre in top_genres:

    hist = historical_df[historical_df['main_genre'] == genre]
    plt.plot(hist['release_year'], hist['predicted_count'], marker='o', label=f'{genre} Actual')

    fore = forecast_df[forecast_df['main_genre'] == genre]
    plt.plot(fore['release_year'], fore['predicted_count'], linestyle='--', marker='x', label=f'{genre} Forecast')

plt.title('Netflix Genre Popularity: Historical vs Forecast (2021-2025)')
plt.xlabel('Year')
plt.ylabel('Number of Shows')
plt.legend()
plt.grid(True)
plt.show()