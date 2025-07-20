import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("netflix_titles.csv")
print(df.head())
print(df.columns)
print("\nShape of dataset:", df.shape)
df.dropna(subset=['rating', 'duration'], inplace=True)
df['director'].fillna('Unknown', inplace=True)
df['cast'].fillna('Not Listed', inplace=True)
df['country'].fillna('Unknown', inplace=True)
df['date_added'].fillna('Unknown', inplace=True)
print("\nMissing values:\n", df.isnull().sum())
df.reset_index(drop=True, inplace=True)
df.info()

#Distribution of Content Types
type_counts = df['type'].value_counts()
plt.figure(figsize=(6, 4))
sns.barplot(x=type_counts.index, y=type_counts.values, palette="Set2")
plt.title("Distribution of Content Types")
plt.xlabel("Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#Number of Titles Released Each Year'
release_counts = df['release_year'].value_counts().sort_index()

plt.figure(figsize=(12,6))
sns.lineplot(x=release_counts.index, y=release_counts.values)
plt.title('Number of Titles Released Each Year')
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.show()

#Top 5 Genres Released Over Years
df['genres'] = df['listed_in'].str.split(',')
df_exploded = df.explode('genres')
df_exploded['genres'] = df_exploded['genres'].str.strip()
genre_year = df_exploded.groupby(['release_year', 'genres']).size().unstack(fill_value=0)

top_genres = genre_year.sum().sort_values(ascending=False).head(5).index

plt.figure(figsize=(14,7))
for genre in top_genres:
    sns.lineplot(data=genre_year[genre], label=genre)

plt.title('Top 5 Genres Released Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.legend()
plt.show()

#'Top 5 Countries Producing Netflix Content Over Years
df['country_list'] = df['country'].str.split(',')
df_country = df.explode('country_list')
df_country['country_list'] = df_country['country_list'].str.strip()
country_year = df_country.groupby(['release_year', 'country_list']).size().unstack(fill_value=0)
top_countries = country_year.sum().sort_values(ascending=False).head(5).index
plt.figure(figsize=(14,7))
for country in top_countries:
    sns.lineplot(data=country_year[country], label=country)

plt.title('Top 5 Countries Producing Netflix Content Over Years')
plt.xlabel('Year')
plt.ylabel('Number of Titles')
plt.legend()
plt.show()

