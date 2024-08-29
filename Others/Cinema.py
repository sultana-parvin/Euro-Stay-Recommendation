import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    # Use raw string for Windows path
    file_path = r"C:\Users\parvi\OneDrive\Desktop\My project\Streamlit\data\movies_final.csv"
    
    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return None
    
    try:
        movies_final1 = pd.read_csv(file_path, sep=',', low_memory=False)
        return movies_final1
    except Exception as e:
        st.error(f"Error loading the file: {str(e)}")
        return None

movies_final1 = load_data()

if movies_final1 is not None:
    movies_final = movies_final1[movies_final1['Note Pondere'] >= 8]
    # Continue with the rest of your code
else:
    st.error("Unable to load data. Please check the file path and try again.")
    st.stop()  # S
# Dummies for genres, actress, actor, and director
df_dummies = pd.concat([movies_final, movies_final['genres'].str.get_dummies(sep=',')], axis=1)
df_dummies = pd.concat([df_dummies, df_dummies['actress'].str.get_dummies(sep=', ')], axis=1)
df_dummies = pd.concat([df_dummies, df_dummies['actor'].str.get_dummies(sep=', ')], axis=1)
df_dummies = pd.concat([df_dummies, df_dummies['director'].str.get_dummies(sep=', ')], axis=1)

# Normalize numerical columns
numerical_cols = df_dummies.select_dtypes(include=['number']).columns
numerical_cols = numerical_cols[numerical_cols != 'tconst']
numerical_cols = numerical_cols[9:]
scaler = StandardScaler()
df_dummies[numerical_cols] = scaler.fit_transform(df_dummies[numerical_cols])

# Train KNN model
model = NearestNeighbors(n_neighbors=10, metric='cosine')
model.fit(df_dummies.drop(columns=['tconst', 'runtimeMinutes', 'primaryTitle', 'genres', 'averageRating', 'numVotes', 'Note Pondere', 'actress', 'actor', 'director']))

# Predict neighbors for the test set, EXCLUDING columns not used for training
distances, indices = model.kneighbors(df_dummies.drop(columns=['tconst', 'primaryTitle', 'runtimeMinutes', 'genres', 'averageRating', 'numVotes', 'Note Pondere', 'actress', 'actor', 'director']))

# You'll need to define how to evaluate the model's effectiveness without 'true_neighbor'

for i in range(5):  # Look at recommendations for the first 5 movies
    movie_title = df_dummies.iloc[i]['primaryTitle']  # Assuming 'primaryTitle' exists
    recommended_indices = indices[i]
    recommended_movies = df_dummies.iloc[recommended_indices]['primaryTitle']  # Assuming 'primaryTitle' exists
    print(f"Recommendations for {movie_title}: {recommended_movies.values}")

#  get input interactif recommendation system input = actor or actress or productor or genres or title

def get_recommendations(movie_title, num_recommendations=5):
  """
  Provides movie recommendations based on a given title.

  Args:
    movie_title: The title of the movie to get recommendations for.
    num_recommendations: The number of recommendations to return.

  Returns:
    A list of recommended movie titles.
  """

  if movie_title not in df_dummies['primaryTitle'].values:
    print(f"Movie '{movie_title}' not found in the dataset.")
    return []

  # Get the index of the movie
  movie_index = df_dummies[df_dummies['primaryTitle'] == movie_title].index[0]

  # Get the nearest neighbors for the movie
  distances, indices = model.kneighbors(df_dummies.drop(columns=['tconst', 'primaryTitle', 'runtimeMinutes', 'genres', 'averageRating', 'numVotes', 'Note Pondere', 'actress', 'actor', 'director']).iloc[movie_index].values.reshape(1, -1), n_neighbors=num_recommendations + 1)

  # Exclude the movie itself from the recommendations
  recommended_indices = indices[0][1:]

  # Return the titles of the recommended movies
  return df_dummies.iloc[recommended_indices]['primaryTitle'].tolist()


# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #f0f0f0;
}
.stTitle {
    color: #1e3d59;
    font-size: 50px;
    font-weight: bold;
    margin-bottom: 30px;
}
.stSelectbox, .stTextInput {
    background-color: #ffffff;
    color: #1e3d59;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 20px;
}
.stButton > button {
    background-color: #ff6e40;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    padding: 10px 20px;
}
.movie-recommendation {
    background-color: #ffc13b;
    color: #1e3d59;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Planete PopCorn")

input_type = st.selectbox("Select the type of recommendation:", 
                          ["actor", "actress", "director", "genres", "primaryTitle"])

input_value = st.text_input(f"Enter the {input_type} you want recommendations for:")

if st.button("Get Recommendations"):
    if input_type == 'genres':
        matching_movies = df_dummies[df_dummies['genres'].str.contains(input_value, case=False)]['primaryTitle'].tolist()
    else:
        matching_movies = df_dummies[df_dummies[input_type].str.contains(input_value, case=False)]['primaryTitle'].tolist()

    if matching_movies:
        recommendations = get_recommendations(matching_movies[0])
        st.subheader(f"Recommendations based on {input_type} '{input_value}':")
        for movie in recommendations:
            st.markdown(f'<div class="movie-recommendation">{movie}</div>', unsafe_allow_html=True)
    else:
        st.error(f"No movies found matching {input_type} '{input_value}'.")