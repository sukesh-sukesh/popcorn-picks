import difflib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


dataset_path = 'dataset/movies.csv'
movies_data = pd.read_csv(dataset_path, on_bad_lines='skip', engine='python')

selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

combined_features = movies_data['genres'] + ' ' + movies_data['keywords'] + ' ' + movies_data[
    'tagline'] + ' ' + movies_data['cast'] + ' ' + movies_data['director']

vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
similarity = cosine_similarity(feature_vectors)

@app.route('/')
def index():
    return render_template('front-end.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_input = request.json['input']
    recommendations = get_movie_recommendations(user_input)
    return jsonify(recommendations)

def get_movie_recommendations(movie_name):
    list_of_all_titles = movies_data['title'].tolist()
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    
    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]
        
   
        movie_details = movies_data[movies_data.index == index_of_the_movie]
        genres = movie_details['genres'].values[0]
        release_date = movie_details['release_date'].values[0][:4]  # Extracting the year only
        director = movie_details['director'].values[0]
        cast = movie_details['cast'].values[0]
        vote_average = movie_details['vote_average'].values[0]

       
        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
        
        recommended_movies = []
        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movies_data[movies_data.index == index]['title'].values[0]
            rating_from_index = movies_data[movies_data.index == index]['vote_average'].values[0]
            if title_from_index != close_match:  # Exclude the original movie
                recommended_movies.append({'title': title_from_index, 'rating': rating_from_index})
                if len(recommended_movies) >= 10:  # Limit to top 10 recommendations
                    break
        
        return {
            'movie': close_match,
            'genres': genres,
            'release_date': release_date,
            'director': director,
            'cast': cast,
            'rating': vote_average,
            'recommendations': recommended_movies
        }
    else:
        return {"error": "No close matches found for your input. Please try again."}

if __name__ == '__main__':
    app.run(debug=True)
