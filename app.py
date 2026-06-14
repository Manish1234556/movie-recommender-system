
import streamlit as st
import pandas as pd
import pickle
import requests

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp{
    background: linear-gradient(to right, #141E30, #243B55);
}

.main-title{
    text-align:center;
    color:white;
    font-size:60px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:#dcdcdc;
    font-size:20px;
    margin-bottom:30px;
}

.movie-name{
    color:white;
    text-align:center;
    font-size:16px;
    font-weight:bold;
    margin-top:10px;
}

.stButton > button{
    width:100%;
    background:linear-gradient(90deg,#ff512f,#dd2476);
    color:white;
    border:none;
    border-radius:10px;
    height:50px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TMDB API KEY
# --------------------------------------------------


api_key = st.secrets["TMDB_API_KEY"]


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

movies_dict = pickle.load(open("movie_dict.pkl", "rb"))
movies = pd.DataFrame(movies_dict)

# --------------------------------------------------
# CREATE SIMILARITY MATRIX
# --------------------------------------------------

@st.cache_resource
def create_similarity():

    cv = CountVectorizer(
        max_features=5000,
        stop_words='english'
    )

    vectors = cv.fit_transform(
        movies['tags']
    ).toarray()

    similarity = cosine_similarity(vectors)

    return similarity

similarity = create_similarity()

# --------------------------------------------------
# FETCH POSTER
# --------------------------------------------------

@st.cache_data
def fetch_poster(movie_id):

    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}&language=en-US"

    try:
        response = requests.get(url, timeout=10)

        print("Status:", response.status_code)

        data = response.json()

        print(data)

        poster_path = data.get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

        return None

    except Exception as e:
        print("ERROR:", e)
        return None

# --------------------------------------------------
# RECOMMEND FUNCTION
# --------------------------------------------------

def recommend(movie):

    movie_index = movies[
        movies['title'] == movie
    ].index[0]

    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_posters = []

    for i in movies_list:

        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(
            movies.iloc[i[0]].title
        )

        recommended_posters.append(
            fetch_poster(movie_id)
        )

    return recommended_movies, recommended_posters

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    "<h1 class='main-title'>🎬 Movie Recommender</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p class='subtitle'>Find your next favorite movie instantly</p>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# MOVIE SELECTOR
# --------------------------------------------------

selected_movie = st.selectbox(
    "Choose a movie",
    movies['title'].values
)

# --------------------------------------------------
# RECOMMEND BUTTON
# --------------------------------------------------

if st.button("Show Recommendations"):

    names, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    cols = [col1, col2, col3, col4, col5]

    for idx, col in enumerate(cols):

        with col:

            if posters[idx]:
                st.image(posters[idx])

            st.markdown(
                f"""
                <div class='movie-name'>
                {names[idx]}
                </div>
                """,
                unsafe_allow_html=True
            )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("<br><hr>", unsafe_allow_html=True)

st.markdown(
    """
    <center>
    <p style='color:white'>
    Made with ❤️ using Streamlit & TMDB API
    </p>
    </center>
    """,
    unsafe_allow_html=True
)