# Import necessary libraries
import streamlit as st
import pandas as pd
from scipy.sparse import vstack
from sklearn.metrics.pairwise import linear_kernel
from underthesea import word_tokenize
import re
import joblib
from PIL import Image

# Load the product data
products_df = pd.read_csv('sampled_products.csv')

# Load saved TF-IDF vectorizer and TF-IDF matrix
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')

# Load saved cosine similarities
cosine_similarities = joblib.load('cosine_similarities.pkl')

# Function to clean content
def simple_clean_content(text):
    # Remove special characters
    text = re.sub('[\-\–\,\+\?\%\/\•\*\&\[\]\(\)\:\;]', ' ', text).replace('v.',' ').replace('...',' ').replace('.',' ').replace('…',' ')
    # Lowercase and split sentences separated by '\n'
    text = text.lower().split('\n')
    # Remove leading and trailing spaces in each sentence
    text = [e.strip() for e in text]
    return text

# Import Vietnamese stopwords
STOP_WORD_FILE = 'vietnamese-stopwords.txt'
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read()
stop_words = stop_words.split('\n')

# Function to get product recommendations
def cosine_recommendation(df, cosine_similarities, product_id, num_recommendations=5, selected_brands=None):
    select_product_idx = df[df['item_id'] == product_id].index[0]
    similar_indices = cosine_similarities[select_product_idx].argsort()[-(num_recommendations + 1):-1][::-1]

    # Filter out the selected product
    similar_indices = [idx for idx in similar_indices if idx != select_product_idx]

    recommendations = df.iloc[similar_indices, :][['item_id', 'name', 'rating', 'list_price', 'brand', 'image']]

    # Apply brand filter if selected
    if selected_brands:
        recommendations = recommendations[recommendations['brand'].isin(selected_brands)]

    return recommendations

# Function to get product recommendations based on content
def search_cosine_recommendation(text, df, tfidf_vectorizer, tfidf_matrix, cosine_similarities, num_recommendations=5, selected_brands=None):
    # Clean and preprocess the search text
    cleaned_text = simple_clean_content(text)

    # Create a new DataFrame with the cleaned search text
    search_df = pd.DataFrame(cleaned_text, columns=['content'])

    # Tokenize and preprocess the cleaned search text
    search_df['content_wt'] = search_df['content'].apply(lambda x: word_tokenize(x, format='text'))

    # Calculate the TF-IDF matrix for the search text
    tfidf_new = tfidf_vectorizer.transform(search_df['content_wt'])

    # Concatenate the TF-IDF matrices of existing products and the search text
    tfidf_concat = vstack((tfidf_matrix, tfidf_new))

    # Calculate cosine similarities with existing products and the search text
    cosine_similarities_search = linear_kernel(tfidf_concat, tfidf_concat)

    # Get the indices of the most similar products
    most_similar_indices = cosine_similarities_search[-1].argsort()[-(num_recommendations + 1):-1][::-1]

    # Get the most similar products
    recommendations = df.iloc[most_similar_indices][['item_id', 'name', 'rating', 'list_price', 'brand', 'image']]

    # Apply brand filter if selected
    if selected_brands:
        recommendations = recommendations[recommendations['brand'].isin(selected_brands)]

    return recommendations

# Danh sách các thương hiệu bạn muốn lọc
brands_to_filter = ['OEM', 'Samsung', 'Logitech', 'Sony', 'LG', 'Panasonic', 'Yoosee', 'SanDisk', 'UGREEN', 'TP-Link', 'Apple']

# Streamlit app
st.sidebar.title('Menu')
page = st.sidebar.selectbox('Select a Page', ['Home', 'Product Name', 'Product ID'])

if page == 'Home':
    st.title('Welcome to the Product Recommendation App')
    st.write('This app allows you to search for products and get recommendations based on product descriptions.')

# Trang Product Name
if page == 'Product Name':
    st.markdown(
        """
        <h1 style='color: blue; font-size: 20px;'>Recommendation by Product Name or Description</h1>
        """,
        unsafe_allow_html=True,
    )
    query = st.text_input('Enter a Product Name or Description:')

    # Add an option to choose the number of recommendations to display
    num_recommendations = st.number_input('Number of Recommendations', min_value=1, max_value=50, value=10)

    # Add filtering options
    min_rating = st.select_slider('Rating', options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], format_func=lambda x: f'{x} ⭐')
    sort_by = st.selectbox('Sort By', ['Rating', 'Price', 'Brand'])

    # Add brand filtering options
    selected_brands = st.multiselect('Select Brands to Filter', brands_to_filter)

    if st.button('Get Recommendations'):
        results = search_cosine_recommendation(query, products_df, tfidf_vectorizer, tfidf_matrix, cosine_similarities, num_recommendations=num_recommendations)

        # Filter by minimum rating
        results = results[results['rating'] >= min_rating]

        # Filter by selected brands
        if selected_brands:
            results = results[results['brand'].isin(selected_brands)]

        # Sort by selected criteria
        if sort_by == 'Rating':
            results = results.sort_values(by=['rating'], ascending=False)
        elif sort_by == 'Price':
            results = results.sort_values(by=['list_price'])
        elif sort_by == 'Brand':
            results = results.sort_values(by=['brand'])

        # Display recommendations in a grid format
        col1, col2, col3 = st.columns(3)
        products = results.iterrows()
        try:
            for i in range(num_recommendations):
                index, row = next(products)
                if i % 3 == 0:
                    column = col1
                elif i % 3 == 1:
                    column = col2
                else:
                    column = col3

                with column:
                    st.image(row['image'], caption=row['name'], use_column_width=True)
                    st.write(f"**Brand:** {row['brand']}")
                    st.write(f"**Rating:** {row['rating']} ⭐")
                    st.write(f"**Price:** {row['list_price']:.2f} VND")
        except StopIteration:
            pass

        if results.empty:
            st.info('No matching products found.')

# Trang Product ID
if page == 'Product ID':
    st.markdown(
        """
        <h1 style='color: blue; font-size: 20px;'>Recommendation by Product ID</h1>
        """,
        unsafe_allow_html=True,
    )
    selected_product_id = st.selectbox('Select a Product:', products_df['item_id'])

    # Add an option to choose the number of recommendations to display
    num_recommendations = st.number_input('Number of Recommendations', min_value=1, max_value=50, value=10)

    # Add filtering options
    min_rating = st.select_slider('Rating', options=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], format_func=lambda x: f'{x} ⭐')
    sort_by = st.selectbox('Sort By', ['Rating', 'Price', 'Brand'])

    # Add brand filtering options
    selected_brands = st.multiselect('Select Brands to Filter', brands_to_filter)

    if st.button('Get Recommendations'):
        recommendations = cosine_recommendation(products_df, cosine_similarities, selected_product_id, num_recommendations=num_recommendations)

        # Filter by minimum rating
        recommendations = recommendations[recommendations['rating'] >= min_rating]

        # Filter by selected brands
        if selected_brands:
            recommendations = recommendations[recommendations['brand'].isin(selected_brands)]

        # Sort by selected criteria
        if sort_by == 'Rating':
            recommendations = recommendations.sort_values(by=['rating'], ascending=False)
        elif sort_by == 'Price':
            recommendations = recommendations.sort_values(by=['list_price'])
        elif sort_by == 'Brand':
            recommendations = recommendations.sort_values(by=['brand'])
        # Display recommendations in a grid format
        col1, col2, col3 = st.columns(3)
        products = recommendations.iterrows()
        try:
            for i in range(num_recommendations):
                index, row = next(products)
                if i % 3 == 0:
                    column = col1
                elif i % 3 == 1:
                    column = col2
                else:
                    column = col3

                with column:
                    st.image(row['image'], caption=row['name'], use_column_width=True)
                    st.write(f"**Brand:** {row['brand']}")
                    st.write(f"**Rating:** {row['rating']} ⭐")
                    st.write(f"**Price:** {row['list_price']:.2f} VND")
        except StopIteration:
            pass

        if recommendations.empty:
            st.info('No matching products found.')

# About section
st.sidebar.title('About')
st.sidebar.info('This is a product recommendation app using a machine learning model.')