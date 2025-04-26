import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# --- Streamlit Page Config ---
st.set_page_config(page_title="Ecommerce Product Recommender", layout="wide")

# --- App Title ---
st.title('🛒 E-commerce Product Recommendation System')
st.markdown('''
Welcome to your personalized product recommendation engine!  
Select a method from the sidebar to begin! 🚀
''')

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/263/263115.png", width=120)
st.sidebar.title('🔍 Select Recommendation Type')

# --- Upload CSV ---
uploaded_file = st.sidebar.file_uploader("Upload Ratings CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader('📊 Dataset Preview')
    st.dataframe(df.head(10), use_container_width=True)

    # Data cleaning
    df.dropna(inplace=True)

    recommender_type = st.sidebar.radio('Recommendation Method:',
                        ['Rank-Based', 'User-Based CF', 'Model-Based CF (SVD)'])

    # --- Common helper functions ---

    def get_top_products(df, n=10):
        product_group = df.groupby('Product_ID').agg({'Rating': ['count', 'mean']})
        product_group.columns = ['Rating_Count', 'Average_Rating']
        product_group = product_group.sort_values(by='Rating_Count', ascending=False)
        return product_group.head(n)

    @st.cache_resource
    def train_svd_model(df):
        reader = Reader(rating_scale=(1,5))
        data = Dataset.load_from_df(df[['User_ID', 'Product_ID', 'Rating']], reader)
        trainset, _ = train_test_split(data, test_size=0.2)
        model = SVD()
        model.fit(trainset)
        return model

    # --- Rank-Based Recommendation ---
    if recommender_type == 'Rank-Based':
        st.header('🏆 Rank-Based Recommendation')

        top_products = get_top_products(df)

        st.subheader('Top 10 Popular Products:')
        st.dataframe(top_products, use_container_width=True)

    # --- User-Based Collaborative Filtering ---
    elif recommender_type == 'User-Based CF':
        st.header('👥 User-Based Collaborative Filtering')

        user_id = st.text_input('Enter User ID for recommendations:')

        if user_id:
            if user_id in df['User_ID'].values:
                user_product_matrix = df.pivot_table(index='User_ID', columns='Product_ID', values='Rating')
                user_similarity = user_product_matrix.T.corr().loc[user_id]
                similar_users = user_similarity.sort_values(ascending=False).dropna().index.tolist()

                recommended_products = []
                for similar_user in similar_users:
                    products = df[df['User_ID'] == similar_user]['Product_ID'].tolist()
                    recommended_products.extend(products)
                    if len(recommended_products) > 5:
                        break

                recommended_products = list(set(recommended_products))
                st.success('🎯 Top Recommendations for You:')
                for prod in recommended_products[:5]:
                    st.markdown(f"✅ **Product ID:** {prod}")
            else:
                st.error('⚠️ User ID not found!')

    # --- Model-Based Collaborative Filtering ---
    elif recommender_type == 'Model-Based CF (SVD)':
        st.header('🧠 Model-Based Collaborative Filtering (SVD)')

        with st.spinner('Training SVD model...'):
            model = train_svd_model(df)

        st.success('✅ Model trained successfully!')

        user_id = st.text_input('Enter User ID for ML Recommendations:')

        if user_id:
            if user_id in df['User_ID'].values:
                all_products = df['Product_ID'].unique()
                rated_products = df[df['User_ID'] == user_id]['Product_ID'].tolist()

                predictions = []
                for product_id in all_products:
                    if product_id not in rated_products:
                        pred = model.predict(user_id, product_id)
                        predictions.append((product_id, pred.est))

                predictions.sort(key=lambda x: x[1], reverse=True)

                st.success('🎯 Top Predicted Products for You:')
                for product, score in predictions[:5]:
                    st.markdown(f"⭐ **Product ID:** {product} | Predicted Rating: {round(score,2)}")
            else:
                st.error('⚠️ User ID not found!')
else:
    st.info('👈 Please upload a CSV file to continue.')

