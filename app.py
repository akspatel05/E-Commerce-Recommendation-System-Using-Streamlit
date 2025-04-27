import streamlit as st
import pandas as pd
import numpy as np


# --- Streamlit Page Config ---
st.set_page_config(page_title="Ecommerce Product Recommender", layout="wide")

# --- App Title ---
st.title('🛒 E-commerce Product Recommendation System')
st.markdown('''
Welcome to your personalized product recommendation engine!  
Upload your ratings dataset and select a recommendation method from the sidebar! 🚀
''')

# --- Sidebar ---

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/263/263115.png", width=120)
st.sidebar.title('🔍 Select Recommendation Type')

with st.sidebar:
    uploaded_file = st.file_uploader("📄 Upload products.csv", type=["csv"])
    if uploaded_file is not None:
        products_df = pd.read_csv(uploaded_file)
        st.success("Products loaded ✅")
    else:
        st.info("Please upload products.csv to proceed.")

# --- Upload CSV ---
uploaded_file = st.sidebar.file_uploader("Upload Ratings CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Sample only 5% of data to work safely
    df = df.sample(frac=0.05, random_state=42)  # adjust the fraction if needed
    st.success(f"Using {len(df)} rows after sampling.")

    st.subheader('📊 Dataset Preview')
    st.dataframe(df.head(10), use_container_width=True)

    # Data cleaning
    df.dropna(inplace=True)

    recommender_type = st.sidebar.radio('Recommendation Method:',
                        ['Rank-Based', 'User-Based CF', 'Model-Based CF (SVD Approximation)'])

    # --- Common helper functions ---

    def get_top_products(df, n=10):
        product_group = df.groupby('Product_ID').agg({'Rating': ['count', 'mean']})
        product_group.columns = ['Rating_Count', 'Average_Rating']
        product_group = product_group.sort_values(by='Rating_Count', ascending=False)
        return product_group.head(n)

    def compute_svd_recommendations(df, user_id, n_recommendations=5):
        pivot = df.pivot_table(index='User_ID', columns='Product_ID', values='Rating').fillna(0)
        U, sigma, Vt = np.linalg.svd(pivot, full_matrices=False)
        
        sigma_diag_matrix = np.diag(sigma)
        svd_pred = np.dot(np.dot(U, sigma_diag_matrix), Vt)
        
        svd_df = pd.DataFrame(svd_pred, index=pivot.index, columns=pivot.columns)
        
        if user_id not in svd_df.index:
            return []
        
        user_ratings = svd_df.loc[user_id]
        already_rated = df[df['User_ID'] == user_id]['Product_ID'].tolist()
        
        recommended = user_ratings.drop(labels=already_rated)
        top_products = recommended.sort_values(ascending=False).head(n_recommendations)
        
        return top_products.index.tolist()

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
            if user_id in df['User_ID'].astype(str).values:
                user_product_matrix = df.pivot_table(index='User_ID', columns='Product_ID', values='Rating')
                user_similarity = user_product_matrix.corrwith(user_product_matrix.loc[int(user_id)]).dropna()
                similar_users = user_similarity.sort_values(ascending=False).index.tolist()

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

    # --- Model-Based Collaborative Filtering (SVD Approximation) ---
    elif recommender_type == 'Model-Based CF (SVD Approximation)':
        st.header('🧠 Model-Based Collaborative Filtering (SVD Approximation)')

        user_id = st.text_input('Enter User ID for ML Recommendations:')

        if user_id:
            try:
                recommended_products = compute_svd_recommendations(df, int(user_id))
    
                if recommended_products:
                    st.success('🎯 Top Predicted Products for You:')
                    for prod in recommended_products:
                        st.markdown(f"⭐ **Product ID:** {prod}")
                else:
                    st.warning("⚠️ No recommendations found for this User ID.", icon="⚠️")
    
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
