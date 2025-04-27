import streamlit as st
import pandas as pd
import numpy as np

# --- Streamlit Page Config ---
st.set_page_config(page_title="🛒 E-commerce Recommender", layout="wide")

# --- App Title ---
st.title('🛒 E-commerce Product Recommendation System')
st.markdown('''
Welcome to your personalized product recommendation engine!  
Upload your datasets and start getting awesome suggestions 🚀
''')

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/263/263115.png", width=100)
st.sidebar.title('🔍 Select Recommendation Type')

# --- Upload Products CSV ---
st.sidebar.header('Step 1: Upload products.csv')
products_file = st.sidebar.file_uploader("📦 Upload Products CSV", type=["csv"])

if products_file:
    products_df = pd.read_csv(products_file)
    st.sidebar.success("Products loaded ✅")
else:
    st.sidebar.info("Please upload products.csv to proceed.")

# --- Upload Ratings CSV ---
st.sidebar.header('Step 2: Upload ratings.csv')
ratings_file = st.sidebar.file_uploader("📄 Upload Ratings CSV", type=["csv"])

# --- Helper Functions ---
def get_product_details(product_id, products_df):
    try:
        row = products_df.loc[products_df['Product_ID'] == product_id].iloc[0]
        return row['Product_Name'], row['Product_Image']
    except:
        return "Unknown Product", None

def get_top_products(df, n=10):
    product_group = df.groupby('Product_ID').agg({'Rating': ['count', 'mean']})
    product_group.columns = ['Rating_Count', 'Average_Rating']
    product_group = product_group.sort_values(by='Rating_Count', ascending=False)
    return product_group.head(n)

def compute_svd_recommendations(df, user_id, n_recommendations=5):
    pivot = df.pivot_table(index='User_ID', columns='Product_ID', values='Rating').fillna(0)
    if pivot.shape[0] * pivot.shape[1] > 1e7:  # Arbitrary threshold
        st.warning("Dataset too large for exact SVD, consider reducing file size.", icon="⚠️")
        pivot = pivot.sample(frac=0.5, axis=1)
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

# --- Main Logic ---
if products_file and ratings_file:
    df = pd.read_csv(ratings_file)

    # Handling very large files
    if len(df) > 500000:
        st.warning(f"⚠️ Large dataset detected: {len(df)} rows. Sampling 5% for faster performance.", icon="⚡")
        df = df.sample(frac=0.05, random_state=42)
    elif len(df) > 100000:
        st.info(f"ℹ️ Medium dataset detected: {len(df)} rows. Sampling 10%.", icon="📊")
        df = df.sample(frac=0.1, random_state=42)

    st.subheader('📊 Dataset Preview')
    st.dataframe(df.head(10), use_container_width=True)

    # Cleaning
    df.dropna(inplace=True)

    recommender_type = st.sidebar.radio('Step 3: Choose Recommendation Method:',
                        ['🏆 Rank-Based', '👥 User-Based CF', '🧠 Model-Based CF (SVD Approximation)'])

    # --- Rank-Based Recommendation ---
    if recommender_type == '🏆 Rank-Based':
        st.header('🏆 Rank-Based Recommendation')

        top_products = get_top_products(df)

        st.subheader('Top 10 Popular Products:')
        for pid in top_products.index:
            name, image = get_product_details(pid, products_df)
            st.markdown(f"### 🛒 {name}")
            if image:
                st.image(image, width=200)
            st.caption(f"Product ID: {pid}")
            st.markdown("---")

    # --- User-Based Collaborative Filtering ---
    elif recommender_type == '👥 User-Based CF':
        st.header('👥 User-Based Collaborative Filtering')

        user_id = st.text_input('Enter User ID for recommendations:')

        if user_id:
            if int(user_id) in df['User_ID'].unique():
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
                    name, image = get_product_details(prod, products_df)
                    st.markdown(f"### 🛒 {name}")
                    if image:
                        st.image(image, width=200)
                    st.caption(f"Product ID: {prod}")
                    st.markdown("---")
            else:
                st.error('⚠️ User ID not found!')

    # --- Model-Based Collaborative Filtering (SVD Approximation) ---
    elif recommender_type == '🧠 Model-Based CF (SVD Approximation)':
        st.header('🧠 Model-Based Collaborative Filtering (SVD Approximation)')

        user_id = st.text_input('Enter User ID for ML Recommendations:')

        if user_id:
            try:
                recommended_products = compute_svd_recommendations(df, int(user_id))

                if recommended_products:
                    st.success('🎯 Top Predicted Products for You:')
                    for prod in recommended_products:
                        name, image = get_product_details(prod, products_df)
                        st.markdown(f"### 🛒 {name}")
                        if image:
                            st.image(image, width=200)
                        st.caption(f"Product ID: {prod}")
                        st.markdown("---")
                else:
                    st.warning("⚠️ No recommendations found for this User ID.", icon="⚠️")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
else:
    st.info("Please upload both products.csv and ratings.csv files to continue.")
