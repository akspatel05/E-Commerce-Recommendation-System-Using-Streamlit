import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex

# --- Streamlit Page Config ---
st.set_page_config(page_title="Ecommerce Product Recommender", layout="wide")

# --- App Title ---
st.title('🛒 E-commerce Product Recommendation System')
st.markdown('''
Welcome to your personalized product recommendation engine!  
Upload your ratings and products dataset and select a recommendation method from the sidebar! 🚀
''')

# --- Sidebar Uploads ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/263/263115.png", width=120)
st.sidebar.title('🔍 Select Recommendation Type')

# Upload products.csv
products_file = st.sidebar.file_uploader("📦 Upload products.csv", type=["csv"])

if products_file is not None:
    products_df = pd.read_csv(products_file)
    st.sidebar.success("✅ Products loaded successfully!")
else:
    st.sidebar.warning("⚠️ Please upload products.csv to show product names/images.")

# Upload ratings CSV
ratings_file = st.sidebar.file_uploader("⭐ Upload Ratings CSV", type=["csv"])

if ratings_file is not None:
    df = pd.read_csv(ratings_file)

    # --- Data Preprocessing ---
    if df.shape[0] > 500000:
        st.sidebar.warning("Large dataset detected! Sampling 5% for performance 🚀")
        df = df.sample(frac=0.05, random_state=42)

    df.dropna(inplace=True)
    df['User_ID'] = df['User_ID'].astype(str)
    df['Product_ID'] = df['Product_ID'].astype(str)

    st.success(f"✅ Using {len(df)} rows of ratings after preprocessing.")

    st.subheader('📊 Dataset Preview')
    st.dataframe(df.head(10), use_container_width=True)

    recommender_type = st.sidebar.radio('Recommendation Method:',
                        ['🏆 Rank-Based', '👥 User-Based CF', '🧠 Model-Based CF (SVD Approximation)', '🔍 ANN-Based'])

    # --- Helper Functions ---
    def get_top_products(df, n=10):
        product_group = df.groupby('Product_ID').agg({'Rating': ['count', 'mean']})
        product_group.columns = ['Rating_Count', 'Average_Rating']
        product_group = product_group.sort_values(by='Rating_Count', ascending=False)
        return product_group.head(n)

    def compute_svd_recommendations(df, user_id, n_recommendations=5):
        pivot = df.pivot_table(index='User_ID', columns='Product_ID', values='Rating').fillna(0)
        sparse_matrix = csr_matrix(pivot.values)

        # Perform SVD using TruncatedSVD for efficiency
        from sklearn.decomposition import TruncatedSVD
        svd = TruncatedSVD(n_components=50, random_state=42)
        svd_pred = svd.fit_transform(sparse_matrix)

        # Reconstruct the full prediction matrix
        svd_df = pd.DataFrame(np.dot(svd_pred, svd.components_), index=pivot.index, columns=pivot.columns)

        if user_id not in svd_df.index:
            return []

        user_ratings = svd_df.loc[user_id]
        already_rated = df[df['User_ID'] == user_id]['Product_ID'].tolist()

        recommended = user_ratings.drop(labels=already_rated, errors='ignore')
        top_products = recommended.sort_values(ascending=False).head(n_recommendations)

        return top_products.index.tolist()

    def display_product(product_id):
        if products_file is None:
            st.markdown(f"🛒 **Product ID:** {product_id}")
            return

        product = products_df[products_df['Product_ID'] == product_id]

        if not product.empty:
            row = product.iloc[0]
            st.image(row['Product_Image'], width=200)
            st.markdown(f"**🛍️ {row['Product_Name']}**")
        else:
            st.markdown(f"🛒 **Unknown Product**\n\nProduct ID: {product_id}")

    # --- ANN-Specific Helper Functions ---
    def create_ann_index(df, n_trees=10):
        """Create an ANN index for faster product recommendations."""
        user_product_matrix = df.pivot_table(index='User_ID', columns='Product_ID', values='Rating').fillna(0)
        user_product_matrix_sparse = csr_matrix(user_product_matrix.values)
        
        # Initialize the Annoy index
        annoy_index = AnnoyIndex(user_product_matrix_sparse.shape[1], 'angular')  # Using angular distance
        
        # Add items to the index
        for i in range(user_product_matrix_sparse.shape[0]):
            annoy_index.add_item(i, user_product_matrix_sparse[i].toarray()[0])

        annoy_index.build(n_trees)
        return annoy_index, user_product_matrix.index, user_product_matrix

    def get_ann_recommendations(user_id, annoy_index, user_index, top_n=5):
        """Get recommendations using ANN (Nearest Neighbors)."""
        user_idx = user_index.get_loc(user_id)
        neighbors = annoy_index.get_nns_by_item(user_idx, top_n, include_distances=False)
        
        recommended_products = set()
        
        # Retrieve the product IDs for the recommended users
        for idx in neighbors:
            recommended_products.update(user_product_matrix.iloc[idx].index[user_product_matrix.iloc[idx] > 0])
        
        return list(recommended_products)

    # --- Main Recommendation Logic ---
    if recommender_type == '🏆 Rank-Based':
        st.header('🏆 Rank-Based Recommendation')

        top_products = get_top_products(df)

        st.subheader('Top 10 Popular Products:')
        for prod_id in top_products.index:
            display_product(prod_id)

    elif recommender_type == '👥 User-Based CF':
        st.header('👥 User-Based Collaborative Filtering')
    
        active_users = df.groupby('User_ID').size()
        valid_user_ids = active_users[active_users > 0].index.tolist()
    
        if valid_user_ids:
            user_id = st.selectbox('Select User ID for recommendations:', valid_user_ids)
    
            if user_id:
                # Create the user-product matrix and fill missing ratings with 0
                user_product_matrix = df.pivot_table(index='User_ID', columns='Product_ID', values='Rating').fillna(0)
    
                # Check if selected user exists
                if user_id in user_product_matrix.index:
                    user_vector = user_product_matrix.loc[[user_id]]  # Keep as 2D for sklearn
    
                    # Compute cosine similarity
                    similarity_scores = cosine_similarity(user_vector, user_product_matrix)[0]
    
                    # Create a series with User IDs
                    user_similarity = pd.Series(similarity_scores, index=user_product_matrix.index)
    
                    # Remove self similarity
                    user_similarity = user_similarity.drop(user_id)
    
                    # Sort users by similarity
                    similar_users = user_similarity.sort_values(ascending=False).index.tolist()
    
                    recommended_products = []
    
                    for similar_user in similar_users:
                        products = df[df['User_ID'] == similar_user]['Product_ID'].tolist()
                        recommended_products.extend(products)
    
                        if len(recommended_products) > 5:
                            break
    
                    recommended_products = list(set(recommended_products))
    
                    if recommended_products:
                        st.success('🎯 Top Recommendations for You:')
                        for prod in recommended_products[:5]:
                            display_product(prod)
                    else:
                        st.warning("⚠️ No recommendations found.")
                else:
                    st.error('⚠️ Selected user not found in the ratings matrix.')
        else:
            st.warning("⚠️ No valid users found with ratings. Please check your dataset.")

    elif recommender_type == '🧠 Model-Based CF (SVD Approximation)':
        st.header('🧠 Model-Based Collaborative Filtering (SVD Approximation)')

        user_id = st.text_input('Enter User ID for ML Recommendations:')

        if user_id:
            try:
                recommended_products = compute_svd_recommendations(df, user_id)

                if recommended_products:
                    st.success('🎯 Top Predicted Products for You:')
                    for prod in recommended_products:
                        display_product(prod)
                else:
                    st.warning("⚠️ No recommendations found for this User ID.", icon="⚠️")

            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

    elif recommender_type == '🔍 ANN-Based':
        st.header('🔍 Approximate Nearest Neighbor (ANN) Recommendation')

        # Create the ANN index for faster recommendations
        with st.spinner('Building ANN index...'):
            annoy_index, user_index, user_product_matrix = create_ann_index(df)

        # Get recommendations
        user_id = st.text_input('Enter User ID for ANN Recommendations:')
        if user_id:
            with st.spinner('Finding recommendations...'):
                recommended_products = get_ann_recommendations(user_id, annoy_index, user_index)
                
                if recommended_products:
                    st.success('🎯 Top Recommendations for You:')
                    for prod in recommended_products[:5]:
                        display_product(prod)
                else:
                    st.warning("⚠️ No recommendations found for this User ID.", icon="⚠️")

else:
    st.warning('🚀 Please upload a ratings CSV to start!')
