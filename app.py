import streamlit as st
import pandas as pd

st.set_page_config(page_title="E-commerce Recommender", page_icon="🛒", layout="wide")

st.title("🛍️ E-commerce Product Recommendation System")

# Upload the products.csv
st.sidebar.header("Upload your CSV files")

uploaded_products = st.sidebar.file_uploader("📦 Upload products.csv", type=["csv"])

# Rename columns in case there are extra spaces or mismatches
products_df.columns = [col.strip() for col in products_df.columns]

# Now try displaying again
if uploaded_products is not None:
    st.success("✅ Products file uploaded successfully!")
    st.subheader("🛒 Available Products")
    
    for index, row in products_df.iterrows():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(row['Product_Image'], width=150)
        with col2:
            st.subheader(row['Product_Name'])
            st.caption(f"Product ID: {row['Product_ID']}")


else:
    st.warning("⚠️ Please upload the products.csv file from the sidebar to proceed.")

