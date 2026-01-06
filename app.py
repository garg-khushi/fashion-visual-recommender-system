import os
import json
import torch
import pandas as pd
import streamlit as st
from PIL import Image

# Debug function to print messages in the Streamlit app
def debug(message):
    st.write(f"[DEBUG] {message}")

# Load precomputed data with caching
@st.cache_data
def load_data():
    try:
        styles = pd.read_csv('INPUT_DATA/styles.csv', on_bad_lines='skip')
        debug(f"Loaded styles.csv with {len(styles)} rows")
        with open('OUTPUT_DATA/similar_products.json', 'r') as f:
            similar_products = json.load(f)
        debug(f"Loaded similar_products.json with {len(similar_products)} entries")
        return styles, similar_products
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Load precomputed features and product IDs
features = torch.load('OUTPUT_DATA/features_resnet50.pt')
with open('OUTPUT_DATA/ids_resnet50.txt', 'r') as f:
    feature_ids = f.read().splitlines()

styles, similar_products = load_data()
if styles is None or similar_products is None:
    st.stop()

# Expanded size charts for personalized size recommendations
size_charts = {
    'Shirts': {'S': (0, 36), 'M': (36, 40), 'L': (40, 44), 'XL': (44, 48)},
    'Tshirts': {'S': (0, 36), 'M': (36, 40), 'L': (40, 44), 'XL': (44, 48)},
    'Jeans': {'28': (0, 28), '30': (28, 30), '32': (30, 32), '34': (32, 34)},
    'Dresses': {'S': (0, 34), 'M': (34, 38), 'L': (38, 42), 'XL': (42, 46)},
    'Jackets': {'S': (0, 38), 'M': (38, 42), 'L': (42, 46), 'XL': (46, 50)},
    'Headband': {'One Size': (0, float('inf'))},
    'Watches': {'One Size': (0, float('inf'))},
}

def get_recommended_size(product_type, chest=None, waist=None):
    debug(f"Checking size for articleType: {product_type}, Chest: {chest}, Waist: {waist}")
    if product_type not in size_charts:
        return "Not applicable for this item"
    chart = size_charts[product_type]
    measurement = chest if product_type in ['Shirts', 'Tshirts', 'Dresses'] else waist
    if measurement is None or measurement <= 0:
        return "Please enter measurements"
    for size, (min_val, max_val) in chart.items():
        if min_val <= measurement < max_val:
            return size
    debug(f"Measurement {measurement} out of range for {product_type}")
    return "Out of range"

def clean_product_id(rec_id):
    """
    Converts a product ID that might be in the format 'tensor(15971)' or a plain string to an integer.
    """
    if isinstance(rec_id, str):
        if rec_id.startswith("tensor(") and rec_id.endswith(")"):
            num_str = rec_id[len("tensor("):-1]
            return int(num_str)
        else:
            return int(rec_id)
    elif isinstance(rec_id, torch.Tensor):
        return rec_id.item()
    else:
        return int(rec_id)

def get_content_based_recommendations(user_product_id, features, feature_ids, styles, top_n=5):
    """
    Returns the top_n product IDs most similar to user_product_id based on cosine similarity,
    filtering to include only products with the same gender.
    """
    user_product_id_clean = clean_product_id(user_product_id)
    user_product_id_str = str(user_product_id_clean)
    
    try:
        target_idx = feature_ids.index(user_product_id_str)
    except ValueError:
        return []
    
    target_feat = features[target_idx]
    
    try:
        base_product_gender = styles[styles['id'] == user_product_id_clean]['gender'].iloc[0]
    except IndexError:
        base_product_gender = None
    
    # Filter indices to those with the same gender
    filtered_indices = []
    for i, pid in enumerate(feature_ids):
        try:
            pid_int = clean_product_id(pid)
        except ValueError:
            continue
        product_gender_series = styles[styles['id'] == pid_int]['gender']
        if not product_gender_series.empty:
            product_gender = product_gender_series.iloc[0]
            if base_product_gender is None or product_gender == base_product_gender:
                filtered_indices.append(i)
    
    similarities = torch.nn.functional.cosine_similarity(target_feat.unsqueeze(0), features, dim=1)
    filtered_similarities = [(i, similarities[i].item()) for i in filtered_indices]
    filtered_similarities.sort(key=lambda x: x[1], reverse=True)
    
    recommended_ids = []
    for i, sim in filtered_similarities:
        pid = feature_ids[i]
        if pid != user_product_id_str:
            recommended_ids.append(pid)
        if len(recommended_ids) >= top_n:
            break
    return recommended_ids

# def get_content_based_recommendations(user_product_id, features, feature_ids, styles, chest, waist, top_n=5, target_gender=None):
#     """
#     Returns the top_n product IDs most similar to user_product_id based on cosine similarity,
#     filtering to include only products with the same gender and, if measurements are provided,
#     with a matching recommended size.
    
#     Parameters:
#       - user_product_id: The reference product ID (as int, string, or tensor-like string)
#       - features: Precomputed feature tensor of shape [num_products, feature_dim]
#       - feature_ids: List of product IDs corresponding to the features (as strings)
#       - styles: Pandas DataFrame containing product metadata (including a 'gender' and 'articleType' column)
#       - chest, waist: User measurement inputs (floats)
#       - top_n: Number of recommendations to return
#       - target_gender: If provided, only products matching this gender are considered.
#     """
#     user_product_id_clean = clean_product_id(user_product_id)
#     user_product_id_str = str(user_product_id_clean)
    
#     try:
#         target_idx = feature_ids.index(user_product_id_str)
#     except ValueError:
#         return []
    
#     target_feat = features[target_idx]
    
#     # Determine the gender to filter by.
#     if target_gender is None:
#         try:
#             base_product_gender = styles[styles['id'] == user_product_id_clean]['gender'].iloc[0]
#         except IndexError:
#             base_product_gender = None
#     else:
#         base_product_gender = target_gender
    
#     # Compute the recommended size for the base product (if measurements are provided).
#     base_product_row = styles[styles['id'] == user_product_id_clean]
#     if not base_product_row.empty and (chest > 0 or waist > 0):
#         base_product_type = base_product_row.iloc[0]['articleType']
#         base_recommended_size = get_recommended_size(base_product_type, chest, waist)
#     else:
#         base_recommended_size = None

#     # Filter candidate indices: by gender and (if measurements are provided) matching recommended size.
#     filtered_indices = []
#     for i, pid in enumerate(feature_ids):
#         try:
#             pid_int = clean_product_id(pid)
#         except ValueError:
#             continue
#         product_rows = styles[styles['id'] == pid_int]
#         if product_rows.empty:
#             continue
#         product_gender = product_rows.iloc[0]['gender']
#         product_type = product_rows.iloc[0]['articleType']
        
#         # Gender filtering
#         if base_product_gender is not None and product_gender != base_product_gender:
#             continue
        
#         # If measurements are provided, check that the candidate's recommended size matches the base's.
#         if chest > 0 or waist > 0:
#             candidate_size = get_recommended_size(product_type, chest, waist)
#             if candidate_size != base_recommended_size:
#                 continue
                
#         filtered_indices.append(i)
    
#     # Compute cosine similarities between the target feature and all features.
#     similarities = torch.nn.functional.cosine_similarity(target_feat.unsqueeze(0), features, dim=1)
#     filtered_similarities = [(i, similarities[i].item()) for i in filtered_indices]
#     filtered_similarities.sort(key=lambda x: x[1], reverse=True)
    
#     recommended_ids = []
#     for i, sim in filtered_similarities:
#         pid = feature_ids[i]
#         if pid != user_product_id_str:
#             recommended_ids.append(pid)
#         if len(recommended_ids) >= top_n:
#             break
#     return recommended_ids


def overlay_dress(user_img, dress_img, x, y, scale):
    try:
        dress_width = int(dress_img.width * scale)
        dress_height = int(dress_img.height * scale)
        dress_img = dress_img.resize((dress_width, dress_height), Image.Resampling.LANCZOS)
        if dress_img.mode != 'RGBA':
            dress_img = dress_img.convert('RGBA')
        if user_img.mode != 'RGBA':
            user_img = user_img.convert('RGBA')
        x = max(0, min(x, user_img.width - dress_width))
        y = max(0, min(y, user_img.height - dress_height))
        user_img.paste(dress_img, (x, y), dress_img)
        return user_img
    except Exception as e:
        st.error(f"Error in overlay_dress: {e}")
        return user_img

# --- Streamlit App Layout ---

st.title("Style Point - Your Fashion Portal")

# Sidebar: User measurements for personalized size recommendations
st.sidebar.header("Your Measurements")
chest = st.sidebar.number_input("Chest (inches)", min_value=0.0, value=0.0, key="chest")
waist = st.sidebar.number_input("Waist (inches)", min_value=0.0, value=0.0, key="waist")
if chest > 0 or waist > 0:
    st.sidebar.success("Measurements saved!")

# Navigation options
page = st.sidebar.selectbox("Navigate", ["Home", "Product Details", "Virtual Try-On"])

# --- Home Page: Featured Products with AI-driven, Content-based Recommendations ---
if page == "Home":
    st.header("Featured Products")
    
    # Determine a base product for recommendations.
    if 'favorite_product' in st.session_state:
        base_product = st.session_state['favorite_product']
    else:
        base_product = feature_ids[0]
        st.session_state['favorite_product'] = base_product

    st.write(f"Recommendations based on Product ID: {base_product}")

    # Generate content-based recommendations filtered by gender
    recommended_ids = get_content_based_recommendations(base_product, features, feature_ids, styles, top_n=5)
    
    # Fallback: if no recommendations, show a random sample of products
    if not recommended_ids:
        st.write("No content-based recommendations available. Showing default products:")
        recommended_ids = styles.sample(5)['id'].astype(str).tolist()
    
    # Display products in columns
    cols = st.columns(5)
    for col, rec_id in zip(cols, recommended_ids):
        with col:
            rec_id_int = clean_product_id(rec_id)
            img_path = f"INPUT_DATA/images/{rec_id_int}.jpg"
            if os.path.exists(img_path):
                st.image(img_path, width=120, caption=f"ID: {rec_id_int}")
            else:
                st.image("https://via.placeholder.com/120", width=120, caption="Image unavailable")
            product_name = styles[styles['id'] == rec_id_int]['productDisplayName'].values[0]
            st.write(product_name)
            if st.button("View", key=f"view_{rec_id_int}"):
                st.session_state['current_product'] = rec_id_int

# if page == "Home":
#     st.header("Featured Products")
    
#     # Gender filter selection from the sidebar.
#     gender_filter = st.sidebar.selectbox("Select Gender", ["Men", "Women"], index=0)
    
#     # Determine a base product for recommendations.
#     # If a favorite product is already set, use it; otherwise choose one from the selected gender.
#     if 'favorite_product' in st.session_state:
#         base_product = st.session_state['favorite_product']
#     else:
#         candidates = styles[styles['gender'] == gender_filter]['id'].astype(str).tolist()
#         if candidates:
#             base_product = candidates[0]
#             st.session_state['favorite_product'] = base_product
#         else:
#             base_product = feature_ids[0]
#             st.session_state['favorite_product'] = base_product

#     # Ensure the base product matches the selected gender.
#     base_product_gender = styles[styles['id'] == clean_product_id(base_product)]['gender'].iloc[0]
#     if base_product_gender != gender_filter:
#         candidates = styles[styles['gender'] == gender_filter]['id'].astype(str).tolist()
#         if candidates:
#             base_product = candidates[0]
#             st.session_state['favorite_product'] = base_product

#     st.write(f"Recommendations based on Product ID: {base_product} for {gender_filter}")
    
#     # Generate content-based recommendations filtered by gender and size (using measurement inputs).
#     recommended_ids = get_content_based_recommendations(
#         base_product, features, feature_ids, styles,
#         chest, waist, top_n=5, target_gender=gender_filter
#     )
    
#     # Fallback: if no recommendations were found, show a random sample of products from the selected gender.
#     if not recommended_ids:
#         st.write("No content-based recommendations available. Showing default products:")
#         candidates = styles[styles['gender'] == gender_filter]['id'].astype(str).tolist()
#         recommended_ids = candidates[:5]
    
#     # Display each recommended product.
#     cols = st.columns(5)
#     for col, rec_id in zip(cols, recommended_ids):
#         with col:
#             rec_id_int = clean_product_id(rec_id)
#             img_path = f"INPUT_DATA/images/{rec_id_int}.jpg"
#             if os.path.exists(img_path):
#                 st.image(img_path, width=120, caption=f"ID: {rec_id_int}")
#             else:
#                 st.image("https://via.placeholder.com/120", width=120, caption="Image unavailable")
#             product_name = styles[styles['id'] == rec_id_int]['productDisplayName'].values[0]
#             st.write(product_name)
#             if st.button("View", key=f"view_{rec_id_int}"):
#                 st.session_state['current_product'] = rec_id_int


# --- Product Details Page ---
elif page == "Product Details":
    if 'current_product' not in st.session_state:
        st.warning("Please select a product from the Home page.")
    else:
        id_ = int(st.session_state['current_product'])
        product_rows = styles[styles['id'] == id_]
        if product_rows.empty:
            st.error(f"No product found with ID {id_}")
        else:
            product = product_rows.iloc[0]
            st.header(product['productDisplayName'])
            img_path = f"INPUT_DATA/images/{id_}.jpg"
            if os.path.exists(img_path):
                st.image(img_path, width=300)
            else:
                debug(f"Image not found: {img_path}")
                st.image("https://via.placeholder.com/300", width=300, caption="Image unavailable")

            st.write(f"**Category**: {product['masterCategory']}")
            st.write(f"**Subcategory**: {product['subCategory']}")
            st.write(f"**Type**: {product['articleType']}")
            st.write(f"**Color**: {product['baseColour']}")
            st.write(f"**Season**: {product['season']}")
            st.write(f"**Year**: {product['year']}")
            st.write(f"**Usage**: {product['usage']}")
            size = get_recommended_size(product['articleType'], chest, waist)
            st.write(f"**Recommended Size**: {size}")

            st.subheader("You May Also Like")
            similar_ids = similar_products.get(str(id_), [])
            if not similar_ids:
                st.write("No recommendations available.")
                fallback_ids = styles.sample(5)['id'].tolist()
                st.write("Showing default recommendations:")
                for fid in fallback_ids:
                    st.image(f"INPUT_DATA/images/{fid}.jpg", width=120)
            else:
                cols = st.columns(5)
                for col, sim_id in zip(cols, similar_ids[:5]):
                    with col:
                        sim_id_int = clean_product_id(sim_id)
                        sim_img_path = f"INPUT_DATA/images/{sim_id_int}.jpg"
                        if os.path.exists(sim_img_path):
                            st.image(sim_img_path, width=120)
                        else:
                            st.image("https://via.placeholder.com/120", width=120, caption="Image unavailable")
                        sim_rows = styles[styles['id'] == sim_id_int]
                        sim_name = sim_rows['productDisplayName'].values[0] if not sim_rows.empty else "Unknown"
                        st.write(sim_name)
                        if st.button("View", key=f"sim_{sim_id_int}"):
                            st.session_state['current_product'] = sim_id_int

# --- Virtual Try-On Page ---
elif page == "Virtual Try-On":
    st.header("Virtual Try-On")
    dresses = styles[styles['articleType'] == 'Dresses']
    if dresses.empty:
        st.error("No dresses found in the dataset.")
    else:
        dress_options = {row['productDisplayName']: row['id'] for _, row in dresses.iterrows()}
        selected_dress = st.selectbox("Choose a dress", list(dress_options.keys()))
        dress_id = dress_options[selected_dress]

        user_img_file = st.file_uploader("Upload your full-body photo", type=['jpg', 'png'])
        if user_img_file:
            user_img = Image.open(user_img_file).convert('RGBA')
            st.image(user_img, caption="Your Photo", width=300)
            dress_img_path = f"INPUT_DATA/transparent_dresses/{dress_id}.png"
            if os.path.exists(dress_img_path):
                dress_img = Image.open(dress_img_path).convert('RGBA')
                st.write("Adjust the dress position and size:")
                x = st.slider("X Position", 0, max(0, user_img.width - 50), 100)
                y = st.slider("Y Position", 0, max(0, user_img.height - 50), 200)
                scale = st.slider("Scale", 0.1, 2.0, 1.0, step=0.1)
                result_img = overlay_dress(user_img.copy(), dress_img, x, y, scale)
                st.image(result_img, caption="Try-On Result", width=300)
            else:
                st.error(f"Dress image not found: {dress_img_path}")
                debug(f"Ensure that 'INPUT_DATA/transparent_dresses/' contains {dress_id}.png")
