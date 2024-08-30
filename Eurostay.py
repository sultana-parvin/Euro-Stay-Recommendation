import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path
import pydeck as pdk

def get_room_type_image(room_type):
    image_map = {
        "Private room": "https://images.unsplash.com/photo-1522771739844-6a9f6d5f14af?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1471&q=80",
        "Shared room": "https://images.unsplash.com/photo-1555854877-bab0e564b8d5?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1469&q=80",
        "Entire home/apt": "https://images.unsplash.com/photo-1502672260266-1c1ef2d93688?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1380&q=80"
    }
    return image_map.get(room_type, "https://images.unsplash.com/photo-1518780664697-55e3ad937233?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1465&q=80")

# Set page config and style
st.set_page_config(page_title="EuroStay Property Recommendation", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: #e0f2f1;
    }
    .stApp {
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3 {
        color: #26a69a;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #26a69a;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease 0s;
    }
    .stButton>button:hover {
        background-color: #2bbbad;
        box-shadow: 0px 15px 20px rgba(46, 229, 157, 0.4);
        transform: translateY(-7px);
    }
    .stSelectbox, .stSlider {
        color: #26a69a;
    }
    .css-1aumxhk {
        background-color: #b2dfdb;
    }
</style>
""", unsafe_allow_html=True)

# Load datasets
@st.cache_data
def load_data():
    europe_data = pd.read_csv("data/europe_data.csv")
    europe_data_processed = pd.read_csv("data/europe_data_processed.csv")
    return europe_data, europe_data_processed

europe_data, europe_data_processed = load_data()

# Display logo
logo_path = Path("data/logo Eurostay.png")
st.image(str(logo_path), width=200)

st.title("EuroStay Property Recommendation")

# Sidebar for user inputs
st.sidebar.header("🏠 Your Preferences")

# City selection
city = st.sidebar.selectbox("🌆 Select City", europe_data['city'].unique())

# Other filters
room_type = st.sidebar.selectbox("🛏️ Room Type", ["Private room", "Shared room", "Entire home/apt"])
person_capacity = st.sidebar.slider("👥 Person Capacity", 1, int(europe_data['person_capacity'].max()), 2)
price_category = st.sidebar.selectbox("💰 Price Category", europe_data['price_category'].unique())
max_distance = st.sidebar.slider("🚶‍♂️ Maximum Distance from Center (km)", 0.0, 10.0, 5.0, 0.1)
max_metro_distance = st.sidebar.slider("🚇 Maximum Distance from Metro (km)", 0.0, 5.0, 1.0, 0.1)

# New filter for week time
week_time = st.sidebar.selectbox("📅 Week Time", ["Any", "Weekdays", "Weekends"])

# New filters
superhost_only = st.sidebar.checkbox("🌟 Superhost properties only")
min_attr_index = st.sidebar.slider("🏛️ Minimum Attraction Index", 0.0, 1.0, 0.5, 0.01)
min_rest_index = st.sidebar.slider("🍽️ Minimum Restaurant Index", 0.0, 1.0, 0.5, 0.01)

# Recommendation system
if st.sidebar.button("🔍 Find Properties"):
    # Filter data by selected city
    city_mask = europe_data_processed[f'city_{city.lower()}'] == 1
    city_filtered_df = europe_data_processed[city_mask]

    # Apply user preferences
    room_type_mask = city_filtered_df[f'room_type_{room_type}'] == 1
    person_capacity_mask = europe_data.loc[city_filtered_df.index, 'person_capacity'] >= person_capacity
    price_category_mask = europe_data.loc[city_filtered_df.index, 'price_category'] == price_category
    distance_mask = city_filtered_df['dist'] <= max_distance
    metro_mask = city_filtered_df['metro_dist'] <= max_metro_distance

    # Apply week time filter
    if week_time == "Weekdays":
        week_time_mask = city_filtered_df['week time_weekdays'] == 1
    elif week_time == "Weekends":
        week_time_mask = city_filtered_df['week time_weekends'] == 1
    else:
        week_time_mask = pd.Series([True] * len(city_filtered_df))

    # Apply new filters
    if superhost_only:
        superhost_mask = city_filtered_df['host_is_superhost'] == 1
    else:
        superhost_mask = pd.Series([True] * len(city_filtered_df))
    
    attr_index_mask = city_filtered_df['attr_index_norm'] >= min_attr_index
    rest_index_mask = city_filtered_df['rest_index_norm'] >= min_rest_index

    # Combine all filters
    final_mask = (room_type_mask & person_capacity_mask & price_category_mask &
                  distance_mask & metro_mask & week_time_mask & superhost_mask &
                  attr_index_mask & rest_index_mask)
    filtered_df = city_filtered_df[final_mask]

    if filtered_df.empty:
        st.warning("No properties match your criteria. Please try adjusting your filters.")
    else:
        # Select features for recommendation
        features = ['cleanliness_rating', 'guest_satisfaction_overall', 'dist', 'metro_dist']

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(filtered_df[features])

        # Find nearest neighbors
        nn = NearestNeighbors(n_neighbors=min(10, len(X_scaled)), metric='euclidean')
        nn.fit(X_scaled)
        distances, indices = nn.kneighbors(X_scaled)

        # Get recommended properties
        recommended_indices = filtered_df.iloc[indices[0]].index
        recommended_properties = europe_data.loc[recommended_indices]

        # Display recommendations
        st.subheader("🏠 Recommended Properties")
        #st.dataframe(recommended_properties[['realSum', 'room_type', 'guest_satisfaction_overall', 'price_category', 'dist', 'metro_dist']])
        for _, property in recommended_properties.iterrows():
          col1, col2 = st.columns([1, 3])
          with col1:
            image_url = get_room_type_image(property['room_type'])
            st.image(image_url, use_column_width=True)
          with col2:
             st.write(f"**{property['room_type']} in {property['city']}**")
             st.write(f"Price: €{property['realSum']:.2f}")
             st.write(f"Guest Satisfaction: {property['guest_satisfaction_overall']:.1f}/100")
             st.write(f"Distance from Center: {property['dist']:.2f} km")
             st.write(f"Distance from Metro: {property['metro_dist']:.2f} km")
             st.write(f"Attraction Index: {property['attr_index_norm']:.2f}")
             st.write(f"Restaurant Index: {property['rest_index_norm']:.2f}")
             st.write(f"Superhost: {'Yes' if property['host_is_superhost'] else 'No'}")


        # Add the map to display the locations of the recommended properties
        st.subheader("📍 Property Locations on Map")
        map_data = recommended_properties[['lat', 'lng']].rename(columns={'lng': 'lon'})
        st.map(map_data)



        # Visualize recommendations
        st.subheader("💰 Price vs. Distance from Center")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(recommended_properties['dist'],
                            recommended_properties['realSum'],
                            c=recommended_properties['guest_satisfaction_overall'],
                            cmap='viridis',
                            s=100, alpha=0.7)
        ax.set_xlabel("Distance from Center (km)")
        ax.set_ylabel("Price")
        plt.colorbar(scatter, label='Guest Satisfaction')
        ax.set_title("Property Recommendations")
        st.pyplot(fig)

# Display some stats about the filtered dataset
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Quick Stats")
st.sidebar.write(f"Properties in {city}: {len(europe_data[europe_data['city'] == city])}")
st.sidebar.write(f"Average price in {city}: €{europe_data[europe_data['city'] == city]['realSum'].mean():.2f}")
st.sidebar.write(f"Average rating in {city}: {europe_data[europe_data['city'] == city]['guest_satisfaction_overall'].mean():.2f}/100")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by EuroStay")