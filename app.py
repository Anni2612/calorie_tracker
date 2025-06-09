import streamlit as st
import pandas as pd
from PIL import Image
from transformers import pipeline
from sqlalchemy import create_engine, text

# 📌 Supabase DB setup
password = st.secrets["DB_PASSWORD"]
db_url = f"postgresql://postgres:{password}@db.hvnmzrrskpgdhyvumbrr.supabase.co:6543/postgres"
engine = create_engine(db_url)

# 🔍 Load model (cached)
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="nateraw/food")

classifier = load_model()

# === UI ===
st.title("🍽️ AI-Powered Food Nutrient Detector")
uploaded_file = st.file_uploader("📷 Upload a food image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    with st.spinner("🔍 Identifying food..."):
        prediction = classifier(image)[0]
        raw_label = prediction["label"]
        label = raw_label.replace("_", " ").lower()
    st.success(f"🤖 Detected: **{raw_label}**")

    # 🔎 Search for close matches
    with engine.connect() as conn:
        sql = text("SELECT DISTINCT food FROM foods WHERE food ILIKE :pattern ORDER BY food")
        similar_foods_df = pd.read_sql(sql, conn, params={"pattern": f"%{label}%"})
        similar_foods = similar_foods_df["food"].tolist()

    if not similar_foods:
        similar_foods = [label]

    # 👤 Manual correction
    st.markdown("🔧 **Refine or confirm food name (type any food to search DB):**")
    user_input = st.text_input("Type a food name (e.g., biryani, paratha)", value=label)

    food_matches = []
    if user_input.strip():
        with engine.connect() as conn:
            search_sql = text("SELECT DISTINCT food FROM foods WHERE food ILIKE :term ORDER BY food")
            food_df = pd.read_sql(search_sql, conn, params={"term": f"%{user_input.strip()}%"})
            food_matches = food_df["food"].tolist()

    if food_matches:
        final_food_label = st.selectbox("Select from DB matches:", food_matches, index=0)
    else:
        st.warning("⚠️ No match found — using your typed input.")
        final_food_label = user_input.strip().lower()

    servings = st.number_input("🍽️ Number of servings (per standard unit):", min_value=0.1, step=0.1, value=1.0)
    st.caption("ℹ️ Nutritional values are per 1 standard serving — e.g., 1 piece or 100g.")

    if st.button("🔎 Get Nutrition Info"):
        try:
            with engine.connect() as conn:
                query = text("SELECT * FROM foods WHERE LOWER(food) = :label LIMIT 1")
                result = pd.read_sql(query, conn, params={"label": final_food_label})

                if not result.empty:
                    st.subheader("📊 Nutrition Info")
                    for col in result.columns:
                        if col.lower() != "food":
                            val = result[col].iloc[0]
                            scaled = round(float(val) * servings, 2) if pd.notnull(val) else "N/A"
                            st.write(f"**{col.replace('_', ' ').title()}:** {scaled}")
                    st.markdown(f"🧪 Based on **{servings} serving(s)** of _{final_food_label}_")
                else:
                    st.error("❌ Not found in DB.")
        except Exception as e:
            st.error(f"❌ DB error: {e}")