


# import streamlit as st
# import pandas as pd
# from PIL import Image
# from transformers import pipeline
# from sqlalchemy import create_engine, text

# # # 🔒 Supabase DB setup
# password = st.secrets["DB_PASSWORD"]
# db_url = f"postgresql://postgres:{password}@db.hvnmzrrskpgdhyvumbrr.supabase.co:6543/postgres"
# engine = create_engine(db_url)

# # Database connection
# # password = "Summyvinita1!"
# # db_url = f"postgresql://postgres:{password}@db.hvnmzrrskpgdhyvumbrr.supabase.co:5432/postgres"
# # engine = create_engine(db_url)


# # 🔍 Load model (cached)
# @st.cache_resource
# def load_model():
#     return pipeline("image-classification", model="nateraw/food")

# classifier = load_model()

# # === 👤 User Health Planner Section ===
# st.title("🏥 Smart Health + Calorie Planner")

# with st.form("health_form"):
#     st.subheader("👤 Enter Your Info")
#     col1, col2 = st.columns(2)
#     with col1:
#         weight = st.number_input("Current Weight (kg)", min_value=30.0, max_value=300.0)
#         target_weight = st.number_input("Target Weight (kg)", min_value=30.0, max_value=300.0)
#         height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0)
#     with col2:
#         age = st.number_input("Age", min_value=10, max_value=100, value=25)
#         gender = st.selectbox("Gender", ["Male", "Female"])
#         activity = st.selectbox("Activity Level", ["Sedentary", "Light", "Moderate", "Active", "Very Active"])
#         months = st.slider("Goal Duration (months)", 1, 12, 3)

#     submitted = st.form_submit_button("🎯 Calculate My Plan")

# if submitted:
#     # BMR calculation
#     if gender == "Male":
#         bmr = 10 * weight + 6.25 * height - 5 * age + 5
#     else:
#         bmr = 10 * weight + 6.25 * height - 5 * age - 161

#     activity_map = {
#         "Sedentary": 1.2,
#         "Light": 1.375,
#         "Moderate": 1.55,
#         "Active": 1.725,
#         "Very Active": 1.9
#     }

#     tdee = round(bmr * activity_map[activity])
#     diff_kg = round(target_weight - weight, 2)
#     total_kcal_diff = abs(diff_kg * 7700)
#     daily_kcal_adjustment = round(total_kcal_diff / (months * 30))

#     if diff_kg < 0:
#         goal_type = "Weight Loss"
#         target_kcal = tdee - daily_kcal_adjustment
#     elif diff_kg > 0:
#         goal_type = "Weight Gain"
#         target_kcal = tdee + daily_kcal_adjustment
#     else:
#         goal_type = "Maintenance"
#         target_kcal = tdee

#     st.success(f"🧮 BMR: `{round(bmr)}` kcal/day")
#     st.info(f"🔥 TDEE (Maintenance): `{tdee}` kcal/day")
#     st.success(f"🎯 Goal: **{goal_type}** of {abs(diff_kg)} kg in {months} months")
#     st.success(f"📅 Target Calories/Day: `{target_kcal}` kcal")

# # === 🍽️ Image + Nutrition Detection Section ===
# st.header("🍽️ AI-Powered Food Nutrient Detector")
# uploaded_file = st.file_uploader("📷 Upload a food image", type=["jpg", "jpeg", "png"])

# if uploaded_file:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image")

#     with st.spinner("🔍 Identifying food..."):
#         prediction = classifier(image)[0]
#         raw_label = prediction["label"]
#         label = raw_label.replace("_", " ").lower()
#     st.success(f"🤖 Detected: **{raw_label}**")

#     # DB match
#     with engine.connect() as conn:
#         sql = text("SELECT DISTINCT food FROM foods WHERE food ILIKE :pattern ORDER BY food")
#         similar_foods_df = pd.read_sql(sql, conn, params={"pattern": f"%{label}%"})
#         similar_foods = similar_foods_df["food"].tolist()

#     if not similar_foods:
#         similar_foods = [label]

#     st.markdown("🔧 **Refine or confirm food name (type any food to search DB):**")
#     user_input = st.text_input("Type food name:", value=label)

#     food_matches = []
#     if user_input.strip():
#         with engine.connect() as conn:
#             search_sql = text("SELECT DISTINCT food FROM foods WHERE food ILIKE :term ORDER BY food")
#             food_df = pd.read_sql(search_sql, conn, params={"term": f"%{user_input.strip()}%"})
#             food_matches = food_df["food"].tolist()

#     if food_matches:
#         final_food_label = st.selectbox("Select from DB matches:", food_matches, index=0)
#     else:
#         st.warning("⚠️ No match found — using your typed input.")
#         final_food_label = user_input.strip().lower()

#     servings = st.number_input("🍽️ Number of servings:", min_value=0.1, step=0.1, value=1.0)
#     st.caption("ℹ️ Nutrition values are for 1 serving — e.g., 100g or 1 piece.")

#     if st.button("🔎 Get Nutrition Info"):
#         try:
#             with engine.connect() as conn:
#                 query = text("SELECT * FROM foods WHERE LOWER(food) = :label LIMIT 1")
#                 result = pd.read_sql(query, conn, params={"label": final_food_label})

#                 if not result.empty:
#                     st.subheader("📊 Nutrition Info")
#                     for col in result.columns:
#                         if col.lower() != "food":
#                             val = result[col].iloc[0]
#                             scaled = round(float(val) * servings, 2) if pd.notnull(val) else "N/A"
#                             st.write(f"**{col.replace('_', ' ').title()}:** {scaled}")

#                     st.markdown(f"🧪 Based on **{servings} serving(s)** of _{final_food_label}_")

#                     # 🧮 Compare to calorie goal
#                     if submitted and "calories" in result.columns:
#                         consumed = round(float(result["calories"].iloc[0]) * servings, 2)
#                         pct = round((consumed / target_kcal) * 100, 2)
#                         st.info(f"⚖️ This meal contributes ~**{pct}%** to your daily goal of {target_kcal} kcal")
#                 else:
#                     st.error("❌ Food not found in the database.")
#         except Exception as e:
#             st.error(f"❌ Database error: {e}")



###########################################

# import streamlit as st
# import pandas as pd
# from PIL import Image
# from transformers import pipeline
# from sqlalchemy import create_engine, text
# import hashlib

# # === DB Setup ===
# password = st.secrets["DB_PASSWORD"]
# db_url = f"postgresql://postgres:{password}@db.hvnmzrrskpgdhyvumbrr.supabase.co:6543/postgres"
# engine = create_engine(db_url)

# # === Model Setup ===
# @st.cache_resource
# def load_model():
#     return pipeline("image-classification", model="nateraw/food")
# classifier = load_model()

# # === Utility ===
# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# # === User Session ===
# if "user_id" not in st.session_state:
#     st.session_state.user_id = None
#     st.session_state.email = None
#     st.session_state.health = {}

# # === Login / Signup ===
# st.title("\U0001F510 Login or Signup")
# mode = st.radio("Select", ["Login", "Signup"])
# email = st.text_input("Email")
# password_input = st.text_input("Password", type="password")

# if mode == "Signup":
#     name = st.text_input("Name")
#     if st.button("Create Account"):
#         if not email or not password_input or not name:
#             st.warning("Please fill in all fields.")
#         else:
#             try:
#                 with engine.begin() as conn:
#                     result = conn.execute(text("SELECT * FROM users WHERE email = :email"), {"email": email}).fetchone()
#                     if result:
#                         st.error("\u274C Email already registered.")
#                     else:
#                         conn.execute(text("""
#                             INSERT INTO users (email, name, password)
#                             VALUES (:email, :name, :password)
#                         """), {
#                             "email": email,
#                             "name": name,
#                             "password": hash_password(password_input)
#                         })
#                         st.success("\u2705 Account created! You can now log in.")
#                         st.balloons()
#             except Exception as e:
#                 st.error(f"\u274C Signup failed: {e}")

# elif mode == "Login":
#     if st.button("Login"):
#         with engine.connect() as conn:
#             result = conn.execute(text("SELECT * FROM users WHERE email = :email AND password = :password"), {
#                 "email": email,
#                 "password": hash_password(password_input)
#             }).fetchone()
#             if result:
#                 st.session_state.user_id = result.id
#                 st.session_state.email = result.email
#                 with engine.connect() as conn:
#                     profile = conn.execute(text("SELECT * FROM users WHERE id = :id"), {"id": result.id}).fetchone()
#                     if profile:
#                         st.session_state.health = dict(profile._mapping)
#                 st.success(f"\u2705 Logged in as {result.name}")
#             else:
#                 st.error("\u274C Invalid credentials.")

# # === Main App After Login ===
# if st.session_state.user_id:
#     st.header("\U0001F3E5 Health Planner")
#     with st.form("health_form"):
#         weight = st.number_input("Current Weight (kg)", 30.0, 300.0, value=st.session_state.health.get("weight", 70.0))
#         target_weight = st.number_input("Target Weight (kg)", 30.0, 300.0, value=st.session_state.health.get("target_weight", 65.0))
#         height = st.number_input("Height (cm)", 100.0, 250.0, value=st.session_state.health.get("height", 170.0))
#         age = st.number_input("Age", 10, 100, value=st.session_state.health.get("age", 25))
#         gender = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.health.get("gender", "Male") == "Male" else 1)

#         activity_levels = ["Sedentary", "Light", "Moderate", "Active", "Very Active"]
#         default_activity = st.session_state.health.get("activity_level", "Moderate")
#         if default_activity not in activity_levels:
#             default_activity = "Moderate"
#         activity = st.selectbox("Activity", activity_levels, index=activity_levels.index(default_activity))

#         months = st.slider("Goal Duration (months)", 1, 12, st.session_state.health.get("months", 3))
#         save = st.form_submit_button("\U0001F4BE Save Plan")

#     if save:
#         bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == "Male" else -161)
#         activity_map = {"Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55, "Active": 1.725, "Very Active": 1.9}
#         tdee = round(bmr * activity_map[activity])
#         diff = target_weight - weight
#         total_kcal = abs(diff * 7700)
#         kcal_day = round(total_kcal / (months * 30))
#         goal = tdee + kcal_day if diff > 0 else tdee - kcal_day

#         with engine.begin() as conn:
#             conn.execute(text("""
#                 UPDATE users SET 
#                     age = :age, gender = :gender, height = :height, weight = :weight,
#                     target_weight = :target_weight, activity_level = :activity,
#                     months = :months, bmr = :bmr, tdee = :tdee, target_calories = :goal
#                 WHERE id = :user_id
#             """), {
#                 "user_id": st.session_state.user_id,
#                 "age": age, "gender": gender, "height": height, "weight": weight,
#                 "target_weight": target_weight, "activity": activity, "months": months,
#                 "bmr": bmr, "tdee": tdee, "goal": goal
#             })
#         st.success(f"\U0001F3AF Your daily calorie target is {goal} kcal")

#     st.header("\U0001F4F7 Food Upload")
#     file = st.file_uploader("Upload Food Image", type=["jpg", "png"])

#     if file:
#         img = Image.open(file)
#         st.image(img, caption="Uploaded Image")
#         with st.spinner("Detecting food..."):
#             prediction = classifier(img)[0]
#             label = prediction["label"].replace("_", " ").lower()
#             st.success(f"Detected: {label}")

#         servings = st.number_input("Servings", 0.1, 10.0, 1.0)
#         with engine.connect() as conn:
#             df = pd.read_sql(text("SELECT * FROM foods WHERE food ILIKE :label"), conn, params={"label": f"%{label}%"})

#         if not df.empty:
#             row = df.iloc[0]
#             st.subheader("Nutritional Info")
#             for col in df.columns:
#                 if col != "food":
#                     val = round(float(row[col]) * servings, 2) if pd.notnull(row[col]) else "N/A"
#                     st.write(f"{col.title()}: {val}")

#             total_calories = round(float(row["caloric_value"]) * servings, 2)
#             with engine.connect() as conn:
#                 goal_cal = conn.execute(text("SELECT target_calories FROM users WHERE id = :id"), {"id": st.session_state.user_id}).scalar()
#                 pct = round((total_calories / goal_cal) * 100, 2) if goal_cal else 0
#                 st.info(f"This meal contributes ~{pct}% of your daily goal.")

#                 conn.execute(text("""
#                     INSERT INTO meal_logs (user_id, food, calories, servings, total_calories, goal_calories, intake_pct)
#                     VALUES (:uid, :food, :cals, :serv, :total, :goal, :pct)
#                 """), {
#                     "uid": int(st.session_state.user_id),
#                     "food": str(row["food"]),
#                     "cals": float(row["caloric_value"]),
#                     "serv": float(servings),
#                     "total": float(total_calories),
#                     "goal": float(goal_cal) if goal_cal is not None else None,
#                     "pct": float(pct)
#                 })

#     with engine.connect() as conn:
#         log_df = pd.read_sql(text("SELECT * FROM meal_logs WHERE user_id = :uid ORDER BY created_at DESC"), conn, params={"uid": st.session_state.user_id})
#     if not log_df.empty:
#         st.header("\U0001F4DC Food Log")
#         st.dataframe(log_df[["created_at", "food", "total_calories", "intake_pct"]])



#########################





import streamlit as st
import pandas as pd
from PIL import Image
from transformers import pipeline
from sqlalchemy import create_engine, text
import hashlib
import matplotlib.pyplot as plt

# === DB Setup ===
password = st.secrets["DB_PASSWORD"]
db_url = f"postgresql://postgres:{password}@db.hvnmzrrskpgdhyvumbrr.supabase.co:6543/postgres"
engine = create_engine(db_url)

# === Model Setup ===
@st.cache_resource
def load_model():
    return pipeline("image-classification", model="nateraw/food")
classifier = load_model()

# === Utility ===
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# === User Session ===
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.email = None
    st.session_state.health = {}

# === Login / Signup ===
TEST_MODE = False

if TEST_MODE:
    st.session_state.user_id = 1
    st.session_state.email = "animeshdubey2612@gmail.com"
else:   
    st.title("\U0001F510 Login or Signup")
    mode = st.radio("Select", ["Login", "Signup"])
    email = st.text_input("Email")
    password_input = st.text_input("Password", type="password")

    if mode == "Signup":
        name = st.text_input("Name")
        if st.button("Create Account"):
            if not email or not password_input or not name:
                st.warning("Please fill in all fields.")
            else:
                try:
                    with engine.begin() as conn:
                        result = conn.execute(text("SELECT * FROM users WHERE email = :email"), {"email": email}).fetchone()
                        if result:
                            st.error("\u274C Email already registered.")
                        else:
                            conn.execute(text("""
                                INSERT INTO users (email, name, password)
                                VALUES (:email, :name, :password)
                            """), {
                                "email": email,
                                "name": name,
                                "password": hash_password(password_input)
                            })
                            st.success("\u2705 Account created! You can now log in.")
                            st.balloons()
                except Exception as e:
                    st.error(f"\u274C Signup failed: {e}")

    elif mode == "Login":
        if st.button("Login"):
            with engine.connect() as conn:
                result = conn.execute(text("SELECT * FROM users WHERE email = :email AND password = :password"), {
                    "email": email,
                    "password": hash_password(password_input)
                }).fetchone()
                if result:
                    st.session_state.user_id = result.id
                    st.session_state.email = result.email
                    with engine.connect() as conn:
                        profile = conn.execute(text("SELECT * FROM users WHERE id = :id"), {"id": result.id}).fetchone()
                        if profile:
                            st.session_state.health = dict(profile._mapping)
                    st.success(f"\u2705 Logged in as {result.name}")
                else:
                    st.error("\u274C Invalid credentials.")

# === Main App After Login ===
if st.session_state.user_id:
    st.header("🏥 Health Planner")
    with st.form("health_form"):
        st.markdown("### 📝 Personal Goals Form")

        col1, col2 = st.columns(2)

        with col1:
            weight = st.number_input("💪 Current Weight (kg)", 30.0, 300.0, value=st.session_state.health.get("weight", 70.0))
            height = st.number_input("📏 Height (cm)", 100.0, 250.0, value=st.session_state.health.get("height", 170.0))
            age = st.number_input("🎂 Age", 10, 100, value=st.session_state.health.get("age", 25))

        with col2:
            target_weight = st.number_input("🎯 Target Weight (kg)", 30.0, 300.0, value=st.session_state.health.get("target_weight", 65.0))
            gender = st.selectbox("🚻 Gender", ["Male", "Female"], index=0 if st.session_state.health.get("gender", "Male") == "Male" else 1)

            activity_levels = ["Sedentary", "Light", "Moderate", "Active", "Very Active"]
            default_activity = st.session_state.health.get("activity_level", "Moderate")
            if default_activity not in activity_levels:
                default_activity = "Moderate"
            activity = st.selectbox("⚡ Activity Level", activity_levels, index=activity_levels.index(default_activity))

        st.slider("📆 Goal Duration (months)", 1, 12, st.session_state.health.get("months", 3), key="months_slider")

        save = st.form_submit_button("💾 Save Plan")

    if save:
        bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == "Male" else -161)
        activity_map = {"Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55, "Active": 1.725, "Very Active": 1.9}
        tdee = round(bmr * activity_map[activity])
        diff = target_weight - weight
        total_kcal = abs(diff * 7700)
        kcal_day = round(total_kcal / (months * 30))
        goal = tdee + kcal_day if diff > 0 else tdee - kcal_day

        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE users SET 
                    age = :age, gender = :gender, height = :height, weight = :weight,
                    target_weight = :target_weight, activity_level = :activity,
                    months = :months, bmr = :bmr, tdee = :tdee, target_calories = :goal
                WHERE id = :user_id
            """), {
                "user_id": st.session_state.user_id,
                "age": age, "gender": gender, "height": height, "weight": weight,
                "target_weight": target_weight, "activity": activity, "months": months,
                "bmr": bmr, "tdee": tdee, "goal": goal
            })
        st.success(f"\U0001F3AF Your daily calorie target is {goal} kcal")

    st.header("\U0001F4F7 Food Upload")
    file = st.file_uploader("Upload Food Image", type=["jpg", "png"])

    if file:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image")
        with st.spinner("Detecting food..."):
            prediction = classifier(img)[0]
            label = prediction["label"].replace("_", " ").lower()
            st.success(f"Detected: {label}")

        # --- Refine prediction ---
        with engine.connect() as conn:
            df_suggest = pd.read_sql(text("SELECT DISTINCT food FROM foods WHERE food ILIKE :pattern ORDER BY food"), conn, params={"pattern": f"%{label}%"})
        suggestions = df_suggest["food"].tolist()
        if not suggestions:
            suggestions = [label]

        st.markdown("🔧 **Refine or confirm food name:**")
        user_input = st.text_input("Type or refine food name:", value=label)

        matches = []
        if user_input.strip():
            with engine.connect() as conn:
                df_match = pd.read_sql(text("SELECT DISTINCT food FROM foods WHERE food ILIKE :term ORDER BY food"), conn, params={"term": f"%{user_input.strip()}%"})
                matches = df_match["food"].tolist()

        if matches:
            final_food = st.selectbox("if you think food is not matched select its option :", matches, index=0)
        else:
            final_food = user_input.strip().lower()
            st.warning("⚠️ No match found — using typed value.")

        servings = st.number_input("Servings", 0.1, 10.0, 1.0)

        with engine.connect() as conn:
            df = pd.read_sql(text("SELECT * FROM foods WHERE LOWER(food) = :label LIMIT 1"), conn, params={"label": final_food.lower()})

        if not df.empty:
            row = df.iloc[0]
            st.subheader("Nutritional Info")
            st.caption("ℹ️ 1 serving = standard portion size, typically 100 grams or 1 unit (e.g., 1 egg, 1 slice, or 1 bowl). Adjust based on what you consumed.")
            


        # 🎯 Extract macronutrients from `row` (after fetching food data)
            # 🎯 Extract macronutrients from `row` (after fetching food data)
            carbs = float(row.get("carbohydrates", 0) or 0)
            protein = float(row.get("protein", 0) or 0)
            fat = float(row.get("fat", 0) or 0)
            sugar = float(row.get("free_sugar_g", 0) or 0)
            fibre = float(row.get("fibre_g", 0) or 0)

            # 🔍 Filter and prepare data
            labels = ["Carbohydrates", "Protein", "Fat", "Free Sugar", "Fibre"]
            values = [carbs, protein, fat, sugar, fibre]
            filtered = [(l, v) for l, v in zip(labels, values) if v > 0]
            labels, values = zip(*filtered) if filtered else ([], [])

            # 🥯 Donut Chart with colors
            if values:
                fig, ax = plt.subplots()
                colors = ["#FFD700", "#90EE90", "#FF6347", "#87CEFA", "#DA70D6"]

                wedges, texts, autotexts = ax.pie(
                    values,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=140,
                    colors=colors,
                    wedgeprops=dict(width=0.4),  # 🍩 Donut effect
                    textprops={"color": "black", "fontsize": 12}
                )

                ax.axis("equal")
                ax.set_title("Macronutrient Breakdown", fontsize=14)
                plt.legend(wedges, labels, title="Nutrients", loc="center left", bbox_to_anchor=(1, 0.5))

                st.pyplot(fig)
            else:
                st.info("ℹ️ Macronutrient values not available to plot.")
                        
            
            
            for col in df.columns:
                if col != "food":
                    val = round(float(row[col]) * servings, 2) if pd.notnull(row[col]) else "N/A"
                    st.write(f"{col.title()}: {val}")

            total_calories = round(float(row["caloric_value"]) * servings, 2)
            with engine.connect() as conn:
                goal_cal = conn.execute(text("SELECT target_calories FROM users WHERE id = :id"), {"id": st.session_state.user_id}).scalar()
                pct = round((total_calories / goal_cal) * 100, 2) if goal_cal else 0
                st.info(f"This meal contributes ~{pct}% of your daily goal.")

                conn.execute(text("""
                    INSERT INTO meal_logs (user_id, food, calories, servings, total_calories, goal_calories, intake_pct)
                    VALUES (:uid, :food, :cals, :serv, :total, :goal, :pct)
                """), {
                    "uid": int(st.session_state.user_id),
                    "food": str(row["food"]),
                    "cals": float(row["caloric_value"]),
                    "serv": float(servings),
                    "total": float(total_calories),
                    "goal": float(goal_cal) if goal_cal is not None else None,
                    "pct": float(pct)
                })

    with engine.connect() as conn:
        log_df = pd.read_sql(text("SELECT * FROM meal_logs WHERE user_id = :uid ORDER BY created_at DESC"), conn, params={"uid": st.session_state.user_id})
    if not log_df.empty:
        st.header("\U0001F4DC Food Log")
        st.dataframe(log_df[["created_at", "food", "total_calories", "intake_pct"]])
