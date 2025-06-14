







import streamlit as st
import pandas as pd
from PIL import Image
from transformers import pipeline
from sqlalchemy import create_engine, text
import hashlib
import numpy as np
import cv2
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# === DB Setup ===
password = st.secrets["DB_PASSWORD"]
db_url = f"postgresql://postgres:{password}@db.hvnmzrrskpgdhyvumbrr.supabase.co:6543/postgres"
engine = create_engine(db_url)

# === Enhanced Multi-Modal Food Classifier ===
class MultiModalFoodClassifier:
    def __init__(self):
        self.primary_classifier = pipeline("image-classification", model="nateraw/food")
        self.backup_classifier = pipeline("image-classification", model="Kaludi/food-category-classification-v2.0")
        
    def classify_food_ensemble(self, image):
        try:
            primary_pred = self.primary_classifier(image)
            primary_result = primary_pred[0] if primary_pred else None
            
            backup_pred = self.backup_classifier(image)
            backup_result = backup_pred[0] if backup_pred else None
            
            if primary_result and backup_result:
                combined_confidence = (primary_result['score'] * 0.7 + backup_result['score'] * 0.3)
                best_label = primary_result['label'] if primary_result['score'] > backup_result['score'] else backup_result['label']
            elif primary_result:
                combined_confidence = primary_result['score']
                best_label = primary_result['label']
            elif backup_result:
                combined_confidence = backup_result['score']
                best_label = backup_result['label']
            else:
                return None
                
            return {
                'label': best_label.replace("_", " ").lower(),
                'confidence': combined_confidence,
                'primary_prediction': primary_result,
                'backup_prediction': backup_result
            }
        except Exception as e:
            st.error(f"Classification error: {e}")
            return None

# === Recommendation Engine ===
class PersonalizedRecommendationEngine:
    def __init__(self, engine):
        self.engine = engine
        
    def get_content_based_recommendations(self, user_id, n_recommendations=5):
        try:
            with self.engine.connect() as conn:
                history_query = """
                SELECT COUNT(*) as meal_count
                FROM meal_logs 
                WHERE user_id = :user_id
                """
                history_result = conn.execute(text(history_query), {'user_id': user_id}).fetchone()
                
                if not history_result or history_result.meal_count == 0:
                    return [
                        {'food': 'apple', 'reason': 'Healthy choice - rich in fiber'},
                        {'food': 'banana', 'reason': 'Good source of potassium'},
                        {'food': 'chicken breast', 'reason': 'High protein, low fat'},
                        {'food': 'broccoli', 'reason': 'Rich in vitamins and minerals'},
                        {'food': 'salmon', 'reason': 'Omega-3 fatty acids'}
                    ]
                
                user_query = """
                SELECT food, AVG(intake_pct) as satisfaction, COUNT(*) as frequency
                FROM meal_logs 
                WHERE user_id = :user_id AND intake_pct > 0
                GROUP BY food
                ORDER BY satisfaction DESC, frequency DESC
                LIMIT 3
                """
                user_foods_result = conn.execute(text(user_query), {'user_id': user_id})
                user_foods = user_foods_result.fetchall()
                
                recommendations = []
                for row in user_foods:
                    food_name = row.food
                    
                    similarity_query = """
                    SELECT f2.food, 
                           ABS(COALESCE(f1.caloric_value, 0) - COALESCE(f2.caloric_value, 0)) + 
                           ABS(COALESCE(f1.protein, 0) - COALESCE(f2.protein, 0)) + 
                           ABS(COALESCE(f1.carbohydrates, 0) - COALESCE(f2.carbohydrates, 0)) as similarity_score
                    FROM foods f1, foods f2
                    WHERE LOWER(f1.food) = LOWER(:food_name)
                    AND f1.food != f2.food
                    AND f2.caloric_value IS NOT NULL
                    ORDER BY similarity_score ASC
                    LIMIT 2
                    """
                    similar_result = conn.execute(text(similarity_query), {'food_name': food_name})
                    similar_foods = similar_result.fetchall()
                    
                    for similar_row in similar_foods:
                        recommendations.append({
                            'food': similar_row.food,
                            'reason': f'Similar to {food_name} (you liked this before)'
                        })
                
                seen = set()
                unique_recommendations = []
                for rec in recommendations:
                    if rec['food'] not in seen and len(unique_recommendations) < n_recommendations:
                        seen.add(rec['food'])
                        unique_recommendations.append(rec)
                
                if len(unique_recommendations) < n_recommendations:
                    defaults = [
                        {'food': 'greek yogurt', 'reason': 'High protein snack'},
                        {'food': 'oatmeal', 'reason': 'Fiber-rich breakfast'},
                        {'food': 'spinach', 'reason': 'Iron and vitamin rich'}
                    ]
                    for default in defaults:
                        if len(unique_recommendations) < n_recommendations and default['food'] not in seen:
                            unique_recommendations.append(default)
                            seen.add(default['food'])
                
                return unique_recommendations
                
        except Exception as e:
            st.error(f"Error getting recommendations: {e}")
            return [
                {'food': 'apple', 'reason': 'Healthy default choice'},
                {'food': 'chicken breast', 'reason': 'High protein default'}
            ]

def create_enhanced_calorie_chart(log_df):
    """Create enhanced calorie visualization with bar chart only"""
    if log_df.empty:
        return None
    
    # Process data for better visualization
    log_df['created_at'] = pd.to_datetime(log_df['created_at'])
    log_df['date'] = log_df['created_at'].dt.date
    
    # Daily aggregation
    daily_calories = log_df.groupby('date').agg({
        'total_calories': 'sum',
        'food': 'count'
    }).reset_index()
    daily_calories.columns = ['Date', 'Total Calories', 'Meals Count']
    
    # Create bar chart only
    fig = px.bar(daily_calories, x='Date', y='Total Calories',
                title='📊 Daily Calorie Intake',
                hover_data=['Meals Count'],
                color='Total Calories',
                color_continuous_scale='viridis')
    
    # Add goal line if available
    if 'goal_calories' in log_df.columns and log_df['goal_calories'].iloc[0] > 0:
        goal_cal = log_df['goal_calories'].iloc[0]
        fig.add_hline(y=goal_cal, line_dash="dash", 
                     annotation_text=f"Daily Goal: {goal_cal} cal",
                     line_color="red")
    
    # Enhanced styling
    fig.update_layout(
        height=500,
        showlegend=True,
        template="plotly_white",
        font=dict(size=12),
        title_font_size=16
    )
    
    return fig

def create_nutrition_summary_chart(log_df):
    """Create comprehensive nutrition summary with pie chart and gauge"""
    if log_df.empty:
        return None
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Recent Meals Calorie Distribution pie chart
        if len(log_df) > 1:
            recent_meals = log_df.head(10)  # Get last 10 meals
            fig_pie = px.pie(recent_meals, values='total_calories', names='food',
                           title='🥧 Recent Meals Calorie Distribution')
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Weekly progress gauge
        today = pd.Timestamp.now().date()
        week_start = today - timedelta(days=7)
        
        weekly_data = log_df[pd.to_datetime(log_df['created_at']).dt.date >= week_start]
        weekly_total = weekly_data['total_calories'].sum()
        weekly_goal = 14000  # Assuming 2000 cal/day * 7 days
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = weekly_total,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Weekly Calories"},
            delta = {'reference': weekly_goal},
            gauge = {
                'axis': {'range': [None, weekly_goal * 1.2]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, weekly_goal * 0.8], 'color': "lightgray"},
                    {'range': [weekly_goal * 0.8, weekly_goal], 'color': "gray"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': weekly_goal}}))
        
        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)



# === Navigation Functions ===
def create_navigation_buttons(current_page):
    """Create navigation buttons with proper session state handling"""
    st.markdown("---")
    
    page_flow = ["Health Planner", "Food Recognition", "Recommendations", "Food Log"]
    current_index = page_flow.index(current_page) if current_page in page_flow else 0
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if current_index > 0:
            prev_key = f"prev_btn_{current_page}_{current_index}"
            if st.button("⬅️ Previous", key=prev_key, help=f"Go to {page_flow[current_index-1]}"):
                st.session_state.current_page = page_flow[current_index-1]
                st.rerun()
    
    with col2:
        progress = (current_index + 1) / len(page_flow)
        st.progress(progress, text=f"Step {current_index + 1} of {len(page_flow)}: {current_page}")
    
    with col3:
        if current_index < len(page_flow) - 1:
            next_key = f"next_btn_{current_page}_{current_index}"
            if st.button("Next ➡️", key=next_key, help=f"Go to {page_flow[current_index+1]}"):
                st.session_state.current_page = page_flow[current_index+1]
                st.rerun()

def create_quick_navigation():
    """Create quick navigation with unique keys"""
    st.markdown("### 🚀 Quick Navigation")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🏥 Health Setup", use_container_width=True, key="nav_health_unique", 
                    help="Set your health goals and preferences"):
            st.session_state.current_page = "Health Planner"
            st.rerun()
    
    with col2:
        if st.button("📷 Scan Food", use_container_width=True, key="nav_scan_unique",
                    help="Upload food images to track calories"):
            st.session_state.current_page = "Food Recognition"
            st.rerun()
    
    with col3:
        if st.button("🎯 Get Suggestions", use_container_width=True, key="nav_suggestions_unique",
                    help="Get personalized food recommendations"):
            st.session_state.current_page = "Recommendations"
            st.rerun()
    
    with col4:
        if st.button("📊 View Progress", use_container_width=True, key="nav_progress_unique",
                    help="Check your meal history and progress"):
            st.session_state.current_page = "Food Log"
            st.rerun()

def show_comprehensive_user_guide():
    """Show comprehensive user guide with detailed instructions"""
    if "show_guide" not in st.session_state:
        st.session_state.show_guide = True
    
    if st.session_state.show_guide:
        with st.expander("📚 Complete User Guide - How to Use This App", expanded=True):
            st.markdown("""
            # 🍽️ Welcome to Your AI-Powered Calorie Tracker!
            
            This comprehensive guide will help you master every feature of your nutrition tracking app.
            
            ---
            
            ## 🏥 **Step 1: Health Planner**
            
            **What it does:** Sets up your personalized health profile and calculates your daily calorie target.
            
            **How to use:**
            1. **Enter your basic info:** Current weight, height, age, and gender
            2. **Set your goal:** Target weight you want to achieve
            3. **Choose activity level:** From sedentary to very active
            4. **Set timeline:** How many months to reach your goal
            5. **Save your plan:** Click "Save Plan & Continue"
            
            **💡 Pro Tips:**
            - Be honest about your activity level for accurate calorie targets
            - Set realistic goals (1-2 lbs per week is healthy)
            - Your BMR and TDEE will be automatically calculated
            
            ---
            
            ## 📷 **Step 2: Food Recognition**
            
            **What it does:** Uses AI to identify food from photos and calculate nutritional information.
            
            **How to use:**
            1. **Upload a photo:** Take a clear picture of your meal
            2. **AI analysis:** Wait for the system to identify the food
            3. **Refine if needed:** Correct the food name if the AI got it wrong
            4. **Select from database:** Choose the exact food from our nutrition database
            5. **Set servings:** Adjust the portion size (AI provides an estimate)
            6. **Review nutrition:** Check calories, protein, carbs, and fats
            7. **Log the meal:** Click "Log This Meal" to save it
            
            **💡 Pro Tips:**
            - Take photos in good lighting for better AI recognition
            - Include reference objects (like plates) for better portion estimation
            - Double-check the AI's food identification before logging
            - Use the search function if your exact food isn't found
            
            ---
            
            ## 🎯 **Step 3: Personalized Recommendations**
            
            **What it does:** Suggests healthy foods based on your eating history and nutritional goals.
            
            **How to use:**
            1. **Generate recommendations:** Click "Get New Recommendations"
            2. **Review suggestions:** See why each food is recommended
            3. **Check nutrition info:** View calories and macronutrients
            4. **Quick add:** Use recommendations for meal planning
            
            **💡 Pro Tips:**
            - Recommendations improve as you log more meals
            - Try new foods suggested by the AI
            - Use recommendations for grocery shopping lists
            
            ---
            
            ## 📊 **Step 4: Progress Tracking**
            
            **What it does:** Visualizes your nutrition journey with charts and analytics.
            
            **Features available:**
            - **Daily metrics:** Today's calories, meal count, goal progress
            - **Interactive charts:** Choose from line, bar, area, or timeline views
            - **Nutrition analysis:** Pie charts showing meal distribution
            - **Weekly progress:** Gauge showing weekly calorie goals
            - **Meal history:** Detailed log of all your meals
            
            **💡 Pro Tips:**
            - Check your progress daily for motivation
            - Use different chart types to spot patterns
            - Look for trends in your eating habits
            - Adjust your goals based on progress
            
            ---
            
            ## 🎮 **Navigation Tips**
            
            **Multiple ways to navigate:**
            - **Quick Navigation buttons:** Jump to any section instantly
            - **Previous/Next arrows:** Follow the guided workflow
            - **Progress bar:** See which step you're on
            - **Action buttons:** Context-specific navigation (e.g., "Log Another Meal")
            
            ---
            
            ## 🔧 **Troubleshooting**
            
            **Food not recognized?**
            - Try a different angle or better lighting
            - Manually type the food name
            - Use the search function
            
            **Nutrition info missing?**
            - Check spelling of food name
            - Try a more generic name (e.g., "chicken" instead of "grilled chicken breast")
            
            **Charts not showing?**
            - Log at least 2-3 meals first
            - Check that meals were saved successfully
            
            ---
            
            ## 🏆 **Best Practices**
            
            1. **Log meals immediately** after eating for accuracy
            2. **Be consistent** - log every meal and snack
            3. **Take clear photos** for better AI recognition
            4. **Review recommendations** regularly for new food ideas
            5. **Check progress weekly** to stay motivated
            6. **Adjust goals** as needed based on results
            
            ---
            
            ## 📱 **Mobile Tips**
            
            - Use your phone's camera for meal photos
            - The app works great on mobile browsers
            - Bookmark the app for quick access
            - Take photos before you start eating
            
            ---
            
            **Ready to start your nutrition journey? Begin with Step 1: Health Planner!**
            """)
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Got it! Let's Start! ✅", key="close_guide_btn"):
                    st.session_state.show_guide = False
                    st.rerun()

# === Utility Functions ===
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_enhanced_nutrition_chart(row, servings):
    carbs = float(row.get("carbohydrates", 0) or 0) * servings
    protein = float(row.get("protein", 0) or 0) * servings
    fat = float(row.get("fat", 0) or 0) * servings
    
    if carbs + protein + fat > 0:
        labels = ["Carbohydrates", "Protein", "Fat"]
        values = [carbs, protein, fat]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values,
            hole=0.4,
            textinfo='label+percent',
            marker=dict(colors=colors)
        )])
        
        fig.update_layout(
            title="Macronutrient Breakdown", 
            height=400,
            showlegend=True,
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

# === Initialize Models ===
@st.cache_resource
def load_enhanced_models():
    classifier = MultiModalFoodClassifier()
    return classifier

@st.cache_resource
def load_recommendation_engine():
    return PersonalizedRecommendationEngine(engine)

# === User Session ===
if "user_id" not in st.session_state:
    st.session_state.user_id = None
    st.session_state.email = None
    st.session_state.health = {}

if "current_page" not in st.session_state:
    st.session_state.current_page = "Health Planner"

# === Login / Signup ===
if not st.session_state.user_id:
    st.title("🍽️ AI Calorie Tracker")
    st.markdown("### Welcome! Please login or create an account to start tracking your nutrition.")
    
    mode = st.radio("Select", ["Login", "Signup"])
    email = st.text_input("Email")
    password_input = st.text_input("Password", type="password")

    if mode == "Signup":
        name = st.text_input("Name")
        if st.button("Create Account", type="primary", key="signup_btn_main"):
            if not email or not password_input or not name:
                st.warning("Please fill in all fields.")
            else:
                try:
                    with engine.begin() as conn:
                        result = conn.execute(text("SELECT * FROM users WHERE email = :email"), {"email": email}).fetchone()
                        if result:
                            st.error("❌ Email already registered.")
                        else:
                            conn.execute(text("""
                                INSERT INTO users (email, name, password)
                                VALUES (:email, :name, :password)
                            """), {
                                "email": email,
                                "name": name,
                                "password": hash_password(password_input)
                            })
                            st.success("✅ Account created! You can now log in.")
                            st.balloons()
                except Exception as e:
                    st.error(f"❌ Signup failed: {e}")

    elif mode == "Login":
        if st.button("Login", type="primary", key="login_btn_main"):
            try:
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT * FROM users WHERE email = :email AND password = :password"), {
                        "email": email,
                        "password": hash_password(password_input)
                    }).fetchone()
                    
                    if result:
                        st.session_state.user_id = result.id
                        st.session_state.email = result.email
                        st.session_state.health = dict(result._mapping)
                        st.success(f"✅ Logged in as {result.name}")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials.")
            except Exception as e:
                st.error(f"❌ Login failed: {e}")

# === Main App After Login ===
if st.session_state.user_id:
    classifier = load_enhanced_models()
    rec_engine = load_recommendation_engine()
    
    # App header
    st.title("🍽️ AI Calorie Tracker")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**Welcome back, {st.session_state.email}!** 👋")
    with col2:
        if st.button("🚪 Logout", key="logout_btn_main"):
            st.session_state.user_id = None
            st.session_state.email = None
            st.session_state.health = {}
            st.session_state.current_page = "Health Planner"
            st.rerun()
    
    # Show comprehensive user guide
    show_comprehensive_user_guide()
    
    # Quick navigation
    create_quick_navigation()
    st.markdown("---")
    
    current_page = st.session_state.current_page
    
    # === HEALTH PLANNER PAGE ===
    if current_page == "Health Planner":
        st.header("🏥 Step 1: Health Planner")
        st.markdown("*Set up your personal health goals and get your daily calorie target*")
        
        with st.form("health_form_main"):
            st.markdown("### 📝 Personal Goals Form")

            col1, col2 = st.columns(2)

            with col1:
                weight = st.number_input("💪 Current Weight (kg)", 30.0, 300.0, 
                                       value=float(st.session_state.health.get("weight") or 70.0))
                height = st.number_input("📏 Height (cm)", 100.0, 250.0, 
                                       value=float(st.session_state.health.get("height") or 170.0))
                age = st.number_input("🎂 Age", 10, 100, 
                                    value=int(st.session_state.health.get("age") or 25))  # ✅ Fix here

            with col2:
                target_weight = st.number_input("🎯 Target Weight (kg)", 30.0, 300.0, 
                                              value=float(st.session_state.health.get("target_weight") or 65.0))
                gender = st.selectbox("🚻 Gender", ["Male", "Female"], 
                                    index=0 if st.session_state.health.get("gender", "Male") == "Male" else 1)

                activity_levels = ["Sedentary", "Light", "Moderate", "Active", "Very Active"]
                default_activity = st.session_state.health.get("activity_level") or "Moderate"
                if default_activity not in activity_levels:
                    default_activity = "Moderate"
                activity = st.selectbox("⚡ Activity Level", activity_levels, 
                                      index=activity_levels.index(default_activity))

            months = st.slider("📆 Goal Duration (months)", 1, 12, 
                             int(st.session_state.health.get("months") or 3))
            save = st.form_submit_button("💾 Save Plan & Continue", type="primary")

        if save:
            bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == "Male" else -161)
            activity_map = {"Sedentary": 1.2, "Light": 1.375, "Moderate": 1.55, "Active": 1.725, "Very Active": 1.9}
            tdee = round(bmr * activity_map[activity])
            diff = target_weight - weight
            total_kcal = abs(diff * 7700)
            kcal_day = round(total_kcal / (months * 30))
            goal = tdee + kcal_day if diff > 0 else tdee - kcal_day

            try:
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
                
                # Update session state
                st.session_state.health.update({
                    "target_calories": goal, "tdee": tdee, "bmr": bmr,
                    "weight": weight, "height": height, "age": age,
                    "target_weight": target_weight, "activity_level": activity, "months": months
                })
                
                st.success(f"🎯 Your daily calorie target is {goal} kcal")
                
                # Show detailed breakdown
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("BMR (Base Metabolic Rate)", f"{bmr:.0f} cal")
                with col2:
                    st.metric("TDEE (Total Daily Energy)", f"{tdee:.0f} cal")
                with col3:
                    st.metric("Daily Goal", f"{goal:.0f} cal")
                
                st.info("✅ Health plan saved! Ready to start tracking food?")
                if st.button("📷 Start Food Tracking ➡️", type="primary", key="start_food_tracking_btn"):
                    st.session_state.current_page = "Food Recognition"
                    st.rerun()
            except Exception as e:
                st.error(f"Error saving health plan: {e}")
        
        # Show current health info if available
        if st.session_state.health.get('target_calories'):
            st.markdown("### 📊 Your Current Health Plan")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Weight", f"{st.session_state.health.get('weight', 'N/A')} kg")
            with col2:
                st.metric("Target Weight", f"{st.session_state.health.get('target_weight', 'N/A')} kg")
            with col3:
                st.metric("Daily Calorie Goal", f"{st.session_state.health.get('target_calories', 'N/A')} cal")
            with col4:
                st.metric("Timeline", f"{st.session_state.health.get('months', 'N/A')} months")
        
        create_navigation_buttons(current_page)
    
    # === FOOD RECOGNITION PAGE ===
    elif current_page == "Food Recognition":
        st.header("📷 Step 2: Food Recognition")
        st.markdown("*Upload photos of your meals and let AI identify the food and calories*")
        
        file = st.file_uploader("📸 Upload Food Image", type=["jpg", "png", "jpeg"])

        if file:
            img = Image.open(file)
            st.image(img, caption="Uploaded Image", width=400)
            
            with st.spinner("🔍 Analyzing food with AI..."):
                prediction = classifier.classify_food_ensemble(img)
                
                if prediction:
                    st.success(f"🍽️ Detected: **{prediction['label'].title()}**")
                    st.info(f"🎯 Confidence: {prediction['confidence']:.1%}")
                    
                    # Show model breakdown
                    with st.expander("🔬 AI Model Details"):
                        if prediction['primary_prediction']:
                            st.write("**Primary Model:**", prediction['primary_prediction']['label'], 
                                    f"({prediction['primary_prediction']['score']:.1%})")
                        if prediction['backup_prediction']:
                            st.write("**Backup Model:**", prediction['backup_prediction']['label'], 
                                    f"({prediction['backup_prediction']['score']:.1%})")
                    
                    label = prediction['label']
                    
                    # Food refinement section
                    st.markdown("### 🔧 Refine Food Selection")
                    user_input = st.text_input("Type or refine food name:", value=label)
                    
                    try:
                        with engine.connect() as conn:
                            df_match = pd.read_sql(text("SELECT DISTINCT food FROM foods WHERE food ILIKE :term ORDER BY food LIMIT 20"), 
                                                 conn, params={"term": f"%{user_input.strip()}%"})
                            matches = df_match["food"].tolist()
                    except Exception as e:
                        st.error(f"Error searching foods: {e}")
                        matches = []

                    if matches:
                        final_food = st.selectbox("Select the correct food:", matches, index=0)
                    else:
                        final_food = user_input.strip().lower()
                        st.warning("⚠️ No match found in database — using typed value.")

                    servings = st.number_input("🍽️ Servings", 0.1, 10.0, 1.0, 
                                             help="1 serving = typical portion size (usually 100g)")

                    try:
                        with engine.connect() as conn:
                            df = pd.read_sql(text("SELECT * FROM foods WHERE LOWER(food) = LOWER(:label) LIMIT 1"), 
                                           conn, params={"label": final_food})

                        if not df.empty:
                            row = df.iloc[0]
                            st.subheader("🥗 Nutritional Information")
                            
                            create_enhanced_nutrition_chart(row, servings)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                calories = round(float(row['caloric_value']) * servings, 1)
                                st.metric("Calories", f"{calories}")
                            with col2:
                                protein = round(float(row.get('protein', 0) or 0) * servings, 1)
                                st.metric("Protein", f"{protein}g")
                            with col3:
                                carbs = round(float(row.get('carbohydrates', 0) or 0) * servings, 1)
                                st.metric("Carbs", f"{carbs}g")

                            if st.button("📝 Log This Meal", type="primary", key="log_meal_btn"):
                                try:
                                    total_calories = round(float(row["caloric_value"]) * servings, 2)
                                    
                                    with engine.connect() as conn:
                                        goal_result = conn.execute(text("SELECT target_calories FROM users WHERE id = :id"), 
                                                              {"id": st.session_state.user_id}).fetchone()
                                        goal_cal = goal_result.target_calories if goal_result and goal_result.target_calories else 2000
                                        pct = round((total_calories / goal_cal) * 100, 2)
                                        
                                        conn.execute(text("""
                                            INSERT INTO meal_logs (user_id, food, calories, servings, total_calories, goal_calories, intake_pct)
                                            VALUES (:uid, :food, :cals, :serv, :total, :goal, :pct)
                                        """), {
                                            "uid": int(st.session_state.user_id),
                                            "food": str(row["food"]),
                                            "cals": float(row["caloric_value"]),
                                            "serv": float(servings),
                                            "total": float(total_calories),
                                            "goal": float(goal_cal),
                                            "pct": float(pct)
                                        })
                                        conn.commit()
                                    
                                    st.success("✅ Meal logged successfully!")
                                    st.info(f"📊 This meal contributes {pct:.1f}% of your daily goal.")
                                    st.balloons()
                                    
                                    # Suggest next action
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button("📷 Log Another Meal", key="log_another_btn"):
                                            st.rerun()
                                    with col2:
                                        if st.button("🎯 Get Recommendations ➡️", key="goto_recs_btn"):
                                            st.session_state.current_page = "Recommendations"
                                            st.rerun()
                                            
                                except Exception as e:
                                    st.error(f"❌ Failed to log meal: {e}")
                        else:
                            st.warning("⚠️ Food not found in nutritional database.")
                    except Exception as e:
                        st.error(f"Error retrieving nutritional information: {e}")
                else:
                    st.error("❌ Could not detect food. Please try another image.")
        else:
            st.info("👆 Upload a food image to get started with AI-powered food recognition!")
            st.markdown("""
            **Tips for better recognition:**
            - Take photos in good lighting
            - Show the food clearly
            - Include reference objects (like plates) for portion estimation
            - Avoid cluttered backgrounds
            """)
        
        create_navigation_buttons(current_page)

    # === RECOMMENDATIONS PAGE ===
    elif current_page == "Recommendations":
        st.header("🎯 Step 3: Personalized Recommendations")
        st.markdown("*Get AI-powered food suggestions based on your preferences and goals*")
        
        # Initialize recommendations in session state
        if 'recommendations' not in st.session_state:
            st.session_state.recommendations = []
        
        if st.button("🔄 Get New Recommendations", type="primary", key="get_new_recs_main_btn"):
            with st.spinner("🤖 Generating personalized recommendations..."):
                recommendations = rec_engine.get_content_based_recommendations(
                    st.session_state.user_id, 
                    n_recommendations=6
                )
                st.session_state.recommendations = recommendations
                st.rerun()
        
        if st.session_state.recommendations:
            st.subheader("🍽️ Recommended Foods for You")
            
            cols = st.columns(2)
            for i, rec in enumerate(st.session_state.recommendations):
                with cols[i % 2]:
                    with st.container():
                        st.markdown(f"### {rec['food'].title()}")
                        st.write(f"💡 **Reason:** {rec['reason']}")
                        
                        try:
                            with engine.connect() as conn:
                                food_info = pd.read_sql(
                                    text("SELECT * FROM foods WHERE LOWER(food) = LOWER(:food) LIMIT 1"),
                                    conn, 
                                    params={'food': rec['food']}
                                )
                            
                            if not food_info.empty:
                                info = food_info.iloc[0]
                                st.write(f"🔥 **Calories:** {info['caloric_value']}")
                                st.write(f"🥩 **Protein:** {info.get('protein', 'N/A')}g")
                                st.write(f"🍞 **Carbs:** {info.get('carbohydrates', 'N/A')}g")
                                st.write(f"🥑 **Fat:** {info.get('fat', 'N/A')}g")
                            else:
                                st.write("Nutritional info not available")
                        except Exception as e:
                            st.write("Error loading nutritional info")
                        
                        st.markdown("---")
        else:
            st.info("👆 Click the button above to get personalized food recommendations!")
            st.markdown("""
            **How recommendations work:**
            - Based on your meal history and preferences
            - Considers nutritional balance and your goals
            - Suggests similar foods to ones you've enjoyed
            - Includes healthy alternatives and new options to try
            """)
        
        create_navigation_buttons(current_page)

    # === FOOD LOG PAGE ===
    elif current_page == "Food Log":
        st.header("📊 Step 4: Your Progress & Food Log")
        st.markdown("*Track your nutrition journey with enhanced visualizations*")
        
        try:
            with engine.connect() as conn:
                log_df = pd.read_sql(text("""
                    SELECT created_at, food, total_calories, intake_pct, servings, goal_calories
                    FROM meal_logs 
                    WHERE user_id = :uid 
                    ORDER BY created_at DESC
                    LIMIT 100
                """), conn, params={"uid": st.session_state.user_id})
            
            if not log_df.empty:
                # Enhanced summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Meals", len(log_df))
                with col2:
                    avg_calories = log_df['total_calories'].mean()
                    st.metric("Avg Calories/Meal", f"{avg_calories:.0f}")
                with col3:
                    today = pd.Timestamp.now().date()
                    today_meals = log_df[pd.to_datetime(log_df['created_at']).dt.date == today]
                    total_today = today_meals['total_calories'].sum()
                    st.metric("Today's Total", f"{total_today:.0f}")
                with col4:
                    goal_cal = st.session_state.health.get('target_calories', 2000)
                    progress_pct = (total_today / goal_cal) * 100 if goal_cal else 0
                    st.metric("Goal Progress", f"{progress_pct:.1f}%")
                
                # Enhanced chart section
                st.subheader("📈 Enhanced Nutrition Analytics")
                # chart_fig = create_enhanced_calorie_chart(log_df)
                # if chart_fig:
                #     st.plotly_chart(chart_fig, use_container_width=True)
                #     st.subheader("📈 Enhanced Nutrition Analytics")
                chart_fig = create_enhanced_calorie_chart(log_df)
                if chart_fig:
                    st.plotly_chart(chart_fig, use_container_width=True)
                
                # ADD THIS NEW SECTION - Nutrition summary charts
                create_nutrition_summary_chart(log_df)
                
                # Recent meals table with better formatting
                st.subheader("📋 Recent Meals")
                display_df = log_df.head(15).copy()
                display_df['created_at'] = pd.to_datetime(display_df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
                display_df['food'] = display_df['food'].str.title()
                display_df = display_df[['created_at', 'food', 'total_calories', 'intake_pct']]
                display_df.columns = ['Date & Time', 'Food', 'Calories', 'Goal %']
                st.dataframe(display_df, use_container_width=True)
                
                # Action buttons
                st.markdown("### 🚀 What's Next?")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📷 Log Another Meal", type="primary", key="log_another_meal_main_btn"):
                        st.session_state.current_page = "Food Recognition"
                        st.rerun()
                with col2:
                    if st.button("🎯 Get New Recommendations", key="get_recs_from_log_main_btn"):
                        st.session_state.current_page = "Recommendations"
                        st.rerun()
                        
            else:
                st.info("📝 No meals logged yet. Start by uploading a food image!")
                if st.button("📷 Start Food Recognition ➡️", type="primary", key="start_food_rec_main_btn"):
                    st.session_state.current_page = "Food Recognition"
                    st.rerun()
        except Exception as e:
            st.error(f"Error loading food log: {e}")
        
        create_navigation_buttons(current_page)























