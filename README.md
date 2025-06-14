<p align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=22&pause=1000&center=true&vCenter=true&width=435&lines=🥗+Smart+Calorie+Tracker+App;Built+with+Streamlit+%2B+Supabase+%2B+ML+Models" />
</p>

# 🧮 calorie_tracker

> **AI-powered Food Logger + Nutrition Analyzer + Personal Calorie Planner**  
> Built with ❤️ for fitness enthusiasts and everyday eaters.

---

## 🚀 Overview

An AI-powered full-stack nutrition tracking web app built with **Streamlit** and backed by a real-time **Supabase PostgreSQL** database. Upload a food image, get instant predictions, and track your daily calorie and macro intake—all personalized to your profile.

---

## 🧠 Key Features

- 🍱 **Food Image Classifier** (HuggingFace `nateraw/food`)
- 🔬 Nutrient Breakdown with **Donut Charts**
- ⚖️ BMR, TDEE & Daily Goal Calculator
- 🧾 Meal Logging & Progress Tracker
- 🧍‍♂️ Custom Profile Based on Age, Weight, Height
- 🔐 Email-Based User Authentication (Supabase)

---
🧹 Data Preprocessing & Integration

A crucial part of this AI-powered calorie tracker was building a robust nutrition database from multiple sources. Here’s how I approached it:

📦 Data Sources:
	•	USDA Nutrition Dataset
	•	OpenFoodFacts API dumps
	•	Additional datasets containing micronutrient information and food group classifications

⸻
## 🧪 Cleaning & Processing Steps

| Step | Description |
|------|-------------|
| 1. Remove Invalid Entries | Dropped rows with nulls, non-edible items, or zero calories |
| 2. Column Normalization | Standardized column names and units (e.g., kcal, g, mg) |
| 3. Label Cleaning | Cleaned food names using regex, `str.replace()` for consistency |
| 4. Nutrient Extraction | Focused on key macros: protein, carbohydrates, fat, free sugar, fiber |
| 5. Serving Size Alignment | Normalized data to 100g serving size for consistency |
| 6. Data Merging | Used `pandas.merge()` and `concat()` to unify multiple datasets |
| 7. Feature Engineering | Added fields like `caloric_density`, `macro_ratio`, `food_group` |
| 8. Output Files Generated | Final datasets: <br>• `merged_food_group_dataset.csv`<br>• `filtered_nutrition_dataset.csv`<br>• `merged_extra_nutrition_dataset.csv`<br>• `final_combined_nutrition_dataset.csv` |

⸻

🛠️ Final Output:
	•	The final dataset was uploaded to a Supabase PostgreSQL table (foods) using Python and SQLAlchemy.
	•	This structured data serves as the core lookup for nutrient values in the app and supports user-specific recommendations and chart visualizations.


---
## ⚙️ Tech Stack
## 🛠️ Installation Guide

### 📁 1. Clone the Repository


git clone https://github.com/your-username/calorie-tracker.git
cd calorie-tracker

Create a .streamlit/secrets.toml file 

DB_PASSWORD = "dm for the access"

▶️ 4. Run the App
streamlit run app.py

📦 3. Install Requirements
pip install -r requirements.txt
⚙️ On M1/M2 or Linux, make sure you install torch, opencv-python-headless, and compatible transformers.


| Layer          | Technology                                |
|----------------|--------------------------------------------|
| 🖥 Frontend     | Streamlit                                  |
| 🧠 AI Model     | HuggingFace `nateraw/food`                 |
| 🗃 Backend DB   | Supabase PostgreSQL                        |
| 🔍 Data Viz     | Plotly, Matplotlib                         |
| 🔐 Auth & API   | Supabase                                   |
| 🧹 Preprocessing| Pandas, PIL, Transformers, SQLAlchemy     |


------
🧪 Example Usage
	1.	Sign up with your name, email, and password
	2.	Fill out your profile (weight, height, goal, activity level)
	3.	Upload a food image → get detected label
	4.	Confirm or refine food name
	5.	Log servings and view donut chart of nutrients
	6.	Monitor your daily intake % and meal history

⸻

🔧 Data Pipeline
	•	✅ Scraped + cleaned food data from OpenFoodFacts and USDA datasets
	•	🧼 Performed preprocessing, normalization, and column mapping using Pandas
	•	☁️ Loaded structured data into Supabase PostgreSQL
	•	🔎 Query foods in real-time and calculate nutrients based on servings
	•	🔁 Log entries saved to meal_logs with timestamp and user reference

⸻

## 📂 File Structure

| File/Folder                      | Description                                      |
|----------------------------------|--------------------------------------------------|
| `app.py`                         | Main Streamlit application script               |
| `requirements.txt`              | Python package dependencies                     |
| `.streamlit/`                   | Contains `secrets.toml` for DB credentials      |
| `README.md`                     | You are here – project documentation            |
| `filtered_nutrition_dataset.csv`| Cleaned dataset with key nutrients              |
| `final_combined_nutrition_dataset.csv` | Final merged dataset with all fields     |
| `merged_extra_nutrition_dataset.csv` | Additional merged data for micronutrients |
| `merged_food_group_dataset.csv` | Dataset including food group classification     |

📄 License

This project is licensed under the MIT License.
Feel free to fork, improve, and contribute!

🙌 Acknowledgements
	•	🤗 HuggingFace for the nateraw/food model
	•	🐘 Supabase for PostgreSQL and Auth
	•	🥗 OpenFoodFacts & USDA Food Data


<p align="center">
  <img src="https://forthebadge.com/images/badges/made-with-python.svg" />
  <img src="https://img.shields.io/badge/Streamlit-%23FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white" />
</p>

