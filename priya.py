# ===================== Imports =====================
import streamlit as st
from PIL import Image
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from keras.models import load_model
import pandas as pd
import os
import json
import random
import re
from string import Template
from typing import Optional


# ===================== Helpers: CSV standardization =====================
def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip all column names for easy access."""
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def load_csv_safe(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return _standardize_columns(df)
    except Exception:
        pass
    return None


# ===================== Load Data/Model =====================
MODEL_PATH = 'FV2.h5'
_model = None
try:
    if os.path.exists(MODEL_PATH):
        _model = load_model(MODEL_PATH)
except Exception:
    _model = None


# Core CSVs (standardized columns)
fruits_df = load_csv_safe("fruits.csv")
vegetables_df = load_csv_safe("vegetables.csv")
nutrient_info_df = load_csv_safe("nutrients_info.csv")
# Add with your other CSV loads
recipes_df = load_csv_safe("food_recipes.csv")
# Load recipes CSV safely
recipes_df = pd.read_csv("recipes.csv")

# Standardize column names
recipes_df.columns = recipes_df.columns.str.strip().str.lower()

# Ensure expected columns exist
# Rename common variations to 'ingredients' and 'recipe'
if "food items" in recipes_df.columns:
    recipes_df.rename(columns={"food items": "ingredients"}, inplace=True)
if "item" in recipes_df.columns:
    recipes_df.rename(columns={"item": "ingredients"}, inplace=True)
if "recipe name" in recipes_df.columns:
    recipes_df.rename(columns={"recipe name": "recipe"}, inplace=True)
if "recipes" in recipes_df.columns:
    recipes_df.rename(columns={"recipes": "recipe"}, inplace=True)


# Calorie catalogue (Tamil Nadu foods)
calorie_df_raw = load_csv_safe("tamil_nadu_foods.csv")
calorie_df = None
if calorie_df_raw is not None:
    rename_map = {}
    cols = set(calorie_df_raw.columns)
    if 'food items' in cols:
        rename_map['food items'] = 'Food Items'
    elif 'food item' in cols:
        rename_map['food item'] = 'Food Items'
    elif 'item' in cols:
        rename_map['item'] = 'Food Items'
    if 'quantity' in cols:
        rename_map['quantity'] = 'Quantity'
    if 'category' in cols:
        rename_map['category'] = 'Category'
    # Add this line to handle your CSV exact column name "calorie value"
    if 'calorie value' in cols:
        rename_map['calorie value'] = 'Calorie Values (kcal)'
    elif 'calorie values (kcal)' in cols:
        rename_map['calorie values (kcal)'] = 'Calorie Values (kcal)'
    elif 'calorie value (kcal)' in cols:
        rename_map['calorie value (kcal)'] = 'Calorie Values (kcal)'
    elif 'calories' in cols:
        rename_map['calories'] = 'Calorie Values (kcal)'
    elif 'kcal' in cols:
        rename_map['kcal'] = 'Calorie Values (kcal)'

    calorie_df = calorie_df_raw.rename(columns=rename_map)


# ===================== Constants & Labels =====================
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
    7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
    14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango', 20: 'onion', 21: 'orange',
    22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'radish',
    29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'
}


fruits = [
    'apple', 'banana', 'bell pepper', 'chilli pepper', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'mango',
    'orange', 'paprika', 'pear', 'pineapple', 'pomegranate', 'watermelon'
]
vegetables = [
    'beetroot', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'corn', 'cucumber', 'eggplant', 'ginger',
    'lettuce', 'onion', 'peas', 'potato', 'radish', 'soy beans', 'spinach', 'garlic', 'sweetcorn',
    'sweetpotato', 'tomato', 'turnip'
]


main_nutrients = [
    "energy (kcal/kJ)", "water (g)", "protein (g)", "total fat (g)", "carbohydrates (g)",
    "fiber (g)", "sugar (g)", "calcium (mg)", "iron (mg)", "vitamin A (µg)", "vitamin C (mg)"
]


rdi = {
    "energy (kcal/kJ)": 2000, "water (g)": 2000, "protein (g)": 50, "total fat (g)": 70, "carbohydrates (g)": 275,
    "fiber (g)": 30, "sugar (g)": 50, "calcium (mg)": 1000, "iron (mg)": 18, "vitamin A (µg)": 900, "vitamin C (mg)": 90
}


mood_food_map = {
    "Tired": ["banana", "orange", "spinach", "almonds", "Onion Pakoda"],
    "Happy": ["mango", "watermelon", "pineapple", "strawberry", "ice cream"],
    "Stressed": ["spinach", "grapes", "lemon", "green tea", "dark chocolate"],
    "Studying": ["apple", "carrot", "kiwi", "nuts", "dates"],
    "Post-Workout": ["banana", "soy beans", "pomegranate", "chicken breast", "eggs"]
}


# ===================== Core Functions =====================
def get_nutrition_info(item_name: str):
    item_name = (item_name or "").lower().strip()

    def _find_in(df):
        if df is None:
            return None
        if 'name' not in df.columns:
            for alt in ['item', 'food', 'title']:
                if alt in df.columns:
                    df = df.rename(columns={alt: 'name'})
                    break
            if 'name' not in df.columns:
                return None
        mask = df['name'].astype(str).str.lower().str.contains(
            re.escape(item_name), na=False)
        match = df[mask]
        return match.iloc[0] if not match.empty else None

    fruit_row = _find_in(fruits_df)
    if fruit_row is not None:
        return fruit_row
    veg_row = _find_in(vegetables_df)
    if veg_row is not None:
        return veg_row

    return {n: round(random.uniform(0.5, rdi[n]*0.2), 2) for n in main_nutrients}


def processed_img(img_path: str):
    if _model is None:
        raise RuntimeError(
            "Model FV2.h5 not loaded. Place the model file next to the script.")
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = _model.predict(img)
    class_index = int(np.argmax(prediction))
    return labels.get(class_index, "unknown")


def plot_chart(labels_js, perc_js, title, chart_key):
    chart_choice = st.radio(
        "📊 Select Chart Type:",
        ["3D Donut (Plotly.js)", "3D Bars"],
        horizontal=True, key=f"chart_type_{chart_key}"
    )

    if chart_choice == "3D Donut (Plotly.js)":
        tpl = Template("""
        <div id="donut_${chart_key}" style="width:100%;height:600px;"></div>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <script>
          const labels = ${labels};
          const values = ${values};
          Plotly.newPlot("donut_${chart_key}", [{
            type:"pie", labels:labels, values:values, hole:0.5,
            textinfo:"label+percent", textposition:"inside",
            marker:{line:{color:"#fff", width:2}}
          }], {
            title:"${title}",
            height:600, showlegend:true
          }, {displaylogo:false,responsive:true});
        </script>
        """)
        html = tpl.safe_substitute(
            chart_key=chart_key,
            labels=json.dumps(labels_js),
            values=json.dumps(perc_js),
            title=title.replace('"', '\\"')
        )
        st.components.v1.html(html, height=640)
    else:
        tpl = Template("""
        <div id="bars3d_${chart_key}" style="width:100%;height:620px;"></div>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
        <script>
          const labels = ${labels};
          const values = ${values};
          const x=[], y=[], z=[], text=[];
          for(let i=0; i<labels.length; i++){
              x.push(i); y.push(0); z.push(values[i]);
              text.push(labels[i]+"<br>"+values[i].toFixed(2)+"%");
          }
          Plotly.newPlot("bars3d_${chart_key}", [{
            type:"scatter3d", mode:"markers+lines", x:x, y:y, z:z, text:text,
            marker:{size:6}, line:{width:18}
          }], {
            title:"${title}",
            scene:{xaxis:{tickvals:x, ticktext:labels}, zaxis:{title:"% of RDI"}},
            margin:{l:0,r:0,t:60,b:0}
          }, {displaylogo:false,responsive:true});
        </script>
        """)
        html = tpl.safe_substitute(
            chart_key=chart_key,
            labels=json.dumps(labels_js),
            values=json.dumps(perc_js),
            title=title.replace('"', '\\"')
        )
        st.components.v1.html(html, height=660)


def show_nutrition_charts(item_name, nutrition):
    st.subheader(f"📋 Nutrition Table for {item_name.capitalize()}")
    nutri_df = pd.DataFrame({
        "Nutrient": main_nutrients,
        "Value": [pd.to_numeric(pd.Series(nutrition).get(n, 0), errors='coerce') for n in main_nutrients]
    })
    nutri_df["Value"] = pd.to_numeric(
        nutri_df["Value"], errors="coerce").fillna(0)
    nutri_df["% of RDI"] = [
        round((v/rdi[n])*100, 2) if rdi[n] > 0 else 0
        for v, n in zip(nutri_df["Value"], nutri_df["Nutrient"])
    ]
    st.dataframe(nutri_df, use_container_width=True)
    plot_chart(
        nutri_df["Nutrient"].tolist(),
        nutri_df["% of RDI"].tolist(),
        f"{item_name.capitalize()} — % of Daily Intake",
        f"{item_name}_rdi"
    )


def show_prediction_results(result, nutrition):
    st.success(f"**Prediction:** {result.capitalize()}")
    if result.lower() in vegetables:
        st.info("**Category:** Vegetable 🥕")
    elif result.lower() in fruits:
        st.info("**Category:** Fruit 🍉")
    st.subheader("📌 Details")
    st.write(f"- **Name:** {result.capitalize()}")
    show_nutrition_charts(result, nutrition)


def recommend_recipes(item_name: str, top_n: int = 6):
    """Return up to top_n recipes for a given food (uses recipes_df)."""
    if recipes_df is None:
        return []
    cols = set(recipes_df.columns)
    if 'food' not in cols or 'recipe' not in cols:
        return []
    mask = recipes_df['food'].astype(str).str.lower() == item_name.lower()
    recs = recipes_df.loc[mask, 'recipe'].dropna().unique().tolist()
    return recs[:top_n]


# ===================== Streamlit App =====================
def run():
    st.set_page_config(
        page_title="Fruit & Vegetable Classifier", layout="wide")
    st.title("🥦 Fruit & Vegetable Classifier — Nutrition Charts 🍎")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📷 Upload Image",
        "🔍 Search by Name",
        "🍱 Meal Planner (Nutrients)",
        "🍛 Calorie Planner (Veg/Non-Veg)",
        "⚡ Mood Food Suggestions"
    ])

    # -------------------- TAB 1: Upload (NO voice search) --------------------
    with tab1:
        img_file = st.file_uploader(
        "📷 Upload an image of fruit/vegetable", type=["jpg", "jpeg", "png"]
    )

    if img_file:
        # Show uploaded image
        img = Image.open(img_file).convert("RGB").resize((250, 250))
        st.image(img, caption="Uploaded Image", use_column_width=False)

        # Save uploaded image
        os.makedirs("upload_images", exist_ok=True)
        save_path = os.path.join("upload_images", img_file.name)
        with open(save_path, "wb") as f:
            f.write(img_file.getbuffer())

        try:
            # Prediction
            result = processed_img(save_path)

            # Nutrition info
            nutrition = get_nutrition_info(result)
            show_prediction_results(result, nutrition)

            # Additional nutrient descriptions
            if nutrient_info_df is not None:
                required_cols = {"nutrient", "description"}
                if required_cols.issubset(nutrient_info_df.columns):
                    st.subheader("ℹ️ Nutrient Information")
                    for _, row in nutrient_info_df.iterrows():
                        st.markdown(
                            f"**{row['nutrient']}** → {row['description']}"
                        )

            # Recipe recommendations
            if recipes_df is not None:
                recs = recommend_recipes(result, top_n=6)
                st.subheader(f"🍴 Recipes using {result.capitalize()}")

                if recs:
                    cols = st.columns(min(3, len(recs)))
                    for i, recipe in enumerate(recs):
                        with cols[i % 3]:
                            st.markdown(
                                f"""
                                <div style="
                                    background: linear-gradient(135deg, #fff7f9, #fefefe);
                                    color: black;
                                    padding: 14px; margin: 8px 0;
                                    border-radius: 12px;
                                    box-shadow: 0 4px 10px rgba(0,0,0,0.08);
                                    text-align: center;
                                    font-weight: 600;
                                    font-size: 15px;">
                                    {recipe}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                else:
                    st.info("No recipes found for this ingredient in recipes.csv.")

        except Exception as e:
            st.error(f"⚠️ Prediction error: {e}")

    # -------------------- TAB 2: Search (text + selectbox only) --------------------
    with tab2:
        st.subheader("🔍 Search Fruit/Vegetable")
        item_query = st.text_input(
            "📝 Type fruit or vegetable:", key="search_text")
        all_items = sorted(fruits + vegetables)
        dropdown_choice = st.selectbox(
            "🔽 Select fruit/vegetable:", [""] + all_items, key="search_dropdown")
        final_query = dropdown_choice if dropdown_choice else item_query
        if final_query:
            nutrition = get_nutrition_info(final_query)
            show_prediction_results(final_query, nutrition)

    # -------------------- TAB 3: Meal Planner Nutrients (text + multiselect only) --------------------
    with tab3:
        st.subheader("🍽️ Build Your Meal Plate (Nutrients)")
        item_list_text = st.text_input(
            "📝 Type items (comma-separated):", key="meal_text")
        multiselect_items = st.multiselect(
            "🔽 Choose items:", sorted(fruits + vegetables), key="meal_multi")

        selected_items = []
        if item_list_text:
            selected_items += [i.strip()
                               for i in item_list_text.split(",") if i.strip()]
        for it in multiselect_items:
            if it not in selected_items:
                selected_items.append(it)

        if selected_items:
            total_nutrition = {n: 0.0 for n in main_nutrients}
            for item in selected_items:
                nutrition = get_nutrition_info(item)
                for n in main_nutrients:
                    try:
                        total_nutrition[n] += float(pd.to_numeric(
                            pd.Series(nutrition).get(n, 0), errors='coerce'))
                    except Exception:
                        total_nutrition[n] += 0.0

            meal_df = pd.DataFrame({
                "Nutrient": main_nutrients,
                "Total Value": [round(total_nutrition[n], 2) for n in main_nutrients],
                "% of RDI": [
                    round((total_nutrition[n] / rdi[n]) * 100, 2) if rdi[n] > 0 else 0 for n in main_nutrients
                ]
            })
            st.subheader("🥗 Nutrition Summary for Selected Items")
            st.dataframe(meal_df, use_container_width=True)
            plot_chart(
                meal_df["Nutrient"].tolist(),
                meal_df["% of RDI"].tolist(),
                "Total Meal — % of Daily Intake",
                "meal_total_rdi"
            )

            if nutrient_info_df is not None:
                ndf = nutrient_info_df
                ndf_cols = set(ndf.columns)
                n_col = 'nutrient' if 'nutrient' in ndf_cols else None
                d_col = 'description' if 'description' in ndf_cols else None
                if n_col and d_col:
                    st.subheader("ℹ️ Nutrient Information")
                    for _, row in ndf.iterrows():
                        st.markdown(f"**{row[n_col]}** → {row[d_col]}")

    # -------------------- TAB 4: Calorie Planner (text + select + multiselect only) --------------------
    with tab4:
        st.subheader("🍛 Build Your Meal Plate (Calories) — Veg / Non-Veg")
        if calorie_df is None:
            st.warning(
                "`tamil_nadu_foods.csv` not found or missing required columns.")
        else:
            search = st.text_input(
                "📝 Filter dishes (optional):", key="calp_text")

            cat_col = 'Category' if 'Category' in calorie_df.columns else None
            category = st.radio(
                "Category:",
                ["All", "Veg", "Non-Veg"],
                horizontal=True,
                key="cal_cat_radio"
            )
            df = calorie_df.copy()
            if 'Calorie Values (kcal)' in df.columns:
                df['Calorie Values (kcal)'] = pd.to_numeric(
                    df['Calorie Values (kcal)'], errors='coerce')

            if category != "All" and cat_col:
                df = df[df[cat_col].astype(
                    str).str.strip().str.lower() == category.lower()]

            fi_col = 'Food Items' if 'Food Items' in df.columns else None
            if search and fi_col:
                df = df[df[fi_col].astype(str).str.contains(
                    search, case=False, na=False)]

            single_select = None
            if fi_col:
                all_foods = [
                    ""] + sorted(df[fi_col].dropna().astype(str).unique().tolist())
                single_select = st.selectbox(
                    "🔽 Pick one dish to add:", all_foods, key="cal_single")

            qty_col = 'Quantity' if 'Quantity' in df.columns else None
            kcal_col = 'Calorie Values (kcal)' if 'Calorie Values (kcal)' in df.columns else None

            def _fmt_row(r):
                name = str(r.get(fi_col, ''))
                qty = str(r.get(qty_col, '')) if qty_col else ''
                kcal_val = r.get(kcal_col, None) if kcal_col else None
                kcal_txt = f"{int(kcal_val)}" if pd.notna(kcal_val) else 'N/A'
                return f"{name} — {qty} (≈ {kcal_txt} kcal)"

            options = [_fmt_row(r) for _, r in df.iterrows()] if fi_col else []
            label_to_rowidx = {opt: idx for opt, idx in zip(options, df.index)}

            chosen_default = []
            if single_select:
                match_idx = df[df[fi_col].astype(str) == single_select].index
                if len(match_idx) > 0:
                    opt = _fmt_row(df.loc[match_idx[0]])
                    chosen_default.append(opt)

            chosen = st.multiselect(
                "Choose items:",
                options=options,
                default=chosen_default,
                key="cal_item_multiselect"
            )

            if chosen:
                st.markdown("**Servings per selected item:**")
                servings = {}
                for opt in chosen:
                    key = f"serv_{opt}"
                    servings[opt] = st.number_input(
                        opt, min_value=1, max_value=100, value=1, step=1, key=key
                    )
                rows = []
                for opt in chosen:
                    idx = label_to_rowidx.get(opt)
                    if idx is None:
                        continue
                    row = df.loc[idx]
                    kcal = row[kcal_col] if kcal_col else None
                    qty = row[qty_col] if qty_col else None
                    name = row[fi_col] if fi_col else None
                    count = servings[opt]
                    total_kcal = (kcal if pd.notna(kcal) else 0) * \
                        count if kcal is not None else 0
                    rows.append({
                        "Food Item": name,
                        "Quantity (per serving)": qty,
                        "Calories (per serving)": round(float(kcal), 2) if pd.notna(kcal) else None,
                        "Servings": count,
                        "Total Calories": round(float(total_kcal), 2)
                    })
                result_df = pd.DataFrame(rows)
                total_calories = float(
                    result_df["Total Calories"].sum()) if not result_df.empty else 0.0
                st.subheader("📊 Selected Items — Calories")
                st.dataframe(result_df, use_container_width=True)
                st.success(f"**Total Calories: {int(total_calories)} kcal**")

                labels_pie = result_df["Food Item"].astype(str).tolist()
                values_pie = result_df["Total Calories"].astype(float).tolist()
                tpl = Template("""
                <div id="cal_pie" style="width:100%;height:520px;"></div>
                <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
                <script>
                const labels = ${labels};
                const values = ${values};
                Plotly.newPlot("cal_pie", [{
                  type: "pie",
                  labels: labels,
                  values: values,
                  hole: 0.35,
                  textinfo: "label+percent",
                  textposition: "inside",
                  marker: { line: { color: "#fff", width: 2 } }
                }], {
                  title: "Calorie Share by Selected Items",
                  height: 520,
                  showlegend: true
                }, { displaylogo: false, responsive: true });
                </script>
                """)
                html = tpl.safe_substitute(labels=json.dumps(
                    labels_pie), values=json.dumps(values_pie))
                st.components.v1.html(html, height=560)

   # -------------------- TAB 5: Mood Food (text + selectbox + radio only) --------------------
    with tab5:
     st.subheader("⚡ Mood-based Food Recommendation")
 
     mood_radio = st.radio(
         "Pick your mood:",
           list(mood_food_map.keys()),
         horizontal=True,
         key="mood_radio"
     )
 
     effective_mood = mood_radio if mood_radio else None
 
     if effective_mood:
         st.success(f"✨ Foods recommended for **{effective_mood}** mood:")
 
         # ================= CSS for 3D card effect =================
         st.markdown("""
         <style>
         .food-card {
             background: linear-gradient(135deg, #f9f9f9, #ffffff);
             border-radius: 15px;
             padding: 20px;
             margin: 15px;
             box-shadow: 0 8px 20px rgba(0,0,0,0.15);
             transition: transform 0.2s ease-in-out;
             text-align: center;
         }
         .food-card:hover {
             transform: scale(1.05) rotate(-1deg);
             box-shadow: 0 12px 25px rgba(0,0,0,0.25);
         }
         .food-title {
             font-size: 20px;
             font-weight: bold;
             margin-bottom: 10px;
             color: #333;
         }
         </style>
         """, unsafe_allow_html=True)

         # ================= Show Foods in Cards =================
         cols = st.columns(3)
         for idx, item in enumerate(mood_food_map[effective_mood][:5]):
             with cols[idx % 3]:
                  st.markdown(
                     f"<div class='food-card'><div class='food-title'>{item.capitalize()}</div></div>",
                     unsafe_allow_html=True
                )

# Run the app
if __name__ == "__main__":
    run()
