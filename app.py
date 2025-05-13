from flask import Flask, jsonify, request
import pandas as pd

app = Flask(__name__)

# Load CSV data
try:
    df = pd.read_csv("nutrition (1).csv")
except Exception as e:
    raise Exception(f"Gagal membaca file CSV: {e}")

# Bersihkan kolom
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Nutrition API!"})

@app.route("/foods", methods=["GET"])
def get_foods():
    category = request.args.get("category")
    if category:
        filtered = df[df['category'].str.lower() == category.lower()]
        return filtered.to_json(orient="records")
    return df.to_json(orient="records")

@app.route("/foods/calories", methods=["GET"])
def filter_by_calories():
    try:
        min_cal = float(request.args.get("min", 0))
        max_cal = float(request.args.get("max", float("inf")))
        filtered = df[(df["calories"] >= min_cal) & (df["calories"] <= max_cal)]
        return filtered.to_json(orient="records")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/foods/stats", methods=["GET"])
def stats():
    try:
        numeric_cols = ["calories", "protein", "fat", "carbohydrates"]
        result = df[numeric_cols].describe().to_dict()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/debug/columns", methods=["GET"])
def show_columns():
    return jsonify({"columns": df.columns.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
