import pandas as pd


def enrich_ingredients(ingredients, tkpi_csv_path):
    df = pd.read_csv(tkpi_csv_path)
    enriched = []

    for item in ingredients:
        name = item['ingredient']
        match = df[df['ingredient'].str.lower() == name]

        if not match.empty:
            data = match.iloc[0]
            enriched.append({
                **item,
                "calories": round(data["calories"] * item["adjusted_amount"] / 100, 2),
                "protein": round(data["protein"] * item["adjusted_amount"] / 100, 2),
                "fat": round(data["fat"] * item["adjusted_amount"] / 100, 2),
                "carbohydrates": round(data["carbohydrates"] * item["adjusted_amount"] / 100, 2),
                "dietary_fiber": round(data["dietary_fiber"] * item["adjusted_amount"] / 100, 2),
                "calcium": round(data["calcium"] * item["adjusted_amount"] / 100, 2),
            })
        else:
            # Bahan tidak ditemukan, tetap dimasukkan untuk diedit manual
            enriched.append({
                **item,
                "calories": None,
                "protein": None,
                "fat": None,
                "carbohydrates": None,
                "dietary_fiber": None,
                "calcium": None,
            })
    return enriched
