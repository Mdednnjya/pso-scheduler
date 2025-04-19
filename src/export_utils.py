import pandas as pd

def export_to_json(df, output_path):
    """Export DataFrame to JSON file."""
    df.to_json(output_path, orient='records', indent=2, force_ascii=False)
    print(f"✅ Data berhasil diproses dan disimpan ke '{output_path}'")

def export_summary_csv(enriched_df, output_path):
    """Export nutrition summary to CSV file."""
    rows = []
    for _, row in enriched_df.iterrows():
        recipe_id = row['ID']
        title = row['Title']
        total = {
            'calories': 0.0,
            'protein': 0.0,
            'fat': 0.0,
            'carbohydrates': 0.0,
            'fiber': 0.0,
            'calcium': 0.0
        }
        for ing in row['Ingredients_Enriched']:
            # Hanya jumlahkan jika data gizi tersedia
            for key in total:
                if ing.get(key) not in [None, ""]:
                    try:
                        total[key] += float(ing[key])
                    except:
                        continue
        rows.append({
            'ID': recipe_id,
            'Title': title,
            'Calories': round(total['calories'], 2),
            'Protein': round(total['protein'], 2),
            'Fat': round(total['fat'], 2),
            'Carbohydrates': round(total['carbohydrates'], 2),
            'Fiber': round(total['fiber'], 2),
            'Calcium': round(total['calcium'], 2),
        })

    df_export = pd.DataFrame(rows)
    df_export.to_csv(output_path, index=False)
    print(f"✅ Ringkasan gizi per resep berhasil disimpan ke '{output_path}'")