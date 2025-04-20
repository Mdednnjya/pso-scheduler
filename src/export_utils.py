import pandas as pd
import json


def export_to_json(df, output_path):
    """Export DataFrame to JSON file."""
    df.to_json(output_path, orient='records', indent=2, force_ascii=False)
    print(f"✅ Data berhasil diproses dan disimpan ke '{output_path}'")


def get_portion_factor(ingredient_name, total_ingredients_count):
    """
    Menentukan faktor porsi berdasarkan jenis bahan dan jumlah total bahan dalam resep.

    Args:
        ingredient_name: Nama bahan
        total_ingredients_count: Jumlah total bahan dalam resep

    Returns:
        Float: Faktor porsi (seberapa banyak dari 100g yang digunakan)
    """
    # Bahan utama biasanya porsinya lebih besar
    main_ingredients = ['udang', 'ayam', 'daging', 'ikan', 'telur', 'tahu', 'tempe']

    # Bumbu biasanya digunakan dalam jumlah kecil
    spices = ['bawang', 'cabai', 'jahe', 'kunyit', 'lengkuas', 'merica', 'ketumbar',
              'garam', 'gula', 'terasi', 'kemiri', 'daun salam', 'daun jeruk', 'serai']

    # Minyak dan santan digunakan dalam jumlah sedang
    fats = ['minyak', 'santan']

    # Default portion factor
    portion_factor = 0.15  # 15g dari 100g, estimasi default untuk 1 porsi

    # Mengurangi faktor porsi jika resep memiliki banyak bahan
    if total_ingredients_count > 10:
        portion_factor *= 0.8

    # Menyesuaikan faktor porsi berdasarkan jenis bahan
    ingredient_lower = ingredient_name.lower()

    # Bahan utama
    if any(main in ingredient_lower for main in main_ingredients):
        portion_factor = 0.25  # 25g dari 100g

    # Bumbu
    elif any(spice in ingredient_lower for spice in spices):
        portion_factor = 0.05  # 5g dari 100g

    # Minyak dan santan
    elif any(fat in ingredient_lower for fat in fats):
        portion_factor = 0.10  # 10g dari 100g

    return portion_factor


def export_summary_csv(enriched_df, output_path):
    """Export nutrition summary to CSV file dengan perhitungan nutrisi yang disesuaikan untuk porsi diet remaja."""
    rows = []

    # Target kalori untuk remaja yang diet
    target_calories = 350  # per porsi

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

        matched_count = 0
        total_ingredients = len(row['Ingredients_Parsed'])

        # Hitung total kalori asli terlebih dahulu
        original_calories = 0
        for ing in row['Ingredients_Enriched']:
            if ing.get('calories') not in [None, "", 0, 0.0, "0", "0.0"]:
                try:
                    original_calories += float(ing['calories']) * get_portion_factor(ing['ingredient'],
                                                                                     total_ingredients)
                except (ValueError, TypeError):
                    continue

        # Jika total kalori masih terlalu tinggi, terapkan faktor penyesuaian
        adjustment_factor = 1.0
        if original_calories > 0 and original_calories > target_calories:
            adjustment_factor = target_calories / original_calories

        # Sekarang hitung semua nutrisi dengan adjustment factor
        for ing in row['Ingredients_Enriched']:
            # Hitung total nutrisi hanya jika nilai ada dan bukan None/kosong/0
            any_nutrient = False
            portion_factor = get_portion_factor(ing.get('ingredient', ''), total_ingredients)

            for key in total.keys():
                if ing.get(key) not in [None, "", 0, 0.0, "0", "0.0"]:
                    try:
                        value = float(ing[key])
                        if value > 0:
                            # Terapkan faktor porsi dan faktor penyesuaian
                            adjusted_value = value * portion_factor * adjustment_factor
                            total[key] += adjusted_value
                            any_nutrient = True
                    except (ValueError, TypeError):
                        continue

            if any_nutrient:
                matched_count += 1

        match_percentage = 0 if total_ingredients == 0 else (matched_count / total_ingredients) * 100

        rows.append({
            'ID': recipe_id,
            'Title': title,
            'Calories': round(total['calories'], 1),
            'Protein': round(total['protein'], 1),
            'Fat': round(total['fat'], 1),
            'Carbohydrates': round(total['carbohydrates'], 1),
            'Fiber': round(total['fiber'], 1),
            'Calcium': round(total['calcium'], 1),
            'Match_Rate': f"{matched_count}/{total_ingredients} ({match_percentage:.1f}%)"
        })

    df_export = pd.DataFrame(rows)
    df_export.to_csv(output_path, index=False)
    print(f"✅ Ringkasan gizi per resep berhasil disimpan ke '{output_path}'")

    # Tambahkan laporan pencocokan untuk debugging
    export_matching_report(enriched_df, output_path.replace('.csv', '_matching_report.json'))


def export_matching_report(enriched_df, output_path):
    """Export a detailed report of ingredient matching for debugging."""
    report = []

    for _, row in enriched_df.iterrows():
        recipe = {
            'ID': row['ID'],
            'Title': row['Title'],
            'Ingredients': []
        }

        for parsed in row['Ingredients_Parsed']:
            # Cari bahan yang diperkaya yang sesuai
            matching_enriched = next((e for e in row['Ingredients_Enriched'] if
                                      e.get('original_ingredient', e.get('ingredient', '')) == parsed), None)

            if matching_enriched:
                ing_report = {
                    'parsed': parsed,
                    'matched_to': matching_enriched.get('ingredient', 'No match'),
                    'match_type': matching_enriched.get('match_type', 'unknown'),
                    'calories': matching_enriched.get('calories', 0),
                    'has_nutrition': any(matching_enriched.get(key, 0) not in [None, "", 0, 0.0, "0", "0.0"]
                                         for key in ['calories', 'protein', 'fat', 'carbohydrates', 'fiber', 'calcium'])
                }
            else:
                ing_report = {
                    'parsed': parsed,
                    'matched_to': 'Not found in enriched data',
                    'match_type': 'none',
                    'has_nutrition': False
                }

            recipe['Ingredients'].append(ing_report)

        report.append(recipe)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"✅ Laporan pencocokan bahan disimpan ke '{output_path}'")