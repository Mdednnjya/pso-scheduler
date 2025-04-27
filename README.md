
# PSO Meal Plan Scheduler

An end-to-end intelligent food recommendation system designed for users with specific dietary goals. It combines **Content-Based Filtering (CBF)** and **Particle Swarm Optimization (PSO)** to suggest and optimize meal plans.

## 📋 Main Features
- **Content-Based Filtering**: Recommend meals based on user preferences and nutritional similarities.
- **PSO Meal Planning**: Automatically generate optimized daily meal plans matching nutritional targets.
- **PSO Hyperparameter Tuning**: Fine-tune the optimization parameters for better personalization.
- **MLflow Integration**: Track all experiments, models, and metrics.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/pso-scheduler.git
   cd pso-scheduler
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Mac/Linux
   venv\Scripts\activate         # Windows
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run MLflow tracking server**:
   ```bash
   mlflow ui
   ```
   Accessible at [http://127.0.0.1:5000](http://127.0.0.1:5000).

## 🚀 How to Run the Pipeline

### 1. Data Preprocessing
Enrich the raw recipes with nutrition data.
```bash
python scripts/run_preprocessing.py
```
- **Input**: \`raw_recipes.json\`, \`raw_ingredients.json\`
- **Output**: \`output/enriched_recipes.json\`

### 2. Content-Based Filtering (CBF) Training
Train the model for meal recommendations.
```bash
python scripts/run_cbf_pipeline.py
```
- **Input**: \`output/enriched_recipes.json\`
- **Output**: \`models/meal_data.json\`

Generate recommendations:
```bash
python scripts/run_cbf_pipeline.py --recommend 1 2 3 --count 5
```
With additional filters:
```bash
python scripts/run_cbf_pipeline.py --recommend 1 2 3 --exclude chicken beef --diet-type vegetarian --count 5
```

### 3. Generate Meal Plans with PSO
Automatically create optimized daily meal plans based on user profile and goals.
```bash
python scripts/run_meal_scheduler.py --age 25 --gender male --weight 70 --height 175 --activity-level moderately_active
```

Examples:
- Vegan meal plan:
  ```bash
  python scripts/run_meal_scheduler.py --age 27 --gender female --weight 58 --height 162 --activity-level moderately_active --diet-type vegan
  ```
- Weight loss meal plan excluding ingredients:
  ```bash
  python scripts/run_meal_scheduler.py --age 45 --gender male --weight 90 --height 180 --activity-level lightly_active --goal lose --exclude chicken
  ```

### 4. Evaluate the Generated Meal Plan
Assess how well the meal plan matches the user's nutritional requirements.
```bash
python scripts/run_meal_evaluator.py --meal-plan output/meal_plan.json --generate-plots
```

### 5. Tune PSO Hyperparameters
Optimize the PSO settings for better meal plan personalization.
```bash
python scripts/run_pso_tuner.py --age 30 --gender male --weight 75 --height 180 --activity-level moderately_active --goal maintain --exclude chicken beef --diet-type vegetarian
```

## 🗂️ Project Structure

```
pso-scheduler/
├── data/                  # Raw and processed data
├── mlruns/                 # MLflow experiment tracking
├── models/                 # Trained models and artifacts
├── output/                 # Generated meal plans
├── scripts/                # Pipeline scripts
├── src/                    # Source code modules
├── venv/                   # Python virtual environment
```

## Requirements
- Python 3.8+
- scikit-learn
- pandas
- numpy
- MLflow