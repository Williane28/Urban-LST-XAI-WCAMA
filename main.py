#%%
import os
import time
import json
import pandas as pd
from sklearn.model_selection import KFold
import config
from utils import (
    load_file, 
    aggregate_external_variables, 
    prepare_data, 
    plot_distribution, 
    apply_selection_to_all, 
    plot_relevance_composite, 
    compare_models_with_kfold,
    optimizer_optuna,
    train_models,
    plot_regression,
    shap_values,
    summary_plot,
    plot_shap_grid_simple)

#%%
# Project Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

DATA_DIR = os.path.join(BASE_DIR, 'data')

# Ensure input and output folders exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# %%
# Dataset Loading
print("\nLoading local datasets...")

df_raw_2018 = load_file(config.VARIABLES_2018)
df_raw_2021 = load_file(config.VARIABLES_2021)

# External Variable Aggregation and TPI Calculation
print("\nAggregating external variables and calculating TPI...")

df_agg_2018 = aggregate_external_variables(
    df_raw_2018,
    config.VARIABLES_2018_MDT_IND_EXT
)

df_agg_2021 = aggregate_external_variables(
    df_raw_2021,
    config.VARIABLES_2021_MDT_IND_EXT
)

# Data Preparation and Cleaning (Typing, Renaming)
print("\nPreparing and cleaning data (explicit renaming)...")

df_2018 = prepare_data(df_agg_2018)
df_2021 = prepare_data(df_agg_2021)

# Distribution Plots and High Correlation Check
print("\nGenerating density plots for explanatory variables...")

plot_distribution(
    df_2018,
    ncols=5,
    file_name='distribution_density_2018.png'
)

plot_distribution(
    df_2021,
    ncols=5,
    file_name='distribution_density_2021.png'
)

#%%

dataset_dict = {"2018": df_2018, "2021": df_2021}

# Intelligent Feature Selection (Pearson r / Multicollinearity)
print("\nRunning feature selection by correlation (Threshold = 0.90)...")

CORR_THRESHOLD = 0.90

selected_cols = apply_selection_to_all(
    dataset_dict,
    threshold=CORR_THRESHOLD,
    target_col='LST',
)

df_reduced_2018 = df_2018[selected_cols['2018'] + ['LST']]
df_reduced_2021 = df_2021[selected_cols['2021'] + ['LST']]

# Pearson r composite plot
reduced_dataset_dict = {
    "2018": df_reduced_2018,
    "2021": df_reduced_2021
}

plot_relevance_composite(
    reduced_dataset_dict,
    threshold=CORR_THRESHOLD,
    target_col='LST',
    file_name='relevance_ranking_composite.png'
)
#%%
# Model Comparison (XGBoost vs LightGBM) using Cross-Validation
print("\nComparing XGBoost and LightGBM models using Cross-Validation (10-Fold)...")

KFOLD_SPLITS = 10
KFOLD_RANDOM_SEED = int(time.time_ns() % (2**32))

kfold = KFold(
    n_splits=KFOLD_SPLITS,
    shuffle=True,
    random_state=KFOLD_RANDOM_SEED
)

cv_results_2018, X_2018, y_2018 = compare_models_with_kfold(
    df_reduced_2018,
    kfold,
    KFOLD_RANDOM_SEED,
    '2018'
)

cv_results_2021, X_2021, y_2021 = compare_models_with_kfold(
    df_reduced_2021,
    kfold,
    KFOLD_RANDOM_SEED,
    '2021'
)
#%%
# Combine results and save locally as CSV
df_cv_comparison = pd.concat(
    [cv_results_2018, cv_results_2021],
    ignore_index=True
)

cv_csv_path = os.path.join(
    config.OUTPUT_DIR,
    'metrics_cv_comparison.csv'
)

df_cv_comparison.to_csv(
    cv_csv_path,
    index=False
)

print(
    f"[SAVED] Cross-validation results saved at: {cv_csv_path}"
)
#%%
# Hyperparameter Optimization using Optuna
print(
    "\nOptimizing hyperparameters with Optuna (RUN_OPTUNA=True)..."
)

study_2018, seed_2018 = optimizer_optuna(
    X_2018,
    y_2018,
    '2018'
)

study_2021, seed_2021 = optimizer_optuna(
    X_2021,
    y_2021,
    '2021'
)

optuna_results = {
    '2018': {
        'best_r2': study_2018.best_value,
        'best_params': study_2018.best_params
    },

    '2021': {
        'best_r2': study_2021.best_value,
        'best_params': study_2021.best_params
    }
}

optuna_json_path = os.path.join(
    config.OUTPUT_DIR,
    'optuna_best_params.json'
)

with open(
    optuna_json_path,
    'w',
    encoding='utf-8'
) as f:

    json.dump(
        optuna_results,
        f,
        indent=4,
        ensure_ascii=False
    )

print(
    f"[SAVED] Best Optuna hyperparameters saved at: {optuna_json_path}"
)

#%% Model training 2018 and 2021
best_params_2018 = optuna_results['2018']['best_params']
best_params_2021 = optuna_results['2021']['best_params']


best_params_2018.update({
    'random_state': seed_2018,
})

best_params_2021.update({
    'random_state': seed_2021,
})

metrics_2018 = train_models(
    best_params_2018,
    X_2018,
    y_2018
)

metrics_2021 = train_models(
    best_params_2021,
    X_2021,
    y_2021
)

# Save detailed training/testing metrics to CSV
final_metrics_summary = [
    {
        'Dataset': '2018',

        'R2_Train': metrics_2018['r2_train'],
        'MAE_Train': metrics_2018['mae_train'],
        'RMSE_Train': metrics_2018['rmse_train'],
        'MAPE_Train': metrics_2018['mape_train'],

        'R2_Test': metrics_2018['r2'],
        'MAE_Test': metrics_2018['mae'],
        'RMSE_Test': metrics_2018['rmse'],
        'MAPE_Test': metrics_2018['mape'],

        'TrainTest_Split_Seed': best_params_2018['random_state']
    },

    {
        'Dataset': '2021',

        'R2_Train': metrics_2021['r2_train'],
        'MAE_Train': metrics_2021['mae_train'],
        'RMSE_Train': metrics_2021['rmse_train'],
        'MAPE_Train': metrics_2021['mape_train'],

        'R2_Test': metrics_2021['r2'],
        'MAE_Test': metrics_2021['mae'],
        'RMSE_Test': metrics_2021['rmse'],
        'MAPE_Test': metrics_2021['mape'],

        'TrainTest_Split_Seed': best_params_2021['random_state']
    }
]

df_final_metrics = pd.DataFrame(
    final_metrics_summary
)

final_metrics_csv_path = os.path.join(
    config.OUTPUT_DIR,
    'final_model_metrics.csv'
)

df_final_metrics.to_csv(
    final_metrics_csv_path,
    index=False
)

print(
    f"[SAVED] Final XGBoost metrics saved locally at: {final_metrics_csv_path}"
)

# Regression scatter plot (Observed vs Predicted)
dataset_metrics = {
    '2018': metrics_2018,
    '2021': metrics_2021
}

plot_regression(
    dataset_metrics,
    file_name='regression_final.png'
)
#%%
# SHAP Values and Beeswarm/Grid Plots 
print("\nCalculating SHAP values and generating visual interpretations...")

dataset_metrics_with_shap = shap_values(
    dataset_metrics
)

shap_dataframe_dict = {
    '2018': dataset_metrics_with_shap['2018']['shap'],
    '2021': dataset_metrics_with_shap['2021']['shap']
}

plot_shap_grid_simple(
    shap_dataframe_dict,
    top_n=10,
    file_name='2018_2021_shap_summary.png'
)

feature_count_dict = {
    '2018': len(X_2018.columns),
    '2021': len(X_2021.columns)
}

summary_plot(
    shap_dataframe_dict,
    feature_count_dict,
    layout='horizontal',
    file_name='beeswarm_summary_horizontal.png'
)

print("\n=========================================================================")

print(
    "[SUCCESS] PIPELINE COMPLETED SUCCESSFULLY!"
)

print(
    f"[FOLDER] All results, metrics, and plots have been saved at: "
    f"{config.OUTPUT_DIR}"
)

print(
    "========================================================================="
)