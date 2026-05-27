import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os 
import config
import time
import optuna
import math
import xgboost as xgb
import lightgbm as lgb
import shap
import joblib
from sklearn.model_selection import (
    train_test_split,
    KFold, 
    cross_validate, 
    cross_val_score)
from sklearn.metrics import (
    r2_score, 
    mean_squared_error, 
    mean_absolute_error, 
    mean_absolute_percentage_error)

# --- Loading and Preparation ---
def load_file(csv_file):
    """
    Loads a local CSV file into a DataFrame and displays summary information.
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"[ERROR] Error loading file {csv_file}: {e}")
        return None

    print(f"\n[FILE] Loaded file: {csv_file}")
    print(f"[INFO] Rows: {df.shape[0]} | Columns: {df.shape[1]}")
    
    print("\n[INFO] First rows:")
    print(df.head())
    
    print("\n[INFO] Last rows:")
    print(df.tail())
    
    print("\n[INFO] DataFrame information:")
    df.info()

    return df

def aggregate_external_variables(df_base, csv_mdt_ind):
    """
    Aggregates external variables (MDT / IND / BUILD) into the base DataFrame
    through a merge operation using 'grid_id'.

    Also calculates TPI (Topographic Position Index)
    for different radii.
    """

    # 1. Load MDT / IND
    try:
        df_mdt_ind_ext = pd.read_csv(csv_mdt_ind)
    except Exception as e:
        print(f"[ERROR] Error loading external variables {csv_mdt_ind}: {e}")
        return df_base

    if 'ANO' in df_mdt_ind_ext.columns:
        df_mdt_ind_ext.drop(columns='ANO', inplace=True)

    print('\n' + '-=' * 50)
    print('[INFO] Final Variable Set: Internal and External')
    print('External -> NDBI | NDVI | TPI')
    print('-=' * 50)

    df_ext = df_mdt_ind_ext.copy()

    # 2. External dataset description
    print(f"\n[INFO] External Dataset Rows: {df_ext.shape[0]} | Columns: {df_ext.shape[1]}")

    print("\n[INFO] First rows of the external dataset:")
    print(df_ext.head())

    print("\n[INFO] External dataset information:")
    df_ext.info()

    # 3. Merge with internal dataset
    df = df_base.merge(df_ext, on='grid_id', how='left')

    # 4. TPI calculation (local DTM - MDT_mean across radii)
    local_col = 'DTM'
    radii = list(range(50, 501, 50))

    if local_col in df.columns:

        for radius in radii:
            col_mdt = f'MDT_mean_{radius}m'

            if col_mdt in df.columns:
                df[f'TPI_{radius}m'] = df[local_col] - df[col_mdt]
                df.drop(columns=col_mdt, inplace=True)

        print('\n[OK] Processing completed. TPIs calculated.')

    else:
        print(f"\n[WARNING] Column '{local_col}' not found in dataset. Skipping TPI calculation.")

    print('\n[INFO] Final Dataset Columns (Internal + External):')

    cols = list(df.columns)
    cols_per_row = 5

    for i in range(0, len(cols), cols_per_row):
        print(', '.join(cols[i:i + cols_per_row]))

    print('\n[INFO] First rows of the final dataset:')
    print(df.head())

    return df

def prepare_data(df):
    """
    Performs missing value cleaning, selection of relevant columns,
    and explicit variable renaming.
    """

    print("[CHECKING] Checking missing values:")
    print(df.isna().sum())

    # 1. Remove rows without target variable (LST_C)
    if 'LST_C' not in df.columns:
        raise ValueError("Column 'LST_C' (target variable) is not in the DataFrame.")

    df = df.dropna(subset=['LST_C']).copy()

    print(f"\n[INFO] Shape after removing NA values from LST_C: {df.shape}")

    # 2. Remove irrelevant identifier or temporal columns
    cols_to_drop = ['grid_id', 'ANO']

    print(f"\n[REMOVING] Removing columns: {cols_to_drop}")

    df = df.drop(
        columns=[c for c in cols_to_drop if c in df.columns],
        errors='ignore'
    )

    # 3. Replace NaN values with 0 in building height data (BH)
    if 'BH_masked_mean' in df.columns:
        print("\n[CLEANING] Replacing NaN values with 0 in 'BH_masked_mean'")

        df['BH_masked_mean'] = (
            df['BH_masked_mean']
            .fillna(0)
        )

    # 4. Replace NaN values with 1.0 in Sky View Factor (SVF)
    if 'SVF_mean_masked' in df.columns:
        print("\n[CLEANING] Replacing NaN values with 1.0 in 'SVF_mean_masked'")

        df['SVF_mean_masked'] = (
            df['SVF_mean_masked']
            .fillna(1.0)
        )

    # 5. Explicit column name mapping
    rename_map = {

        # --- Internal Variables ---
        'LST_C': 'LST',
        'DTM': 'ELE_mean',
        'Vegetação Rasteira': 'HERB_prop',
        'Vegetação Arborizada': 'TREE_prop',
        'Área Urbanizada': 'URB_prop',
        'NDVI': 'NDVI_mean',
        'NDBI': 'NDBI_mean',
        'NDWI': 'NDWI_mean',
        'slope': 'SLP_mean',
        'aspect': 'ASP_mean',
        "Corpo d' Água": 'WAT_prop',
        'Solo exposto': 'BARE_prop',
        'BH_masked_mean': 'BH_mean',
        'building_density': 'Bl_prop',
        'SVF_mean_masked': 'SVF_mean',

        # --- External Variables (50m to 500m) ---
        **{
            f'NDBI_mean_{r}m': f'NDBI_mean_{r}m'
            for r in range(50, 550, 50)
        },

        **{
            f'NDVI_mean_{r}m': f'NDVI_mean_{r}m'
            for r in range(50, 550, 50)
        },

        **{
            f'BH_mean_{r}m': f'BH_mean_{r}m'
            for r in range(50, 550, 50)
        },

        **{
            f'Building_prop_mean_{r}m': f'Bl_prop_{r}m'
            for r in range(50, 550, 50)
        },

        **{
            f'TPI_{r}m': f'TPI_{r}m'
            for r in range(50, 550, 50)
        }
    }

    df = df.rename(columns=rename_map)

    print("\n[INFO] Columns after renaming:")
    print(list(df.columns))

    # 6. Descriptive statistics
    print("\n[STATISTICS] Descriptive statistics of the dataset:")
    print(df.describe())

    print("\nDataset after initial data preparation:")
    print(df.head())

    return df

# --- Graph and Plot Visualization ---
def plot_distribution(df, ncols=5, file_name=None):
    """
    Generates histograms with KDE curves for all numeric variables
    and saves the plot.
    """

    features = df.select_dtypes(include=np.number).columns.tolist()
    n_features = len(features)
    nrows = int(np.ceil(n_features / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * 5, nrows * 4),
        sharex=False,
        sharey=False
    )

    axes = axes.flatten()

    for i, feature in enumerate(features):
        data = df[feature].dropna()

        axes[i].hist(
            data,
            bins=30,
            density=True,
            alpha=0.5,
            color='purple'
        )

        sns.kdeplot(
            data,
            ax=axes[i],
            linewidth=2,
            color='black'
        )

        axes[i].set_title(f"Density of {feature}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")

    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    if file_name is not None:
        save_path = os.path.join(config.OUTPUT_DIR, file_name)

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(f"[SAVED] Distribution plot saved at: {save_path}")

    plt.show()
    plt.close()


def smart_correlation_selection(df,  threshold, target_col='LST'):
    """
    Selects predictor variables by intelligently removing multicollinearity.
    
    For correlated pairs above the threshold, keeps the variable
    with the highest correlation with the target.
    """

    # 1. Compute correlation with LST
    correlations = (
        df.corr()[target_col]
        .abs()
        .drop(target_col, errors='ignore')
    )

    # 2. Sort features by correlation with target (descending)
    sorted_features = (
        correlations
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    selected_features = []

    # 3. Correlation threshold filtering
    for feature in sorted_features:

        keep = True

        for selected_feature in selected_features:

            correlation = df[feature].corr(
                df[selected_feature]
            )

            if abs(correlation) >= threshold:
                keep = False
                break

        if keep:
            selected_features.append(feature)

    return selected_features


def apply_selection_to_all(datasets_dict, threshold, target_col='LST'):
    """
    Applies feature selection to a dictionary of DataFrames
    (e.g., {'2018': df_18, '2021': df_21}).
    """
    results = {}

    for dataset_name, df in datasets_dict.items():

        print(f"--- Smart Correlation Selection: Dataset {dataset_name} ---")

        selected_features = smart_correlation_selection(
            df,
            threshold,
            target_col,
        )

        results[dataset_name] = selected_features

        print(f"Initially: {len(df.columns) - 1} features")
        print(f"Selected: {len(selected_features)} features")
        print(
            f"Removed: {len(df.columns) - 1 - len(selected_features)} features\n"
        )

    return results

def plot_relevance_composite(datasets_dict, threshold, target_col='LST',
                              cols=3, file_name='relevance_ranking_composite.png'):
    """
    Generates an elegant horizontal Pearson r plot
    for variables retained after correlation filtering.
    """

    dataset_names = list(datasets_dict.keys())
    n_plots = len(dataset_names)

    if n_plots == 0:
        print("Empty dataset dictionary.")
        return None

    actual_cols = min(n_plots, cols)
    n_rows = math.ceil(n_plots / actual_cols)

    fig, axes = plt.subplots(
        n_rows,
        actual_cols,
        figsize=(actual_cols * 6.5, n_rows * 7)
    )

    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    sns.set_style("white")

    for i, dataset_name in enumerate(dataset_names):

        ax = axes[i]

        # Selection of retained explanatory variables
        selected_features = smart_correlation_selection(
            datasets_dict[dataset_name],
            threshold,
            target_col,
        )

        # Pearson correlations
        correlations = (
            datasets_dict[dataset_name]
            .corr()[target_col][selected_features]
            .sort_values(ascending=True)
        )

        # Colors (Teal for positive, Red for negative)
        colors = [
            '#1abc9c' if value >= 0 else '#e74c3c'
            for value in correlations
        ]

        # Subtle grid lines
        ax.hlines(
            y=range(len(correlations)),
            xmin=-1,
            xmax=1,
            colors='#bdc3c7',
            linestyles=':',
            linewidth=1,
            alpha=0.8,
            zorder=0
        )

        correlations.plot(
            kind='barh',
            color=colors,
            edgecolor='none',
            width=0.7,
            ax=ax
        )

        if ax.containers:
            ax.bar_label(
                ax.containers[0],
                fmt='%.2f',
                padding=8,
                fontweight='bold',
                fontsize=9
            )

        ax.set_title(
            f'Relevance - {dataset_name}',
            fontsize=15,
            fontweight='bold',
            pad=15
        )

        ax.set_xlabel(
            'Pearson r',
            fontsize=11
        )

        sns.despine(
            ax=ax,
            left=True,
            bottom=True
        )

        ax.grid(
            axis='x',
            linestyle=':',
            alpha=0.4,
            color='#bdc3c7'
        )

        ax.axvline(
            0,
            color='#2c3e50',
            linewidth=1.1,
            alpha=0.6
        )

        max_value = correlations.max()
        min_value = correlations.min()

        ax.set_xlim(
            min_value - 0.15,
            max_value + 0.15
        )

    # Remove extra axes from the grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    save_path = os.path.join(
        config.OUTPUT_DIR,
        file_name
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"[SAVED] Composite Relevance Ranking saved at: {save_path}"
    )

    plt.show()
    plt.close()

    return file_name



def compare_models_with_kfold(df, kfold, random_seed_kfold, dataset_name):
    """
    Compares XGBoost and LightGBM using K-Fold cross-validation.
    
    Returns the metrics DataFrame for saving and the data matrices.
    """

    X = df.drop(columns=['LST'])
    y = df['LST']

    all_results = []

    scoring_metrics = {
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2',
        'mape': 'neg_mean_absolute_percentage_error'
    }

    models = {
        'XGB': xgb.XGBRegressor(
            random_state=random_seed_kfold,
            verbosity=0
        ),

        'LGB': lgb.LGBMRegressor(
            random_state=random_seed_kfold,
            verbosity=-1
        )
    }

    print(
        f"[TRAINING] Starting Model Comparison for Dataset {dataset_name}..."
    )

    for model_name, model_obj in models.items():

        cv_results = cross_validate(
            model_obj,
            X,
            y,
            scoring=scoring_metrics,
            cv=kfold,
            n_jobs=-1,
            return_train_score=True
        )

        result = {
            'Dataset': f'{dataset_name}',
            'Model': model_name,
            'Seed': random_seed_kfold,
            'KFold_Seed': random_seed_kfold,
            'N_Features': len(X.columns),

            # Test Metrics
            'RMSE': -cv_results['test_rmse'].mean(),
            'MAE': -cv_results['test_mae'].mean(),
            'R2': cv_results['test_r2'].mean(),
            'MAPE': -cv_results['test_mape'].mean() * 100,

            # Training Metrics
            'RMSE_Train': -cv_results['train_rmse'].mean(),
            'MAE_Train': -cv_results['train_mae'].mean(),
            'R2_Train': cv_results['train_r2'].mean(),
            'MAPE_Train': -cv_results['train_mape'].mean() * 100,
        }

        all_results.append(result)

    df_all = pd.DataFrame(all_results)

    # Display results in console
    print("\n" + "=" * 50)
    print(
        f"[STATISTICS] COMPARATIVE SUMMARY (CV MEAN) - DATASET: {dataset_name}"
    )

    print("     TRAINING (↑)")
    print("=" * 50)

    train_view = (
        df_all[
            ['Model', 'R2_Train', 'MAE_Train', 'RMSE_Train', 'MAPE_Train']
        ]
        .set_index('Model')
        .rename(
            columns={
                'R2_Train': 'R2',
                'MAE_Train': 'MAE',
                'RMSE_Train': 'RMSE',
                'MAPE_Train': 'MAPE'
            }
        )
        .round(4)
    )

    print(train_view)

    print("\n" + "-" * 50)
    print("     TEST (↓)")
    print("-" * 50)

    test_view = (
        df_all[
            ['Model', 'R2', 'MAE', 'RMSE', 'MAPE']
        ]
        .set_index('Model')
        .round(4)
    )

    print(test_view)

    return df_all, X, y

def optimizer_optuna(X, y):
    """
    Performs hyperparameter optimization with Optuna for XGBoost
    using 5-fold cross-validation.
    """

    dynamic_seed = int(time.time_ns() % (2**32))

    print(
        f"[SEED] Dynamic seed generated for Optuna: {dynamic_seed}"
    )

    def objective(trial):

        params = {
            'max_depth': trial.suggest_int(
                'max_depth', 4, 9
            ),

            'min_child_weight': trial.suggest_int(
                'min_child_weight', 2, 20
            ),

            'reg_alpha': trial.suggest_float(
                'reg_alpha', 1e-3, 10.0, log=True
            ),

            'reg_lambda': trial.suggest_float(
                'reg_lambda', 1e-3, 10.0, log=True
            ),

            'gamma': trial.suggest_float(
                'gamma', 0.1, 5
            ),

            'subsample': trial.suggest_float(
                'subsample', 0.6, 0.9
            ),

            'colsample_bytree': trial.suggest_float(
                'colsample_bytree', 0.6, 0.9
            ),

            'learning_rate': trial.suggest_float(
                'learning_rate', 0.01, 0.08, log=True
            ),

            'n_estimators': trial.suggest_int(
                'n_estimators', 800, 2500
            ),

            'random_state': dynamic_seed,
            'n_jobs': -1,
            'verbosity': 0
        }

        model = xgb.XGBRegressor(**params)

        kfold = KFold(
            n_splits=5,
            shuffle=True,
            random_state=dynamic_seed
        )

        # Optimize to maximize R²
        score = cross_val_score(
            model,
            X,
            y,
            cv=kfold,
            scoring='r2',
            n_jobs=-1
        )

        return score.mean()

    print(
        "[TRAINING] Starting Optuna Optimization (100 trials)..."
    )

    study = optuna.create_study(
        direction='maximize'
    )

    study.optimize(
        objective,
        n_trials=100
    )

    print("\n" + "=" * 30)
    print("[RESULTS] BEST PARAMETERS FOUND")
    print("=" * 30)

    print(
        f"Best Mean CV R²: {study.best_value:.4f}"
    )

    print(
        f"Used Seed: {dynamic_seed}"
    )

    print("Hyperparameters:")

    for key, value in study.best_params.items():
        print(f"  > {key}: {value}")

    return study, dynamic_seed


def train_models(best_params, X, y):
    """
    Splits the dataset into 80% training and 20% testing
    using the defined random seed.

    Trains the final XGBoost model, evaluates both subsets,
    and returns metrics and predictions.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=True,
        random_state=best_params['random_state']
    )

    print(
        f"[TRAINING] Training model on training set ({len(X_train)} samples)..."
    )

    model = xgb.XGBRegressor(**best_params)

    model.fit(
        X_train,
        y_train
    )

    # Prediction and metrics on training data
    y_pred_train = model.predict(X_train)

    r2_train = r2_score(
        y_train,
        y_pred_train
    )

    mae_train = mean_absolute_error(
        y_train,
        y_pred_train
    )

    rmse_train = np.sqrt(
        mean_squared_error(
            y_train,
            y_pred_train
        )
    )

    mape_train = (
        mean_absolute_percentage_error(
            y_train,
            y_pred_train
        ) * 100
    )

    # Prediction and metrics on testing data
    print(
        "[TRAINING] Evaluating predictions on testing set..."
    )

    y_pred_test = model.predict(X_test)

    r2_test = r2_score(
        y_test,
        y_pred_test
    )

    mae_test = mean_absolute_error(
        y_test,
        y_pred_test
    )

    rmse_test = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred_test
        )
    )

    mape_test = (
        mean_absolute_percentage_error(
            y_test,
            y_pred_test
        ) * 100
    )

    print("\n[OK] TRAINING RESULTS:")
    print(f"   > R2: {r2_train:.4f}")
    print(f"   > MAE: {mae_train:.4f} °C")
    print(f"   > RMSE: {rmse_train:.4f} °C")
    print(f"   > MAPE: {mape_train:.4f} %")

    print("\n[OK] TESTING RESULTS:")
    print(f"   > R2: {r2_test:.4f}")
    print(f"   > MAE: {mae_test:.4f} °C")
    print(f"   > RMSE: {rmse_test:.4f} °C")
    print(f"   > MAPE: {mape_test:.4f} %")

    metrics = {
        'model': model,

        'X_train': X_train,
        'y_train': y_train,

        'X_test': X_test,
        'y_test': y_test,

        'y_pred': y_pred_test,

        'r2': r2_test,
        'mae': mae_test,
        'rmse': rmse_test,
        'mape': mape_test,

        'r2_train': r2_train,
        'mae_train': mae_train,
        'rmse_train': rmse_train,
        'mape_train': mape_train
    }

    return metrics

def plot_regression(dataset_metrics, file_name='regression_final.png'):
    """
    Plots Observed vs Predicted values for XGBoost
    in side-by-side subplots for each dataset/year.
    """

    dataset_names = list(dataset_metrics.keys())

    fig, axes = plt.subplots(
        1,
        len(dataset_names),
        figsize=(6 * len(dataset_names), 5)
    )

    if len(dataset_names) == 1:
        axes = [axes]

    for i, dataset_name in enumerate(dataset_names):

        dataset_results = dataset_metrics[dataset_name]

        y_observed = dataset_results["y_test"]
        y_predicted = dataset_results["y_pred"]

        ax = axes[i]

        ax.scatter(
            y_observed,
            y_predicted,
            alpha=0.6,
            color='purple'
        )

        max_value = max(
            max(y_observed),
            max(y_predicted)
        )

        min_value = min(
            min(y_observed),
            min(y_predicted)
        )

        ax.plot(
            [min_value, max_value],
            [min_value, max_value],
            '--',
            color='black'
        )

        ax.set_title(
            f'XGBoost Performance ({dataset_name})',
            fontsize=13,
            fontweight='bold'
        )

        ax.set_xlabel(
            'Observed LST (°C)'
        )

        ax.set_ylabel(
            'Predicted LST (°C)'
        )

        ax.axis('equal')

        ax.grid(
            True,
            linestyle=':',
            alpha=0.6
        )

        ax.legend(
            title=(
                f"R2 = {dataset_results['r2']:.3f}\n"
                f"RMSE = {dataset_results['rmse']:.3f}°C\n"
                f"MAE = {dataset_results['mae']:.3f}°C"
            ),
            loc="upper left",
            frameon=True,
            title_fontsize=11
        )

    plt.tight_layout()

    if file_name is not None:

        save_path = os.path.join(
            config.OUTPUT_DIR,
            file_name
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

        print(
            f"[SAVED] Regression plot saved at: {save_path}"
        )

    plt.show()
    plt.close()

def shap_values(results):
    """
    Computes and locally saves SHAP values for each
    dataset/model using TreeExplainer.
    """

    for dataset_name in results.keys():

        print(
            f"\n[INFO] Calculating SHAP values for dataset {dataset_name}..."
        )

        dataset_results = results[dataset_name]

        # Create the explainer and generate SHAP values
        explainer = shap.TreeExplainer(
            dataset_results['model']
        )

        shap_values = explainer(
            dataset_results['X_test']
        )

        print(
            "[OK] SHAP values successfully calculated!"
        )

        # Save SHAP values locally
        shap_file_path = os.path.join(
            config.OUTPUT_DIR,
            f"shap_{dataset_name}.joblib"
        )

        joblib.dump(
            shap_values,
            shap_file_path
        )

        print(
            f"[SAVED] SHAP object saved at: {shap_file_path}"
        )

        # Store results in the dictionary
        dataset_results["shap"] = shap_values

    return results

def summary_plot(dataset_results, feature_counts,
                 layout='horizontal',
                 file_name='shap_summary_plot.png'):
    """
    Plots SHAP Beeswarm plots arranged side by side
    for each dataset/year.
    """

    dataset_names = list(dataset_results.keys())
    num_plots = len(dataset_names)

    if layout == 'vertical':
        fig, axes = plt.subplots(
            num_plots,
            1,
            figsize=(10, 6 * num_plots)
        )

    elif layout == 'horizontal':
        fig, axes = plt.subplots(
            1,
            num_plots,
            figsize=(7.5 * num_plots, 7)
        )

    else:
        raise ValueError(
            "Layout must be 'vertical' or 'horizontal'"
        )

    if num_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, dataset_name in enumerate(dataset_names):

        ax = axes[i]

        shap_values = dataset_results[dataset_name]

        if isinstance(shap_values, dict) and 'shap' in shap_values:
            shap_values = shap_values['shap']

        print(
            f"[INFO] Generating SHAP Beeswarm plot for dataset {dataset_name}..."
        )

        shap.plots.beeswarm(
            shap_values,
            max_display=feature_counts[dataset_name],
            show=False,
            ax=ax,
            plot_size=None
        )

        ax.set_title(
            f"Summary Beeswarm - {dataset_name}",
            fontsize=13,
            fontweight='bold'
        )

        if layout == 'horizontal' and ax != axes[0]:
            ax.set_ylabel('')

        if layout == 'vertical' and ax != axes[-1]:
            ax.set_xlabel('')

    plt.tight_layout()

    save_path = os.path.join(
        config.OUTPUT_DIR,
        file_name
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"[SAVED] Global SHAP Beeswarm saved at: {save_path}"
    )

    plt.show()
    plt.close()


def plot_shap_grid_simple(shap_dataframes,
                           top_n=10,
                           file_name='2018_2021_shap_summary.png'):
    """
    Generates a 2x2 grid containing the Global Importance
    Bar Plot and the Beeswarm impact distribution
    for 2018 and 2021.
    """

    scenarios = ['2018', '2021']

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 14)
    )

    for col, scenario in enumerate(scenarios):

        if scenario not in shap_dataframes:

            print(
                f"[WARNING] {scenario} not found in SHAP values"
            )

            continue

        shap_values = shap_dataframes[scenario]

        if isinstance(shap_values, dict) and 'shap' in shap_values:
            shap_values = shap_values['shap']

        # 1. Bar plot (Top)
        ax_bar = axes[0, col]

        shap.plots.bar(
            shap_values,
            max_display=top_n,
            show=False,
            ax=ax_bar
        )

        ax_bar.set_title(
            f"{scenario}\nGlobal Importance (Bar)",
            fontsize=14,
            fontweight='bold'
        )

        if col > 0:
            ax_bar.set_ylabel('')

        # 2. Beeswarm plot (Bottom)
        ax_beeswarm = axes[1, col]

        shap.plots.beeswarm(
            shap_values,
            max_display=top_n,
            show=False,
            ax=ax_beeswarm,
            plot_size=None
        )

        ax_beeswarm.set_title(
            f"{scenario}\nImpact Distribution (Beeswarm)",
            fontsize=14,
            fontweight='bold'
        )

        if col > 0:
            ax_beeswarm.set_ylabel('')

    plt.tight_layout()

    save_path = os.path.join(
        config.OUTPUT_DIR,
        file_name
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"[SAVED] SHAP 2x2 grid saved at: {save_path}"
    )

    plt.show()
    plt.close()