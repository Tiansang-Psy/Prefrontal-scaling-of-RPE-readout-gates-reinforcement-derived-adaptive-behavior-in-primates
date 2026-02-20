import os
import numpy as np
import pandas as pd

import pickle
from joblib import Parallel, delayed
from itertools import combinations


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, KFold 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.inspection import permutation_importance
import os
import numpy as np
import pandas as pd

def train_svm_classification_balanced_best_parameters_and_all_data_kernel_choose(
    X, y, 
    svm_kernel,
    test_size=0.2, 
    cv=5, 
    random_state=42, 
    param_grid=None,
    scoring='f1',
    permutation_test=False    
):
    """
    Use SVM + class_weight='balanced' for binary classification to mitigate class imbalance.
    During hyperparameter search, class_weight=['balanced', None] etc. can also be further searched.
    
    Parameters:
      X, y: Feature matrix and target (0/1 or other binary classification labels)
      test_size: Test set proportion
      cv: Number of cross-validation folds
      random_state: Random seed
      param_grid: Hyperparameter grid (SVC parameters); if None, an example is provided
      scoring: Metric used for model selection (e.g., 'f1', 'balanced_accuracy', 'roc_auc', etc.)
      permutation_test: Whether to perform permutation test on training labels
      
    Returns:
      A dictionary containing model and prediction results, adding the best parameters, test set prediction results,
      and the prediction results for all samples using cross-validation.
    """

    # print(f"[Random seed]: {random_state}")

    if param_grid is None:
        if svm_kernel == 'rbf':
            param_grid = {
                'svc__kernel': ['rbf'],
                'svc__C': [0.1, 1, 10, 50, 100],
            }
        elif svm_kernel == 'linear':
            param_grid = {
                'svc__kernel': ['linear'],
                'svc__C': [0.1, 1, 10, 50, 100],
            }

    # Split dataset (can use stratify to maintain class distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        shuffle=True, 
        stratify=y,
        random_state=random_state
    )

    if permutation_test:
        y_train = np.random.permutation(y_train)

    # Build pipeline: Standardization + SVC(class_weight='balanced')
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(random_state=random_state, probability=True, class_weight='balanced'))
    ])

    # Grid search
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='recall',
        n_jobs=1  # Use 1 CPU per job
    )
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_  # Get the best parameters

    # Test set evaluation (only for the test set obtained from train_test_split)
    y_test_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # Output the predicted probability of the positive class (if ROC/PR curves are to be drawn)
    y_scores = best_model.predict_proba(X_test)[:, 1]

    # Feature importance
    feature_importance = None
    if best_model.named_steps['svc'].kernel == 'linear':
        # For linear kernel, use coefficients as feature importance
        feature_importance = best_model.named_steps['svc'].coef_[0]
    else:
        # For non-linear kernel, use permutation feature importance
        result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=random_state)
        feature_importance = result.importances_mean

    # Use cross_val_predict to perform cross-validation prediction on all data
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    all_data_pred = cross_val_predict(best_model, X, y, cv=kf, n_jobs=1)
    # If probability predictions are needed, do this (returns predicted positive class probabilities):
    # all_data_pred_proba = cross_val_predict(best_model, X, y, cv=kf, n_jobs=1, method='predict_proba')[:, 1]

    # # Output best parameters and prediction results for easy viewing
    # print("Best parameters:", best_params)
    # print("Test set prediction results:", y_test_pred)
    # print("Cross-validation prediction results for all data:", all_data_pred)

    return {
        'test_accuracy': test_acc,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'y_scores': y_scores,
        'feature_importance': feature_importance,
        'best_params': best_params,           # Return the best parameters
        'y_test_pred': y_test_pred,           # Return the test set prediction results
        'y_test': y_test,
        'all_data_cv_prediction': all_data_pred  # Return the cross-validation prediction results for all data
    }

if __name__ == '__main__':
    # Predict switch behavior of the next trial using the single-trial RPE beta-map. 
    # The predictors used in this code are masked by human group-level activation regions. 
    # Beta values are extracted from these activation regions within the single-trial RPE beta-map as features 
    # (activated voxels were filtered with a threshold of 20 after FDR-cluster correction, removing small clusters).

    # Parameters that can be modified
    # 1. LSS_file_name: Confirm feature information
    # 2. scoring: Evaluation metrics
    # 3. func_name: Function name -- helps identify which functions were called in the saved results
    # 4. repeat_times: Number of repeated runs
    # 5. permutation_test: Whether to perform a permutation test -- True/False. 
    #    True performs the test to check predictive power; False checks the model's null distribution.
    # 6. pkl_save_name: Saved filename (automatically generated based on parameters, no change needed)

    repeat_times = 100
    seeds_start = 1
    seeds_end = seeds_start + repeat_times
    permutation_test = False

    filename = 'human_regions_pred_switch_ROI_PCA'
    kernel = 'rbf' 

    LSS_merge_data_path = os.path.join('./Data/Fig5_RF_pred_part', filename)
    LSS_file_name = [f for f in os.listdir(LSS_merge_data_path) if f.startswith('PCA_')]

    func_name = 'Random_effect_train_svm_balanced'
    pkl_save_name = f'{func_name}_{kernel}_perm_{permutation_test}_seed_start_{seeds_start}_end{seeds_end}.pkl'
    feature_names_save_path = os.path.join(LSS_merge_data_path, 'Random_effect_check_feature_names.txt')

    target_col = "switch"
    columns_to_drop=['subj_id', 'S', 'A', 'R', 'test', 'is_switch_point', 'optimal_choice']

    # Start execution
    df = pd.read_csv(os.path.join(LSS_merge_data_path, LSS_file_name[0]))
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

    # X = df.drop(columns=target_col).values
    X_pre = df.drop(columns=target_col).values
    x_min, x_max = X_pre.min(), X_pre.max()
    df['PC_Random'] = np.random.uniform(low=x_min, high=x_max, size=len(df))
    X = df.drop(columns=target_col).values
    y = df[target_col].values

    # Get feature names
    feature_names = df.drop(columns=target_col).columns.tolist()
    # Save feature names
    with open(feature_names_save_path, 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")

    selected_features = ['PC_Random']

    # Accumulate each feature
    # for feature in feature_names:
    for i, feature in enumerate(feature_names):
        print(i)
        selected_features.append(feature)  # Add a new column
        if i == 0:
            used_features = 'PC1'
        else:
            used_features = f"PC1_to_PC{i+1}"

        print('used feature name: ', used_features)

        # Construct current data
        # X_selected_df = df[selected_features].copy()
        # X_selected_df['PC_Random'] = np.random.uniform(low=x_min, high=x_max, size=len(df))

        X_selected = df[selected_features].values  # Extract data for current features

        # Print current feature names being used
        print(f"Number of currently used features: {len(selected_features)}")
        print(f"Used feature list: {selected_features}")
        print(f"X shape: {X_selected.shape}")

        # Filename for saving results
        pkl_save_name = f'{func_name}_{kernel}_perm_{permutation_test}_seed_start_{seeds_start}_end{seeds_end}_use_{used_features}.pkl'
        pkl_save_path = os.path.join(LSS_merge_data_path, pkl_save_name)

        # Parallel processing
        results = Parallel(n_jobs=seeds_end-1)(delayed(train_svm_classification_balanced_best_parameters_and_all_data_kernel_choose)(
            X_selected, y,
            svm_kernel=kernel,
            test_size=0.2,
            cv=5,
            random_state=i,  # Iterate from 1 to 100
            permutation_test=permutation_test
        ) for i in range(seeds_start, seeds_end))

        with open(pkl_save_path, 'wb') as f:
            pickle.dump(results, f)

        print("Results saved to svm_results.pkl")