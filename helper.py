# EECS 445 - Fall 2024
# Project 1 - helper.py

import pandas as pd
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
#from sklearn.externals.joblib import Parallel, delayed
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import project1 as project1
from sklearn.linear_model import LogisticRegression

from matplotlib import pyplot as plt

def get_train_test_split() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], npt.NDArray[np.float64], npt.NDArray[np.int64], list[str]]:
    """
    This function performs the following steps:
    - Reads in the data from data/labels.csv and data/files/*.csv (keep only the first 2,500 examples)
    - Generates a feature vector for each example
    - Aggregates feature vectors into a feature matrix (features are sorted alphabetically by name)
    - Performs imputation and normalization with respect to the population
    
    After all these steps, it splits the data into 80% train and 20% test. 
    
    The binary labels take two values:
        -1: survivor
        +1: died in hospital
    
    Returns the features and labels for train and test sets, followed by the names of features.
    """
    print('Loading files from disk')
    path =''
    df_labels = pd.read_csv(path+'data/labels.csv')
    df_labels = df_labels[:2000]
    IDs = df_labels['RecordID'][:2000]
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        print(f'{path}data/files/{i}.csv')
        raw_data[i] = pd.read_csv(f'{path}data/files/{i}.csv')
    
    features = Parallel(n_jobs=16)(delayed(project1.generate_feature_vector)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features).sort_index(axis=1)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_labels['In-hospital_death'].values
    X = project1.impute_missing_values(X)
    X = project1.normalize_feature_matrix(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=3)
    print (X_train, y_train, X_test, y_test, feature_names)
    return X_train, y_train, X_test, y_test, feature_names


def get_challenge_data() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], list[str]]:
    """
    This function is similar to get_train_test_split, except that:
    - It reads in all 10,000 training examples
    - It does not return labels for the 2,000 examples in the heldout test set
    You should replace your preprocessing functions (generate_feature_vector, 
    impute_missing_values, normalize_feature_matrix) with updated versions for the challenge 
    """
    df_labels = pd.read_csv('data/labels.csv')
    df_labels = df_labels
    IDs = df_labels['RecordID']
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        raw_data[i] = pd.read_csv(f'data/files/{i}.csv')
    
    features = Parallel(n_jobs=16)(delayed(project1.generate_feature_vector)(df) for _, df in tqdm(raw_data.items(), desc='Generating feature vectors'))
    df_features = pd.DataFrame(features)
    feature_names = df_features.columns.tolist()
    X, y = df_features.values, df_labels['30-day_mortality'].values
    X = project1.impute_missing_values(X)
    X = project1.normalize_feature_matrix(X)
    
    # Separate categorical and numerical features
    numerical_features = df_features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df_features.select_dtypes(exclude=[np.number]).columns.tolist()
    
    return X[:10000], y[:10000], X[10000:], feature_names, numerical_features, categorical_features
    


def generate_challenge_labels(y_label: npt.NDArray[np.float64], y_score: npt.NDArray[np.float64], uniqname: str) -> None:
    """
    Takes in `y_label` and `y_score`, which are two list-like objects that contain 
    both the binary predictions and raw scores from your linear classifier.
    Outputs the prediction to {uniqname}.csv. 
    
    Please make sure that you do not change the order of the test examples in the heldout set 
    since we will this file to evaluate your classifier.
    """
    pd.DataFrame({'label': y_label, 'risk_score': y_score}).to_csv(uniqname + '.csv', index=False)

def report_feature_statistics(X_train: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """
    Report the name, mean value, and interquartile range (IQR) for each feature in the training set.

    Args:
        X_train: (N, d) matrix. Training feature matrix.
        feature_names: List of feature names.
    
    Returns:
        DataFrame containing the feature name, mean value, and IQR for each feature.
    """
    # Calculate mean values
    means = np.mean(X_train, axis=0)
    
    # Calculate IQR values
    Q1 = np.percentile(X_train, 25, axis=0)
    Q3 = np.percentile(X_train, 75, axis=0)
    IQR = Q3 - Q1
    
    # Create a DataFrame to store the results
    feature_stats = pd.DataFrame({
        'Feature Name': feature_names,
        'Mean Value': means,
        'Interquartile Range (IQR)': IQR
    })
    
    return feature_stats

def draw_table(data: pd.DataFrame):
    """
    Draw a table using matplotlib.

    Args:
        data: DataFrame containing the feature statistics.
    """
    fig, ax = plt.subplots(figsize=(12, 8))  # set size frame
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=data.values, colLabels=data.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(data.columns))))
    plt.show()

def find_best_C(X_train):

    # Assuming the following performance metrics you calculated
    performance_measures = ['accuracy', 'precision', 'f1-score', 'auroc', 'average_precision', 'sensitivity', 'specificity']
    results = []

    # Loop through each performance measure to find best C and penalty
    for metric in performance_measures:
        C_range = [10**i for i in range(-3, 4)]  # C values from 10^-3 to 10^3
        best_C, best_penalty = project1.select_param_logreg(X_train, y_train, metric=metric, C_range=C_range)
        
        # Calculate CV performance with best parameters
        clf = LogisticRegression(penalty=best_penalty, C=best_C, solver='liblinear', fit_intercept=False, random_state=42)
        mean_perf, (min_perf, max_perf) = project1.cv_performance(clf, X_train, y_train, metric=metric, k=5)
        
        results.append({
            'Performance Measure': metric,
            'Optimal C': best_C,
            'Regularization Penalty': best_penalty,
            'Mean CV Performance': mean_perf,
            'Min CV Performance': min_perf,
            'Max CV Performance': max_perf
        })

    # Convert results to DataFrame for better display
    results_df = pd.DataFrame(results)

    # Display the results
    print(results_df)
    
    plot_performance_measures(results_df)
    
def plot_performance_measures(results_df):
    """
    Plot the performance measures using matplotlib.

    Args:
        results_df: DataFrame containing the performance measures.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Extract data for plotting
    metrics = results_df['Performance Measure']
    mean_perf = results_df['Mean CV Performance']
    min_perf = results_df['Min CV Performance']
    max_perf = results_df['Max CV Performance']
    C_values = results_df['Optimal C']
    penalties = results_df['Regularization Penalty']

    # Create error bars
    error_bars = [mean_perf - min_perf, max_perf - mean_perf]

    # Plot the bar chart
    ax.bar(metrics, mean_perf, yerr=error_bars, capsize=5, color='skyblue', edgecolor='black')
    
    # Add labels and title
    ax.set_xlabel('Performance Measure')
    ax.set_ylabel('CV Performance')
    ax.set_title('Performance Measures with Mean (Min, Max) CV Performance')
    ax.set_xticklabels(metrics, rotation=45, ha='right')

    # Add text annotations for C and Penalty
    for i, (mean, C, penalty) in enumerate(zip(mean_perf, C_values, penalties)):
        ax.text(i, mean, f'C={C}\nPenalty={penalty}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()

# run get_train_test_split()
if __name__ == '__main__':
    X_train, y_train, X_test, y_test, feature_names = get_train_test_split()
    feature_stats = report_feature_statistics(X_train, feature_names)

    find_best_C(X_train)
    
