# EECS 445 - Fall 2024
# Project 1 - project1.py

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml

from helper import *
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

config = yaml.load(open("config.yaml"), Loader=yaml.SafeLoader)
seed = config["seed"]
np.random.seed(seed)


# Q1a
def generate_feature_vector(df: pd.DataFrame) -> dict[str, float]:
    """
    Reads a dataframe containing all measurements for a single patient
    within the first 48 hours of the ICU admission, and convert it into
    a feature vector.
    
    Args:
        df: dataframe with columns [Time, Variable, Value]
    
    Returns:
        a dictionary of format {feature_name: feature_value}
    """
    static_variables = config["static"]
    timeseries_variables = config["timeseries"]
    
    # 1. Replace unknown values with np.nan
    unknown_values = ['unknown', 'NA', 'n/a']  # Adjust based on your data
    df.replace(unknown_values, np.nan, inplace=True)
    
    # 2. Extract time-invariant features
    static = df.iloc[0:5]  # First 5 rows
    feature_dict = {}
    
    for var in static_variables:
        feature_dict[var] = static[static['Variable'] == var]['Value'].iloc[0]

    # 3. Extract max of time-varying features
    timeseries = df.iloc[5:]  # Remaining rows
    for var in timeseries_variables:
        max_value = timeseries[timeseries['Variable'] == var]['Value'].max()
        feature_dict[f'max_{var}'] = max_value

    return feature_dict



# Q1b
def impute_missing_values(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    For each feature column, impute missing values  (np.nan) with the
    population mean for that feature.

    Args:
        X: (N, d) matrix. X could contain missing values
    
    Returns:
        X: (N, d) matrix. X does not contain any missing values
    """
    # Create a copy of X to avoid modifying the original array
    X_imputed = X.copy()
    
    # Iterate through each column
    for i in range(X.shape[1]):  # X.shape[1] gives the number of features
        # Calculate the mean of the column, ignoring np.nan values
        mean_value = np.nanmean(X[:, i])
        
        # Impute missing values with the mean
        X_imputed[np.isnan(X[:, i]), i] = mean_value
        
    return X_imputed


# Q1c
def normalize_feature_matrix(X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    For each feature column, normalize all values to range [0, 1].

    Args:
        X: (N, d) matrix.
    
    Returns:
        X: (N, d) matrix. Values are normalized per column.
    """
    # TODO: implement this function according to spec
    # NOTE: sklearn.preprocessing.MinMaxScaler may be helpful
    # Create a MinMaxScaler instance
    scaler = MinMaxScaler()
    
    # Fit the scaler to X and transform X
    X_normalized = scaler.fit_transform(X)
    
    return X_normalized


def get_classifier(
    loss: str = "logistic",
    penalty: str | None = None,
    C: float = 1.0,
    class_weight: dict[int, float] | None = None,
    kernel: str = "rbf",
    gamma: float = 0.1,
) -> KernelRidge | LogisticRegression:
    """
    Return a classifier based on the given loss, penalty function
    and regularization parameter C.

    Args:
        loss: Specifies the loss function to use.
        penalty: The type of penalty for regularization (default: None).
        C: Regularization strength parameter (default: 1.0).
        class_weight: Weights associated with classes.
        kernel : Kernel type to be used in Kernel Ridge Regression. 
            Default is 'rbf'.
        gamma (float): Kernel coefficient (default: 0.1).
    Returns:
        A classifier based on the specified arguments.
    """
    # TODO (optional, but highly recommended): implement function based on docstring

    if loss == "logistic":
        return LogisticRegression(
            penalty=penalty, 
            C=C, 
            class_weight=class_weight, 
            solver="liblinear" if penalty == "l1" else "lbfgs"
        )
    elif loss == "kernel_ridge":
        return KernelRidge(alpha=1 / C, kernel=kernel, gamma=gamma)
    else:
        raise ValueError(f"Unknown loss function: {loss}")


def performance(y_true, y_pred, y_scores: None, metric, bootstrap=False, num_bootstraps=1000):
    # if bootstrap:
    #     medians = []
        
    #     # Bootstrapping process
    #     for _ in range(n_bootstraps):
    #         # Sample with replacement
    #         indices = resample(np.arange(len(y_true)), replace=True)
    #         y_true_sampled = y_true[indices]
    #         y_pred_sampled = y_pred[indices]
            
    #         # Check if the predictions are binary or continuous
    #         if len(np.unique(y_pred_sampled)) == 2:  # Binary case
    #             medians.append(accuracy_score(y_true_sampled, y_pred_sampled))  # Use any other metric as needed
    #         else:  # Continuous case (e.g., probabilities)
    #             medians.append(roc_auc_score(y_true_sampled, y_pred_sampled))  # Example: Use AUROC
    #     # Calculate the median and 95% CI
    #     median_performance = np.median(medians)
    #     lower_ci = np.percentile(medians, 2.5)
    #     upper_ci = np.percentile(medians, 97.5)
        
    #     return median_performance, (lower_ci, upper_ci)
    # else:
    #     # Calculate standard performance metrics
    #     accuracy = accuracy_score(y_true, y_pred)
    #     precision = precision_score(y_true, y_pred)
    #     f1 = f1_score(y_true, y_pred)
    #     auroc = roc_auc_score(y_true, y_pred)
    #     average_precision = average_precision_score(y_true, y_pred)
        
    #     return accuracy, precision, f1, auroc, average_precision
    # Confusion matrix
    # cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    # tn, fp, fn, tp = cm.ravel()
    
    # # Accuracy, Precision, F1-Score
    # accuracy = accuracy_score(y_true, y_pred)
    # precision = precision_score(y_true, y_pred, pos_label=1)
    # f1 = f1_score(y_true, y_pred, pos_label=1)
    
    # # AUROC and Average Precision
    # auroc = roc_auc_score(y_true, y_scores) if y_scores is not None else None
    # avg_precision = average_precision_score(y_true, y_scores) if y_scores is not None else None
    
    # # Sensitivity and Specificity
    # sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0  # True Positive Rate
    # specificity = tn / (tn + fp) if (tn + fp) != 0 else 0  # True Negative Rate
    
    # return {
    #     "accuracy": accuracy,
    #     "precision": precision,
    #     "f1_score": f1,
    #     "auroc": auroc,
    #     "average_precision": avg_precision,
    #     "sensitivity": sensitivity,
    #     "specificity": specificity
    # }
    # Define a helper function to compute a single performance metric
    def compute_metric(y_true, y_pred, metric):
        if metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            return precision_score(y_true, y_pred)
        elif metric == 'f1_score':
            return f1_score(y_true, y_pred)
        elif metric == 'auroc':
            return roc_auc_score(y_true, y_pred)
        elif metric == 'average_precision':
            return average_precision_score(y_true, y_pred)
        elif metric == 'sensitivity':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tp / (tp + fn)
        elif metric == 'specificity':
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)
        else:
            print(metric)
            raise ValueError("Unsupported metric")

    if bootstrap:
        return compute_metric(y_true, y_pred, metric)
        
    # Otherwise, perform bootstrapping
    bootstrapped_scores = []
    n_samples = len(y_true)
    for i in range(1000):
        # Sample with replacement
        indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Compute performance for the bootstrap sample
        score = compute_metric(y_true_boot, y_pred_boot, metric)
        bootstrapped_scores.append(score)

    # Compute the median and 95% confidence interval
    bootstrapped_scores = np.array(bootstrapped_scores)
    median = np.percentile(bootstrapped_scores, 50)
    ci_lower = np.percentile(bootstrapped_scores, 2.5)
    ci_upper = np.percentile(bootstrapped_scores, 97.5)

    return median, (ci_lower, ci_upper)


# Q2.1a
def cv_performance(
    clf: KernelRidge | LogisticRegression,
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Splits the data X and the labels y into k-folds and runs k-fold
    cross-validation: for each fold i in 1...k, trains a classifier on
    all the data except the ith fold, and tests on the ith fold.
    Calculates the k-fold cross-validation performance metric for classifier
    clf by averaging the performance across folds.
    
    Args:
        clf: an instance of a sklearn classifier
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) vector of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy'
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')
    
    Returns:
        a tuple containing (mean, min, max) 'cross-validation' performance across the k folds
    """
    # skf = StratifiedKFold(n_splits=k)
    # scores = []

    # for train_index, test_index in skf.split(X, y):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]

    #     # Fit the classifier
    #     clf.fit(X_train, y_train)

    #     # Make predictions
    #     y_pred = clf.predict(X_test)

    #     # Calculate the metric based on user choice
    #     if metric == "accuracy":
    #         score = accuracy_score(y_test, y_pred)
    #     elif metric == "precision":
    #         score = precision_score(y_test, y_pred)
    #     elif metric == "f1_score":
    #         score = f1_score(y_test, y_pred)
    #     elif metric == "auroc":
    #         score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    #     elif metric == "average_precision":
    #         score = average_precision_score(y_test, clf.predict_proba(X_test)[:, 1])
    #     elif metric == "sensitivity":
    #         tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    #         score = tp / (tp + fn)
    #     elif metric == "specificity":
    #         tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    #         score = tn / (tn + fp)
    #     else:
    #         raise ValueError("Unsupported metric")

    #     scores.append(score)

    # mean_score = np.mean(scores)
    # min_score = np.min(scores)
    # max_score = np.max(scores)

    # return (mean_score, min_score, max_score)
    # Initialize StratifiedKFold to keep class proportions the same across folds
    skf = StratifiedKFold(n_splits=k)
    
    metrics = []
    
    # Loop through each fold
    for train_idx, val_idx in skf.split(X, y):
        # Split the data into training and validation sets
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train the classifier on the training data
        clf.fit(X_train, y_train)
        
        # Get predictions and decision function scores for the validation data
        y_pred = clf.predict(X_val)
        y_scores = clf.decision_function(X_val)
        
        # Calculate the performance metric using the helper function
        perf_metrics = performance(y_val, y_pred, y_scores, metric, bootstrap=False)
        
        # Append the requested metric for this fold
        metrics.append(perf_metrics[metric])
    
    # Calculate the mean, min, and max performance across all folds
    mean_metric = np.mean(metrics)
    min_metric = np.min(metrics)
    max_metric = np.max(metrics)
    
    return mean_metric, min_metric, max_metric


# Q2.1b
def select_param_logreg(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    penalties: list[str] = ["l2", "l1"],
) -> tuple[float, str]:
    """
    Sweeps different settings for the hyperparameter of a logistic regression,
    calculating the k-fold CV performance for each setting on X, y.
    
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric for which to optimize (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision', 'sensitivity',
             and 'specificity')
        C_range: an array with C values to be searched over
        penalties: a list of strings specifying the type of regularization penalties to be searched over
    
    Returns:
        The hyperparameters for a logistic regression model that maximizes the
        average k-fold CV performance.
    """
    # TODO: Implement this function
    # NOTE: You should be using your cv_performance function here
    # to evaluate the performance of each logistic regression classifier

    best_C = None
    best_penalty = None
    best_score = -float('inf')  # Initialize to a very low value

    # Loop through each combination of C and penalty
    for penalty in penalties:
        for C in C_range:
            # Initialize the logistic regression model with the given C and penalty
            clf = LogisticRegression(penalty=penalty, C=C, solver='liblinear', fit_intercept=False, random_state=0)
            
            # Perform cross-validation using the cv_performance function
            mean_metric, _, _ = cv_performance(clf, X, y, metric=metric, k=k)
            
            # Track the best score and parameters
            if mean_metric > best_score:
                best_score = mean_metric
                best_C = C
                best_penalty = penalty

    return best_C, best_penalty


# Q4c
def select_param_RBF(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    metric: str = "accuracy",
    k: int = 5,
    C_range: list[float] = [],
    gamma_range: list[float] = [],
) -> tuple[float, float]:
    """
    Sweeps different settings for the hyperparameter of a RBF Kernel Ridge Regression,
    calculating the k-fold CV performance for each setting on X, y.
    
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: the number of folds (default=5)
        metric: the performance metric (default='accuracy',
             other options: 'precision', 'f1-score', 'auroc', 'average_precision',
             'sensitivity', and 'specificity')
        C_range: an array with C values to be searched over
        gamma_range: an array with gamma values to be searched over
    
    Returns:
        The parameter value for a RBF Kernel Ridge Regression that maximizes the
        average k-fold CV performance.
    """
    print(f"RBF Kernel Ridge Regression Model Hyperparameter Selection based on {metric}:")
    # TODO: Implement this function acording to the docstring
    # NOTE: This function should be very similar in structure to your implementation of select_param_logreg()
    best_C = None
    best_gamma = None
    best_score = -np.inf
    
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    for C in C_range:
        for gamma in gamma_range:
            scores = []
            model = KernelRidge(alpha=1/(2*C), kernel='rbf', gamma=gamma)
            
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # Fit the model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Convert continuous predictions to binary
                y_pred_binary = np.where(y_pred >= 0, 1, -1)
                
                # Calculate the desired metric
                if metric == 'accuracy':
                    score = accuracy_score(y_test, y_pred_binary)
                elif metric == 'precision':
                    score = precision_score(y_test, y_pred_binary, pos_label=1)
                elif metric == 'f1-score':
                    score = f1_score(y_test, y_pred_binary, pos_label=1)
                elif metric == 'auroc':
                    score = roc_auc_score(y_test, y_pred_binary)
                elif metric == 'average_precision':
                    score = average_precision_score(y_test, y_pred_binary)
                elif metric == 'sensitivity':
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
                    score = tp / (tp + fn) if (tp + fn) > 0 else 0
                elif metric == 'specificity':
                    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
                    score = tn / (tn + fp) if (tn + fp) > 0 else 0
                else:
                    raise ValueError(f"Metric {metric} not recognized.")
                
                scores.append(score)
            
            mean_score = np.mean(scores)
            print(f"C: {C}, gamma: {gamma}, Mean {metric}: {mean_score}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_C = C
                best_gamma = gamma

    return best_C, best_gamma


# Q2.1e
def plot_weight(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.int64],
    C_range: list[float],
    penalties: list[str],
) -> None:
    """
    The funcion takes training data X and labels y, plots the L0-norm
    (number of nonzero elements) of the coefficients learned by a classifier
    as a function of the C-values of the classifier, and saves the plot.
    Args:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
    
    Returns:
        None
    """

    print("Plotting the number of nonzero entries of the parameter vector as a function of C")
    # Initialize a dictionary to store the L0-norms for each penalty
    l0_norms = {penalty: [] for penalty in penalties}
    
    for penalty in penalties:
        # elements of norm0 should be the number of non-zero coefficients for a given setting of C
        for C in C_range:
            # TODO Initialize clf according to C and penalty
            clf = LogisticRegression(penalty=penalty, C=C, solver='liblinear')
            # TODO Fit classifier with X and y
            clf.fit(X, y)
            # TODO: Extract learned coefficients/weights from clf into w
            # Note: Refer to sklearn.linear_model.LogisticRegression documentation
            # for attribute containing coefficients/weights of the clf object
            w = clf.coef_[0]

            # TODO: Count number of nonzero coefficients/weights for setting of C
            #      and append count to norm0
            # Compute the L0-norm (number of non-zero elements in θ)
            l0_norm = np.sum(w != 0) # small tolerance to account for numerical precision
            l0_norms[penalty].append(l0_norm)


    # Plotting L0-norm ∥¯θ∥0 against log(C) for each penalty type
    for penalty in penalties:
        plt.plot(np.log10(C_range), l0_norms[penalty], label=f"Penalty: {penalty}")
    
    plt.xlabel('log(C)')
    plt.ylabel('L0-norm of θ (Number of non-zero elements)')
    plt.title('L0-norm of θ vs C for different regularization penalties')
    plt.legend()
    plt.savefig("l0_norm_vs_C.png")
    plt.show()


def main() -> None:
    print(f"Using Seed={seed}")
    # Read data
    # NOTE: READING IN THE DATA WILL NOT WORK UNTIL YOU HAVE FINISHED
    #       IMPLEMENTING generate_feature_vector, impute_missing_values AND normalize_feature_matrix
    X_train, y_train, X_test, y_test, feature_names = get_train_test_split()

    metric_list = [
        "accuracy",
        "precision",
        "f1_score",
        "auroc",
        "average_precision",
        "sensitivity",
        "specificity",
    ]

    # TODO: Questions 1, 2, 3, 4
    # NOTE: It is highly recomended that you create functions for each
    
    def test_impute_missing_values():
        print("Testing impute_missing_values")
        X_test = np.array([[1, 2], [np.nan, 3], [7, np.nan], [np.nan, 5]])
        imputed_X = impute_missing_values(X_test)
        print(imputed_X)
        
    def test_impliment_feature_matrix():
        print("Testing normalize_feature_matrix")
        X_test = np.array([[1, 200], [2, 300], [3, 400], [4, 500]])
        normalized_X = normalize_feature_matrix(X_test)
        print(normalized_X)
        
    def test_feature_statistics(X_train: npt.NDArray[np.float64], feature_names: list[str]) -> None:
        # Create a DataFrame from X_train for easier manipulation
        df = pd.DataFrame(X_train, columns=feature_names)

        # Calculate mean and IQR for each feature
        mean_values = df.mean()
        iqr_values = df.quantile(0.75) - df.quantile(0.25)

        # Create a summary DataFrame
        summary_df = pd.DataFrame({
            'Feature Name': feature_names,
            'Mean Value': mean_values,
            'IQR': iqr_values
        })

        # Sort the summary DataFrame alphabetically by feature name
        summary_df.sort_values(by='Feature Name', inplace=True)

        # Plotting the table
        fig, ax = plt.subplots(figsize=(10, len(summary_df) * 0.2))  # Adjust the figure size as needed
        ax.axis('tight')
        ax.axis('off')
        
        table_data = summary_df.values
        column_labels = summary_df.columns
        
        # Create the table
        table = ax.table(cellText=table_data, colLabels=column_labels, cellLoc = 'center', loc='center')
        
        # Adjust font size
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        
        # Adjust column width
        table.auto_set_column_width([0, 1, 2])
        
        # Title for the table
        plt.title("Feature Statistics: Mean and IQR", fontsize=14)
        
        # Show the plot
        plt.show()
        
    def test_question2PartC():
        # Define performance metrics to evaluate
        performance_metrics = [
            "accuracy",
            "precision",
            "f1_score",
            "auroc",
            "average_precision",
            "sensitivity",
            "specificity"
        ]

        # Store results in a list
        results = []

        # C_range to test
        C_range = [0.1, 1, 10]
        penalties = ["l1", "l2"]

        # Iterate over each performance metric
        for metric in performance_metrics:
            # Get the best parameters and cross-validation performance
            best_C, best_penalty = select_param_logreg(X_train, y_train, metric=metric, k=5, C_range=C_range, penalties=penalties)
            
            clf = LogisticRegression(C=best_C, penalty=best_penalty, solver='liblinear', max_iter=1000)
            mean_performance, min_performance, max_performance = cv_performance(clf, X_train, y_train, metric=metric, k=5)
            
            # Append the results
            results.append({
                "Performance Measure": metric,
                "Best C": best_C,
                "Regularization Penalty": best_penalty,
                "Mean Performance": mean_performance,
                "Min Performance": min_performance,
                "Max Performance": max_performance,
            })

        # Create a DataFrame for better visualization
        results_df = pd.DataFrame(results)

        # Display the results in tabular format
        print(results_df)
        
        # reduce the min max and med to 4 decimal places
        results_df = results_df.round(4)

        # Set the figure size for the plot
        fig, ax = plt.subplots(figsize=(10, 4))  # Set the size of the table

        # Hide the axes
        ax.axis('tight')
        ax.axis('off')

        # Create the table
        table = ax.table(cellText=results_df.values, colLabels=results_df.columns, cellLoc = 'center', loc='center')

        # Adjust the font size of the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # Adjust the column width
        table.scale(1.2, 1.2)

        # Show the table
        plt.title("Hyperparameter Selection Results", fontsize=14)
        plt.show()
        
    def test_question2PartD():
        # Define performance metrics to evaluate
        performance_metrics = [
            "accuracy",
            "precision",
            "f1_score",
            "auroc",
            "average_precision",
            "sensitivity",
            "specificity"
        ]

        # Store results in a list
        results = []

        # C_range to test
        C_range = [1]
        penalties = ["l1"]

        # # Iterate over each performance metric
        # for metric in performance_metrics:
        #     # Get the best parameters and cross-validation performance
        #     best_C, best_penalty = select_param_logreg(X_train, y_train, metric=metric, k=5, C_range=C_range, penalties=penalties)
            
        #     clf = LogisticRegression(C=best_C, penalty=best_penalty, solver='liblinear', max_iter=1000)
        #     mean_performance, min_performance, max_performance = cv_performance(clf, X_train, y_train, metric=metric, k=5)
            
        #     # Append the results
        #     results.append({
        #         "Performance Measure": metric,
        #         "Best C": best_C,
        #         "Regularization Penalty": best_penalty,
        #         "Mean Performance": mean_performance,
        #         "Min Performance": min_performance,
        #         "Max Performance": max_performance,
        #     })
        # # Assume optimal_C and optimal_penalty are derived from the previous analysis
        # # Create and train the logistic regression model
        # model = LogisticRegression(C=best_C, penalty=best_penalty, solver='liblinear')
        # model.fit(X_train, y_train)

        # # Make predictions on the test set
        # y_test_pred = model.predict(X_test)
        # y_test_proba = model.predict_proba(X_test)[:, 1]  # Probabilities for AUROC

        # # Calculate performance metrics with bootstrapping
        # median_accuracy, accuracy_ci = performance(y_test, y_test_pred, bootstrap=True)
        # median_precision, precision_ci = performance(y_test, y_test_pred, bootstrap=True)
        # median_f1, f1_ci = performance(y_test, y_test_pred, bootstrap=True)
        # median_auroc, auroc_ci = performance(y_test, y_test_proba, bootstrap=True)  # Use probabilities for AUROC
        # median_avg_precision, avg_precision_ci = performance(y_test, y_test_proba, bootstrap=True)  # Use probabilities for average precision

        # # Display the results
        # print("Performance Metrics on Test Set:")
        # # display C and penalty
        # print(f"Best C: {best_C}, Best Penalty: {best_penalty}")
        # print(f"Accuracy: {median_accuracy:.4f}, 95% CI: [{accuracy_ci[0]:.4f}, {accuracy_ci[1]:.4f}]")
        # print(f"Precision: {median_precision:.4f}, 95% CI: [{precision_ci[0]:.4f}, {precision_ci[1]:.4f}]")
        # print(f"F1 Score: {median_f1:.4f}, 95% CI: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
        # print(f"AUROC: {median_auroc:.4f}, 95% CI: [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]")
        # print(f"Average Precision: {median_avg_precision:.4f}, 95% CI: [{avg_precision_ci[0]:.4f}, {avg_precision_ci[1]:.4f}]")  
        
        # Predict labels on test data
        clf = LogisticRegression(C=1, penalty="l1", solver='liblinear', max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_scores = clf.decision_function(X_test)  # For metrics like AUROC

        # Store the results in a dictionary for easy reporting
        results = {}

        for metric in performance_metrics:
            if metric == 'auroc':
                # Use decision function scores for AUROC
                median_perf, ci = performance(y_test, y_scores, None, metric, bootstrap=True, num_bootstraps=1000)
            else:
                # Use predicted labels for other metrics
                median_perf, ci = performance(y_test, y_pred, None, metric, bootstrap=True, num_bootstraps=1000)
            
            # Store the median and 95% CI
            results[metric] = {'Median': median_perf, '95% CI': ci}

        # Print the results
        for metric, values in results.items():
            print(f"Metric: {metric}")
            print(f"  Median Performance: {values['Median']:.4f}")
            print(f"  95% Confidence Interval: ({values['95% CI'][0]:.4f}, {values['95% CI'][1]:.4f})")
            print() 

    def test_question2partF():
        # Define the L1-regularized logistic regression model
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)

        # Fit the model on the training data
        clf.fit(X_train, y_train)

        # Extract the learned coefficients (w = θ) as a 1D array
        coefficients = clf.coef_.flatten()
        
        # Find the indices of the 4 most positive and 4 most negative coefficients
        most_positive_indices = np.argsort(coefficients)[-4:]  # 4 largest
        most_negative_indices = np.argsort(coefficients)[:4]  # 4 smallest

        # Get the corresponding coefficient values
        most_positive_coefficients = coefficients[most_positive_indices]
        most_negative_coefficients = coefficients[most_negative_indices]
        
        # Assuming feature_names is a list of the feature names
        most_positive_features = [feature_names[i] for i in most_positive_indices]
        most_negative_features = [feature_names[i] for i in most_negative_indices]

        # Report the features and their coefficients
        for feature, coef in zip(most_positive_features, most_positive_coefficients):
            print(f"Feature: {feature}, Coefficient: {coef}")

        for feature, coef in zip(most_negative_features, most_negative_coefficients):
            print(f"Feature: {feature}, Coefficient: {coef}")
            
    def test_question3Partb():
        weights = [{1: wp, -1: wn} for wp in [1, 5, 10, 20, 30, 40, 50] for wn in [0.1, 0.5, 1]]
        clf = LogisticRegression(penalty='l2', C=1.0, class_weight=weights, solver='liblinear')
        clf.fit(X_train, y_train)
        
        metrics_list = []
        
        def evaluate_metrics(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred),
                'auroc': roc_auc_score(y_true, y_pred),
                'average_precision': average_precision_score(y_true, y_pred),
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
            }
            return metrics
        
        for _ in range(1000):
            # Create a bootstrapped sample from the test set
            X_test_resampled, y_test_resampled = resample(X_test, y_test)
            
            # Predict on the resampled test set
            y_pred_resampled = clf.predict(X_test_resampled)
            
            # Evaluate and store the metrics for this bootstrap sample
            metrics_list.append(evaluate_metrics(y_test_resampled, y_pred_resampled))

        # 5. Convert metrics_list into a dictionary of arrays for easier calculation
        metrics_array = {metric: np.array([m[metric] for m in metrics_list]) for metric in metrics_list[0].keys()}
                
        # 6. Calculate median and confidence intervals
        def calculate_confidence_intervals(metric_values):
            median = np.median(metric_values)
            lower = np.percentile(metric_values, 2.5)
            upper = np.percentile(metric_values, 97.5)
            return median, lower, upper

        # 7. Print out the results for each metric
        for metric, values in metrics_array.items():
            median, lower_ci, upper_ci = calculate_confidence_intervals(values)
            print(f"{metric.capitalize()}: Median = {median:.4f}, 95% CI = ({lower_ci:.4f}, {upper_ci:.4f})")
            
    def test_question32PartA():
        # Class weights to test
        class_weights_options = [
            {1: 1, -1: 1},  # No weighting
            {1: 2, -1: 1},
            {1: 4, -1: 1},
            {1: 5, -1: 1},
            {1: 10, -1: 1},
            {1: 50, -1: 1}
        ]

        # Prepare to collect results
        results = {}

        # Perform Stratified K-Fold Cross Validation
        skf = StratifiedKFold(n_splits=5)

        for class_weights in class_weights_options:
            precision_list = []
            recall_list = []
            
            for train_index, test_index in skf.split(X_train, y_train):
                X_train_fold, X_val_fold = X_train[train_index], X_train[test_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]
                
                # Train model
                model = LogisticRegression(class_weight=class_weights, C=1.0, penalty='l2')
                model.fit(X_train_fold, y_train_fold)
                
                # Validate model
                y_pred = model.predict(X_val_fold)
                
                # Calculate metrics
                report = classification_report(y_val_fold, y_pred, output_dict=True)
                print(report)  # Print the report for debugging

                # Extract weighted average precision and recall
                precision = report['macro avg']['precision']
                recall = report['macro avg']['recall']

                # Debug prints to check values
                print(f"Weighted Precision: {precision}, Weighted Recall: {recall}")
        
                precision_list.append(precision)
                recall_list.append(recall)
            
            # Store results
            results[str(class_weights)] = {
                'mean_precision': np.mean(precision_list),
                'mean_recall': np.mean(recall_list)
            }

        # Print results for comparison
        for class_weights, metrics in results.items():
            print(f'Class Weights: {class_weights}, Mean Precision: {metrics["mean_precision"]:.4f}, Mean Recall: {metrics["mean_recall"]:.4f}')
   
    def test_question32PartB():
        clf = LogisticRegression(penalty='l2', C=1.0, class_weight={1: 5, -1: 1}, solver='liblinear')
        clf.fit(X_train, y_train)
        
        metrics_list = []
        
        def evaluate_metrics(y_true, y_pred):
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'f1_score': f1_score(y_true, y_pred),
                'auroc': roc_auc_score(y_true, y_pred),
                'average_precision': average_precision_score(y_true, y_pred),
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
            }
            return metrics
        
        for _ in range(1000):
            # Create a bootstrapped sample from the test set
            X_test_resampled, y_test_resampled = resample(X_test, y_test)
            
            # Predict on the resampled test set
            y_pred_resampled = clf.predict(X_test_resampled)
            
            # Evaluate and store the metrics for this bootstrap sample
            metrics_list.append(evaluate_metrics(y_test_resampled, y_pred_resampled))

        # 5. Convert metrics_list into a dictionary of arrays for easier calculation
        metrics_array = {metric: np.array([m[metric] for m in metrics_list]) for metric in metrics_list[0].keys()}
                
        # 6. Calculate median and confidence intervals
        def calculate_confidence_intervals(metric_values):
            median = np.median(metric_values)
            lower = np.percentile(metric_values, 2.5)
            upper = np.percentile(metric_values, 97.5)
            return median, lower, upper

        # 7. Print out the results for each metric
        for metric, values in metrics_array.items():
            median, lower_ci, upper_ci = calculate_confidence_intervals(values)
            print(f"{metric.capitalize()}: Median = {median:.4f}, 95% CI = ({lower_ci:.4f}, {upper_ci:.4f})")
    
        # Read challenge data
   
    def test_question33PartA():
        # Train model with class weights {1: 1, -1: 1}
        model_1 = LogisticRegression(C=1.0, class_weight={1: 1, -1: 1})
        model_1.fit(X_train, y_train)
        y_scores_1 = model_1.predict_proba(X_test)[:, 1]

        # Train model with class weights {1: 5, -1: 1}
        model_2 = LogisticRegression(C=1.0, class_weight={1: 5, -1: 1})
        model_2.fit(X_train, y_train)
        y_scores_2 = model_2.predict_proba(X_test)[:, 1]

        # Calculate ROC curve and AUC for both models
        fpr_1, tpr_1, _ = roc_curve(y_test, y_scores_1)
        roc_auc_1 = auc(fpr_1, tpr_1)

        fpr_2, tpr_2, _ = roc_curve(y_test, y_scores_2)
        roc_auc_2 = auc(fpr_2, tpr_2)

        # Plot ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr_1, tpr_1, color='blue', label=f'Wn=1, Wp=1 (AUC = {roc_auc_1:.2f})')
        plt.plot(fpr_2, tpr_2, color='red', label=f'Wn=1, Wp=5 (AUC = {roc_auc_2:.2f})')
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')  # Diagonal line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
    
    def test_question4PartB():
        # Constants
        C = 1.0
        alpha = 1 / (2 * C)
        seed = 42

        # Train Logistic Regression model
        logistic_model = LogisticRegression(penalty='l2', C=C, fit_intercept=False, random_state=seed)
        logistic_model.fit(X_train, y_train)

        # Train Kernel Ridge model
        kernel_model = KernelRidge(alpha=alpha, kernel='linear')
        kernel_model.fit(X_train, y_train)

        # Predictions
        y_pred_logistic = logistic_model.predict(X_test)
        y_pred_kernel_continuous = kernel_model.predict(X_test)
        y_pred_kernel = np.where(y_pred_kernel_continuous >= 0, 1, -1)  # Thresholding

        # Calculate performance metrics
        def calculate_metrics(y_true, y_pred):
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, pos_label=1)
            f1 = f1_score(y_true, y_pred, pos_label=1)
            auroc = roc_auc_score(y_true, y_pred)
            avg_precision = average_precision_score(y_true, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            return accuracy, precision, f1, auroc, avg_precision, sensitivity, specificity

        # Performance for Logistic Regression
        logistic_metrics = calculate_metrics(y_test, y_pred_logistic)

        # Performance for Kernel Ridge
        kernel_metrics = calculate_metrics(y_test, y_pred_kernel)

        # Create a results table
        results = pd.DataFrame({
            'Metric': ['Accuracy', 'Precision', 'F1 Score', 'AUROC', 'Average Precision', 'Sensitivity', 'Specificity'],
            'Logistic Regression': logistic_metrics,
            'Kernel Ridge': kernel_metrics,
        })

        print(results)
        
    def cross_val_auroc_rbf(X: npt.NDArray[np.float64], y: npt.NDArray[np.int64], C: float, gamma_values: list) -> list:
        """
        Calculate cross-validation AUROC performance for RBF Kernel Ridge Regression.

        Args:
            X: (n, d) array of feature vectors.
            y: (n,) array of binary labels {1, -1}.
            C: Regularization parameter for Ridge Regression.
            gamma_values: List of gamma values to evaluate.

        Returns:
            A list of tuples containing (gamma, mean AUROC, max AUROC, min AUROC).
        """
        results = []
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for gamma in gamma_values:
            model = KernelRidge(alpha=1/(2 * C), kernel='rbf', gamma=gamma)
            auroc_scores = []

            for train_index, test_index in kf.split(X, y):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Since y_pred is continuous, we can compute AUROC directly
                score = roc_auc_score(y_test, y_pred)
                auroc_scores.append(score)

            mean_auroc = np.mean(auroc_scores)
            max_auroc = np.max(auroc_scores)
            min_auroc = np.min(auroc_scores)
            results.append((gamma, mean_auroc, max_auroc, min_auroc))

        return results

    def test_question4PartD():
        # Example usage:
        C = 1.0
        gamma_values = [0.001, 0.01, 0.1, 1, 10, 100]
        results = cross_val_auroc_rbf(X_train, y_train, C, gamma_values)

        # Displaying results in tabular format
        print(f"{'Gamma':<10} {'Mean AUROC':<15} {'Max AUROC':<15} {'Min AUROC':<15}")
        for gamma, mean_auroc, max_auroc, min_auroc in results:
            print(f"{gamma:<10} {mean_auroc:<15.4f} {max_auroc:<15.4f} {min_auroc:<15.4f}")
            
    def test_question4PartE():
        metrics = {
            "accuracy": [],
            "precision": [],
            "f1_score": [],
            "auroc": [],
            "average_precision": [],
            "sensitivity": [],
            "specificity": []
        }
        
        C_range = [0.01, 0.1, 1.0, 10, 100]
        gamma_range = [0.01, 0.1, 1, 10]
        
        for C in C_range:
            for gamma in gamma_range:
                # Train the Kernel Ridge model
                model = KernelRidge(alpha=1/(2*C), kernel='rbf', gamma=gamma)
                model.fit(X_train, y_train)
                
                # Get predictions for test set
                y_pred_prob = model.predict(X_test)
                y_pred = np.where(y_pred_prob >= 0, 1, -1)

        for _ in range(1000):
            # Bootstrap sampling
            indices = resample(np.arange(len(y_test)), replace=True)
            y_true_sample = y_test[indices]
            y_pred_sample = y_pred[indices]
            
            # Calculate metrics
            metrics["accuracy"].append(accuracy_score(y_true_sample, y_pred_sample))
            metrics["precision"].append(precision_score(y_true_sample, y_pred_sample, pos_label=1))
            metrics["f1_score"].append(f1_score(y_true_sample, y_pred_sample, pos_label=1))
            metrics["auroc"].append(roc_auc_score(y_true_sample, y_pred_prob[indices]))
            metrics["average_precision"].append(average_precision_score(y_true_sample, y_pred_prob[indices]))
            
            tn, fp, fn, tp = confusion_matrix(y_true_sample, y_pred_sample).ravel()
            metrics["sensitivity"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            metrics["specificity"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)

        # Calculate median and 95% CI
        results = {metric: (np.median(values), np.percentile(values, [2.5, 97.5])) for metric, values in metrics.items()}
        print(results)
        return results

        # Compute metrics on test data with bootstrapping
        results = compute_metrics_with_bootstrap(y_test, y_pred)

        # Print results
        performance_df = pd.DataFrame(results, index=["Median Performance", "95% CI Lower", "95% CI Upper"]).T
        print(performance_df)
    # TODO: Question 5: Apply a classifier to heldout features, and then use
    #       generate_challenge_labels to print the predicted labels

    def challenge(X_challenge, y_challenge, X_test, feautre_names, numerical_features, categorical_features):
        from sklearn.model_selection import cross_val_score
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        
        # # Create a column transformer for preprocessing
        # preprocessor = ColumnTransformer(
        #     transformers=[
        #         ('num', StandardScaler(), numerical_features),  # For numerical features
        #         ('cat', OneHotEncoder(), categorical_features)    # For categorical features
        #     ]
        # )

        # # Create a logistic regression model
        # clf = Pipeline(steps=[
        #     ('preprocessor', preprocessor),
        #     ('classifier', LogisticRegression(max_iter=1000))
        # ])

        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=1000)
        # Step 4: Train the model on the challenge data using
        clf.fit(X_challenge, y_challenge)

        # Step 5: Predict on held-out set
        y_label = clf.predict(X_heldout)
        y_label = np.clip(y_label, -1, 1).astype(int)  # Ensure values are -1, 0, or 1
        y_score = clf.decision_function(X_heldout)  # For risk scores (continuous values)

        # Step 6: Save the output for submission
        uniqname = 'marshrey'  # Replace with your uniqname
        generate_challenge_labels(y_label, y_score, uniqname)

        # Step 7: Confusion Matrix for the challenge data
        y_pred_challenge = clf.predict(X_challenge)
        cm = confusion_matrix(y_challenge, y_pred_challenge)
        print("Confusion Matrix:\n", cm)

        # Step 8: Calculate AUROC on the challenge data
        auc = roc_auc_score(y_challenge, clf.decision_function(X_challenge))
        print(f"AUROC on the challenge data: {auc:.4f}")

        return clf
        
    
    # test_impute_missing_values()
    # test_impliment_feature_matrix()
    # test_feature_statistics(X_train, feature_names)
    # test_question2PartC()
    # test_question2PartD()
    # plot_weight(X_train, y_train, [0.1, 1, 10], ["l1", "l2"])
    # test_question2partF()
    # test_question3Partb()
    # test_question32PartA()
    # test_question32PartB()
    # test_question33PartA()
    # test_question4PartB()
    # test_question4PartD()
    test_question4PartE()
    # X_challenge, y_challenge, X_heldout, feature_names, num_fea, cat_fea = get_challenge_data()
    # challenge(X_challenge, y_challenge, X_heldout, feature_names, num_fea, cat_fea)
    
    
    


if __name__ == "__main__":
    main()
