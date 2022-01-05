'''
Churn customer analysis
Author : Sara Selis
Date : 05/01/2022
'''

# import libraries
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(path: str) -> "pd.Dataframe":
    """Read a pandas Dataframe from a csv path

    Args:
        path: path of csv

    Returns:
        dataframe: pandas dataframe
    """

    return pd.read_csv(path)


def perform_eda(dataframe: 'pd.Dataframe') -> 'pd.Drataframe':
    """Perform Eda and save figures to a images folder

    Args:
        dataframe: pandas dataframe

    Returns:
        eda_data: eda dataframe
    """

    # Copy DataFrame
    eda_data = dataframe.copy(deep=True)

    # Churn
    eda_data['Churn'] = eda_data['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    # Churn Distribution
    plt.figure(figsize=(20, 10))
    eda_data['Churn'].hist()
    plt.savefig(fname='./images/eda/churn_distribution.png')

    # Customer Age Distribution
    plt.figure(figsize=(20, 10))
    eda_data['Customer_Age'].hist()
    plt.savefig(fname='./images/eda/customer_age_distribution.png')

    # Marital Status Distribution
    plt.figure(figsize=(20, 10))
    eda_data.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname='./images/eda/marital_status_distribution.png')

    # Total Transaction Distribution
    plt.figure(figsize=(20, 10))
    sns.histplot(eda_data['Total_Trans_Ct'], kde=True)
    plt.savefig(fname='./images/eda/total_trans_dist.png')

    # Heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(eda_data.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname='./images/eda/heatmap.png')

    return eda_data


def encoder_helper(dataframe: 'pd.Dataframe', category_lst: list,
                   response: list = None) -> 'pd.Dataframe':
    """Turn each categorical column into a new column with
        proportion of churn for each category -
        associated with cell 15 from the notebook

    Args:
        dataframe: eda dataframe
        category_lst: list that contains the category
        response, defalts to none

    Returns:
        encoder_df: encoder dataframe
    """

    encoder_df = dataframe.copy(deep=True)

    for category in category_lst:
        column_lst = []
        column_groups = dataframe.groupby(category).mean()['Churn']

        for val in dataframe[category]:
            column_lst.append(column_groups.loc[val])

        if response:
            encoder_df[category + '_' + response] = column_lst
        else:
            encoder_df[category] = column_lst

    return encoder_df


def perform_feature_engineering(
        dataframe: 'pd.Dataframe',
        response: list = None) -> tuple:
    '''
    input:
        data_frame: pandas DataFrame
        response: string of response name [optional
                    argument that could be used for
                    naming variables or index y column]
    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''
    """Perform feature engineering

    Args:
        dataframe: eda dataframe
        response, defalts to none

    Returns:
        Tuple that contains
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    """

    # categorical features
    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

    # feature engineering
    encoded_df = encoder_helper(
        dataframe=dataframe,
        category_lst=cat_columns,
        response=response)

    # predict feature
    y_df = encoded_df['Churn']

    # Create dataframe
    X_df = pd.DataFrame()

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X_df[keep_cols] = encoded_df[keep_cols]

    # Train and Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_df, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train: 'np.array',
                                y_test: 'np.array',
                                y_train_preds_lr: 'np.array',
                                y_train_preds_rf: 'np.array',
                                y_test_preds_lr: 'np.array',
                                y_test_preds_rf: 'np.array'):
    """Do classification report for training and test
        results and stores report in image folder.

    Args:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest
    """

    # RandomForestClassifier
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Random Forest Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Random Forest Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/rf_results.png')

    # LogisticRegression
    plt.rc('figure', figsize=(6, 6))
    plt.text(0.01, 1.25,
             str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6,
             str('Logistic Regression Test'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/logistic_results.png')


def feature_importance_plot(model, features: 'np.array', output_pth: str):
    """Creates and stores the feature importances in pth

    Args:
        model: serialized model with the feature_importances
        features: pandas dataframe with x values
        output_pth: path to storage
    """

    # Feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort Feature importances in descending order
    indices = np.argsort(importances)[::-1]

    names = [features.columns[i] for i in indices]

    plt.figure(figsize=(25, 15))

    # Create plot
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    plt.bar(range(features.shape[1]), importances[indices])
    plt.xticks(range(features.shape[1]), names, rotation=90)

    # Save the image
    plt.savefig(fname=output_pth + 'feature_importances.png')


def train_models(
        X_train: 'np.array',
        X_test: 'np.array',
        y_train: 'np.array',
        y_test: 'np.array'):
    """Train, store model results, images and scores.

    Args:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    """

    # RandomForestClassifier and LogisticRegression
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
    lrc = LogisticRegression(n_jobs=-1, max_iter=1000)

    # Parameters for Grid Search
    param_grid = {'n_estimators': [200, 500],
                  'max_features': ['auto', 'sqrt'],
                  'max_depth': [4, 5, 100],
                  'criterion': ['gini', 'entropy']}

    # Grid Search and fit for RandomForestClassifier
    cv_rfc = RandomizedSearchCV(rfc, param_grid)
    cv_rfc.fit(X_train, y_train)

    # LogisticRegression
    lrc.fit(X_train, y_train)

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Compute train and test predictions for RandomForestClassifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Compute train and test predictions for LogisticRegression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Compute ROC curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(lrc, X_test, y_test, ax=axis, alpha=0.8)
    plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=axis,
        alpha=0.8)
    plt.savefig(fname='./images/results/roc_curve_result.png')
    # plt.show()

    # Compute and results
    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)

    # Compute and feature importance
    feature_importance_plot(model=cv_rfc,
                            features=X_test,
                            output_pth='./images/results/')


if __name__ == '__main__':
    # Import data
    df = import_data('data/BankChurners.csv')

    # Perform EDA
    eda_df = perform_eda(dataframe=df)

    # Feature engineering
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        dataframe=eda_df, response='Churn')

    # Model training,prediction and evaluation
    train_models(X_train=x_train,
                 X_test=x_test,
                 y_train=y_train,
                 y_test=y_test)
