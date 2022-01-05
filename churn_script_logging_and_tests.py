'''
Module contain logging and test for churn customer analysis
Author : Sara Selis
Date : 05/01/2022
'''

# import libaries
import churn_library as clib
import logging
import os

from math import ceil


# logging configuration
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')


def test_import():
    """Test import_data() function from the churn_library module

    Raises:
        error: FileNotFoundError
        error: AssertionError
    """
    logging.info('Testing if csv is accessible.')

    try:
        dataframe = clib.import_data("data/BankChurners.csv")
        logging.info("Successfully imported dataset.")
    except FileNotFoundError as error:
        logging.error("File not found.")
        raise error

    logging.info("Testing the dataset shape.")
    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
        logging.info(f'Rows: \
                     {dataframe.shape[0]}\tColumns: {dataframe.shape[1]}')
    except AssertionError as error:
        logging.error("The dataset does not have the correct shape.")
        raise error


def test_eda():
    '''
    Test perform_eda() function from the churn_library module
    '''
    """Test perform_eda() function from the churn_library module

    Raises:
        error: KeyError
        error: AssertionError
        error: AssertionError
        error: AssertionError
        error: AssertionError
        error: AssertionError
    """

    logging.info("Testing EDA")
    dataframe = clib.import_data("data/BankChurners.csv")

    try:
        clib.perform_eda(dataframe=dataframe)
        logging.info("Success importing.")
    except KeyError as error:
        logging.error(f'Column {error.args[0]} not found.')
        raise error

    logging.info("Assert if churn_distribution.png is created.")
    try:
        assert os.path.isfile("./images/eda/churn_distribution.png") is True
        logging.info('File churn_distribution.png found.')
    except AssertionError as error:
        logging.error('File not found.')
        raise error

    logging.info("Assert if customer_age_distribution.png is created.")
    try:
        assert os.path.isfile("./images/eda/customer_age_distribution.png") \
            is True
        logging.info('File customer_age_distribution.png was found.')
    except AssertionError as error:
        logging.error('File not found.')
        raise error

    logging.info("Assert if marital_status_distribution.png is created.")
    try:
        assert os.path.isfile("./images/eda/marital_status_distribution.png")\
            is True
        logging.info('File marital_status_distribution.png was found.')
    except AssertionError as error:
        logging.error('File not found.')
        raise error

    logging.info("Assert if total_transaction_distribution.png is created.")
    try:
        assert os.path.isfile("./images/eda/total_trans_dist.png") is True
        logging.info('File total_transaction_distribution.png was found.')
    except AssertionError as error:
        logging.error('File not found.')
        raise error

    logging.info("Assert if heatmap.png is created.")
    try:
        assert os.path.isfile("./images/eda/heatmap.png") is True
        logging.info('File heatmap.png was found.')

    except AssertionError as error:
        logging.error('File not found.')
        raise error


def test_encoder_helper():
    """Test encoder_helper() function from the churn_library module

    Raises:
        error: AssertionError
        error: AssertionError
        error: AssertionError
    """
    logging.info("Loading dataframe.")
    dataframe = clib.import_data("data/BankChurners.csv")

    logging.info("Creating Churn feature.")
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(lambda
                                                           val: 0 if val ==
                                                           "Existing Customer"
                                                           else 1)

    cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
                   'Income_Category', 'Card_Category']

    try:
        encoded_df = clib.encoder_helper(
            dataframe=dataframe,
            category_lst=[],
            response=None)

        logging.info('Verify if data is the same.')
        assert encoded_df.equals(dataframe) is True
        logging.info("Testing encoder_helper.")
    except AssertionError as error:
        logging.error("Testing encoder_helper.")
        raise error

    try:
        encoded_df = clib.encoder_helper(
            dataframe=dataframe,
            category_lst=cat_columns,
            response=None)

        logging.info("Verify if column names if the same.")
        assert encoded_df.columns.equals(dataframe.columns) is True

        assert encoded_df.equals(dataframe) is False
        logging.info("Testing encoder_helper.")
    except AssertionError as error:
        logging.error(
            "Testing encoder_helper.")
        raise error

    try:
        encoded_df = clib.encoder_helper(
            dataframe=dataframe,
            category_lst=cat_columns,
            response='Churn')

        logging.info("Verify if column names is different")
        assert encoded_df.columns.equals(dataframe.columns) is False

        # data should be different
        assert encoded_df.equals(dataframe) is False

        assert len(encoded_df.columns) == (len(dataframe.columns)
                                           + len(cat_columns))
        logging.info("Testing encoder_helper.")
    except AssertionError as error:
        logging.error("Testing encoder_helper.")
        raise error


def test_perform_feature_engineering():
    """Test perform_feature_engineering() functionfrom the churn_library module

    Raises:
        error: KeyError
        error: AssertionError
    """

    logging.info("Loading data.")
    dataframe = clib.import_data("data/BankChurners.csv")

    dataframe['Churn'] = dataframe['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    try:
        (_, X_test, _, _) = clib.perform_feature_engineering(
            dataframe=dataframe,
            response='Churn')

        logging.info("Verify if churn column is present in dataframe.")
        assert 'Churn' in dataframe.columns
        logging.info("Churn column is present.")
    except KeyError as error:
        logging.error('The Churn column is not present in the DataFrame.')
        raise error

    try:
        logging.info('Verify the size of X_test.')
        assert (X_test.shape[0] == ceil(dataframe.shape[0] * 0.3)) is True
        logging.info('DataFrame sizes are OK.')
    except AssertionError as error:
        logging.error('DataFrame sizes are not correct.')
        raise error


def test_train_models():
    """Test train_models() function from the churn_library module

    Raises:
        error: AssertionError
        error: AssertionError
        error: AssertionError
        error: AssertionError
        error: AssertionError
        error: AssertionError
    """

    dataframe = clib.import_data("data/BankChurners.csv")

    dataframe['Churn'] = dataframe['Attrition_Flag'].\
        apply(lambda val: 0 if val == "Existing Customer" else 1)

    # Feature engineering
    (X_train, X_test, y_train, y_test) = clib.perform_feature_engineering(
        dataframe=dataframe,
        response='Churn')

    logging.info("Assert if logistic_model.pkl file is present.")
    try:
        clib.train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile("./models/logistic_model.pkl") is True
        logging.info('Model logistic_model.pkl was found.')
    except AssertionError as error:
        logging.error('Not such model file.')
        raise error

    logging.info("Assert if rfc_model.pkl file is present.")
    try:
        assert os.path.isfile("./models/rfc_model.pkl") is True
        logging.info('Model rfc_model.pkl was found.')
    except AssertionError as error:
        logging.error('Not such model file found.')
        raise error

    logging.info("Assert if roc_curve_result.png file is present.")
    try:
        assert os.path.isfile('./images/results/roc_curve_result.png') is True
        logging.info('File roc_curve_result.png was found.')
    except AssertionError as error:
        logging.error('Not such file found.')
        raise error

    logging.info("Assert if rfc_results.png file is present.")
    try:
        assert os.path.isfile('./images/results/rf_results.png') is True
        logging.info('File rf_results.png was found.')
    except AssertionError as error:
        logging.error('Not such file found.')
        raise error

    logging.info("Assert if logistic_results.png file is present.")
    try:
        assert os.path.isfile('./images/results/logistic_results.png') is True
        logging.info('File logistic_results.png was found.')
    except AssertionError as error:
        logging.error('Not such file found.')
        raise error

    logging.info("Assert if feature_importances.png file is present.")
    try:
        assert os.path.isfile('./images/results/feature_importances.png')\
            is True
        logging.info('File feature_importances.png was found.')
    except AssertionError as error:
        logging.error('Not such file found.')
        raise error


if __name__ == "__main__":
    logging.info("Starting test pipeline")
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
