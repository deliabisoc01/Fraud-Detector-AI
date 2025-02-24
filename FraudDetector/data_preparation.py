import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data, is_file=True):
    # Check if 'data' is a file path or a DataFrame
    if is_file:
        # If 'data' is a file path, read the Excel file
        transactions_df = pd.read_excel(data, sheet_name='Transactions')
    else:
        # If 'data' is already a DataFrame, use it directly
        transactions_df = data

    # Preprocess the data
    transactions_df['DATE'] = pd.to_datetime(transactions_df['DATE'])
    transactions_df['TIME'] = pd.to_datetime(transactions_df['TIME'], format='%H:%M:%S').dt.time

    transactions_df['AMOUNT'] = transactions_df['AMOUNT'].replace({'RON': '', ' ': ''}, regex=True).astype(float)
    transactions_df['TRANSACTION_HOUR'] = pd.to_datetime(transactions_df['TIME'], format='%H:%M:%S').dt.hour
    transactions_df['DAY_OF_WEEK'] = transactions_df['DATE'].dt.day_name()
    transactions_df['IS_WEEKEND'] = transactions_df['DAY_OF_WEEK'].isin(['Saturday', 'Sunday']).astype(int)
    transactions_df['CUSTOMER_TRANSACTION_COUNT'] = transactions_df.groupby('ID_CUSTOMER')['ID_TRANSACTION'].transform('count')

    # Features used for prediction
    features = ['TRANSACTION_TYPE', 'ENTITY', 'TRANSACTION_ENTITY', 'AMOUNT', 
                'TRANSACTION_HOUR', 'IS_WEEKEND', 'CUSTOMER_TRANSACTION_COUNT']
    X = pd.get_dummies(transactions_df[features])

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X
