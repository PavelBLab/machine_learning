import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 100)


def blight_model():

    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import roc_auc_score

    # Load Data
    df_train = pd.read_csv('train.csv', encoding='ISO-8859-1')
    # print(df_train.head())
    # print('======================')
    df_test = pd.read_csv('test.csv')
    # print(df_test.head())
    # print('======================')
    addresses = pd.read_csv('addresses.csv')
    # print(addresses.head())
    # print('======================')
    latlons = pd.read_csv('latlons.csv')
    # print(latlons.head())

    print(df_train.columns)

    # drop all rows with NaN compliance
    # print(df_train['compliance'])
    # print(np.isfinite(df_train['compliance']))
    df_train = df_train[np.isfinite(df_train['compliance'])]
    # print(df_train)

    # drop all rows not in the U.S
    df_train = df_train[df_train['country'] == 'USA']
    df_test = df_test[df_test['country'] == 'USA']
    # print(df_train)

    # merge latlons and addresses with data
    # print(pd.merge(addresses, latlons, on='address'))
    df_train = pd.merge(df_train, pd.merge(addresses, latlons, on='address'), on='ticket_id')
    # print(df_train.head())
    df_test = pd.merge(df_test, pd.merge(addresses, latlons, on='address'), on='ticket_id')
    # print(df_test.head())

    # drop all unnecessary columns
    # print(df_train.columns)
    # print()
    # print(df_test.columns)

    df_train.drop(['agency_name', 'inspector_name', 'violator_name', 'non_us_str_code', 'violation_description',
                'grafitti_status', 'state_fee', 'admin_fee', 'ticket_issued_date', 'hearing_date',
                # columns not available in test
                'payment_amount', 'balance_due', 'payment_date', 'payment_status',
                'collection_status', 'compliance_detail',
                # address related columns
                'violation_zip_code', 'country', 'address', 'violation_street_number',
                'violation_street_name', 'mailing_address_str_number', 'mailing_address_str_name',
                'city', 'state', 'zip_code', 'address'], axis=1, inplace=True)
    # print(df_train.columns)

    # сonvert categorical text data into model-understandable numerical data
    label_encoder = LabelEncoder()
    # print(df_train['disposition'].head())
    # print(df_train['disposition'].shape)
    label_encoder.fit(df_train['disposition'].append(df_test['disposition'], ignore_index=True))
    df_train['disposition'] = label_encoder.transform(df_train['disposition'])
    df_test['disposition'] = label_encoder.transform(df_test['disposition'])
    # print(df_train['disposition'].head())
    # print(df_train['disposition'].shape)

    label_encoder = LabelEncoder()
    label_encoder.fit(df_train['violation_code'].append(df_test['violation_code'], ignore_index=True))
    df_train['violation_code'] = label_encoder.transform(df_train['violation_code'])
    df_test['violation_code'] = label_encoder.transform(df_test['violation_code'])


    # Replace nan values to mean
    # print(df_train['lat'][(np.isnan(df_train['lat']))])
    df_train['lat'] = df_train['lat'].fillna(df_train['lat'].mean())
    df_train['lon'] = df_train['lon'].fillna(df_train['lon'].mean())
    # print(df_train['lat'].iloc[44113])
    # print(df_train['lat'].iloc[124107])

    df_test['lat'] = df_test['lat'].fillna(df_test['lat'].mean())
    df_test['lon'] = df_test['lon'].fillna(df_test['lon'].mean())
    df_train_columns_list = list(df_train.columns)
    # print(type(list(df_train.columns)))
    # print(type(list(df_train.columns.values)))
    # print(df_train_columns_list)
    df_train_columns_list.remove('compliance')
    df_test = df_test[df_train_columns_list]
    # print(len(df_train.columns))
    # print(len(df_test.columns))
    # print('admin_fee' in df_test.columns)
    # print('compliance' in df_test.columns)

    # Train the model
    X_train, X_test, y_train, y_test = train_test_split(df_train.ix[:, df_train.columns != 'compliance'], df_train['compliance'])
    random_forest_clf = RandomForestRegressor()

    # default metric to optimize over grid parameters: accuracy
    # grid_values = {'n_estimators': [10, 100], 'max_depth': [None, 30]}
    # grid_auc_clf = GridSearchCV(random_forest_clf, param_grid=grid_values, scoring='roc_auc')
    # grid_auc_clf.fit(X_train, y_train)
    # print('Grid best parameter (max. AUC): ', grid_auc_clf.best_params_)
    # print('Grid best score (AUC): ', grid_auc_clf.best_score_)
    # return pd.DataFrame(grid_auc_clf.predict(df_test), df_test.ticket_id)
blight_model()