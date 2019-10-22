# im going to make functions that prepares stuff in here

def prep_iris(df):
    train, test = train_test_split(iris, train_size=.8, random_state=123)
    int_encoder = LabelEncoder()
    int_encoder.fit(train.species)
    train.species = int_encoder.transform(train.species)
    return train.species


def prep_titanic(df):
    
    dft.drop(columns=['deck'], inplace=True)
    dft.fillna(np.nan, inplace=True)
    train, test = train_test_split(dft, train_size=.8, random_state=123)
    
    imp_mode = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

    imp_mode.fit(train[['embarked']])

    train['embarked'] = imp_mode.transform(train[['embarked']])

    test['embarked'] = imp_mode.transform(test[['embarked']])
    
    int_encoder = LabelEncoder()
    int_encoder.fit(train.embarked)
    train.embarked = int_encoder.transform(train.embarked)
    
    embarked_array = np.array(train.embarked)
    embarked_array[0:5]
    
    embarked_array = embarked_array.reshape(len(embarked_array), 1)
    
    ohe = OneHotEncoder(sparse=False, categories='auto')
    
    embarked_ohe = ohe.fit_transform(embarked_array)
    embarked_ohe
    
    test.embarked = int_encoder.transform(test.embarked)
    
    embarked_array = np.array(test.embarked).reshape(len(test.embarked), 1)
    
    embarked_test_ohe = ohe.transform(embarked_array)
    
    
    train_age_fare = train[['age', 'fare']]
    test_age_fare = test[['age', 'fare']]
    scaler, train_age_fare_scaled, test_age_fare_scaled = split_scale.my_minmax_scaler(train_age_fare, test_age_fare)
    
    return embarked_test_ohe