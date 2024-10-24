import dill
import pandas as pd
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def filter_data(df):
    columns_to_drop = ['id', 'url', 'region', 'region_url', 'price', 'manufacturer', 'image_url', 'description',
                       'posting_date', 'lat', 'long']
    return df.drop(columns_to_drop, axis=1)


def remove_outliers(df):
    q1 = df['year'].quantile(0.25)
    q3 = df['year'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Apply conditions without using numpy where
    df['year'] = df['year'].apply(lambda x: max(min(x, upper_bound), lower_bound))
    return df


def add_new_features(df):
    import pandas as pd
    df['short_model'] = df['model'].apply(lambda x: x.split(' ')[0] if pd.notnull(x) else x)
    df['age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return df


def main():
    df = pd.read_csv('data/homework.csv')
    X = df.drop(['price_category'], axis=1)
    y = df['price_category']

    filter_transformer = FunctionTransformer(filter_data)
    outliers_transformer = FunctionTransformer(remove_outliers)
    features_transformer = FunctionTransformer(add_new_features)

    numeric_features = ['odometer', 'year']
    numeric_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_features = ['model', 'fuel', 'title_status', 'transmission', 'state', 'short_model', 'age_category']
    categorical_transformer = Pipeline(steps=[
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    models = [
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    ]

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('filter', filter_transformer),
            ('remove_outliers', outliers_transformer),
            ('add_features', features_transformer),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:.4f}')

    best_pipe.fit(X, y)
    data_to_serialize = {
        'model': best_pipe,
        'metadata': {
            'name': 'Loan prediction model',
            'author': 'Tatev Karapetian',
            'version': 1,
            'date': datetime.now(),
            'type': type(best_pipe.named_steps["classifier"]).__name__,
            'accuracy': best_score
        }
    }

    # Serialize the data structure to a file
    with open('cars_pipe.pkl', 'wb') as f:
        dill.dump(data_to_serialize, f)

    print("The pipeline has been successfully saved with dill.")


if __name__ == '__main__':
    main()
