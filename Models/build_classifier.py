import pandas as pd
import pickle
from sklearn import ensemble, metrics
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


def build_classifier():
    df = pd.read_csv('clean_data.csv')
    df.dropna(inplace=True)
    df = df.astype('int32')
    features = list(df.columns)
    target = 'isDeceased'
    features.remove(target)
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    rf = ensemble.RandomForestClassifier(criterion='entropy', max_depth=10, max_features='log2', min_samples_split=50,
                                         n_estimators=500, n_jobs=4)
    model = Pipeline([('o', SMOTE(sampling_strategy=0.2)), ('u', RandomUnderSampler(sampling_strategy=0.5)), ('m', rf)])
    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)
    print(metrics.classification_report(y_test, y_test_pred))
    print(metrics.confusion_matrix(y_test, y_test_pred))
    with open('../model.pkl', 'wb') as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    build_classifier()
