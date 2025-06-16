from sklearn.ensemble import GradientBoostingRegressor

def train_dummy_model():
    import numpy as np
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    data = {
        'resume_text': [
            "Experienced Python developer with ML knowledge",
            "Java backend engineer with 5 years experience",
            "Data scientist proficient in Python, TensorFlow, and ML",
            "Frontend engineer skilled in React and Node.js",
            "AI researcher with publications in top conferences",
        ],
        'years_experience': [3, 5, 4, 2, 6],
        'label_score': [0.8, 0.6, 0.95, 0.5, 0.98]
    }
    df = pd.DataFrame(data)

    tfidf = TfidfVectorizer(max_features=20)
    text_features = tfidf.fit_transform(df['resume_text']).toarray()
    scaler = StandardScaler()
    exp_features = scaler.fit_transform(df[['years_experience']])
    X = np.hstack([text_features, exp_features])
    y = df['label_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model, tfidf, scaler
