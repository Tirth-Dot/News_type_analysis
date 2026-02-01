import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

def clean_text(text):
    """
    Clean text by converting to lowercase and removing special characters
    """
    text = str(text).lower()
    text = ''.join(char if char.isalnum() or char.isspace() else '' for char in text)
    text = ' '.join(text.split())
    return text

def train_and_save_model():
    print("Loading dataset...")
    try:
        df = pd.read_json('News_Category_Dataset_v3.json', lines=True)
    except FileNotFoundError:
        print("Error: News_Category_Dataset_v3.json not found!")
        return
    except ValueError:
        print("Error: Could not read JSON file. Check format.")
        return

    # Filter data
    print("Filtering data...")
    categories_to_keep = ['TECH', 'ENTERTAINMENT', 'POLITICS', 'BUSINESS']
    df_filtered = df[df['category'].isin(categories_to_keep)].copy()
    
    print(f"Filtered dataset size: {len(df_filtered)}")

    # Preprocess
    print("Preprocessing headlines...")
    df_filtered['cleaned_headline'] = df_filtered['headline'].apply(clean_text)

    # Vectorization
    print("Vectorizing...")
    vectorizer = CountVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(df_filtered['cleaned_headline'])
    y = df_filtered['category'].values

    # Train-test split (optional for final model, but good for validation check)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    print("Training model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class='multinomial',
        solver='lbfgs',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Retrain on full dataset for production
    print("Retraining on full dataset...")
    model.fit(X, y)

    # Save artifacts
    print("Saving artifacts...")
    with open('news_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('news_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("✓ Done! Saved 'news_model.pkl' and 'news_vectorizer.pkl'")

if __name__ == "__main__":
    train_and_save_model()
