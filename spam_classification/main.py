import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
import email

def train_model(df):
    df['labeled'] = df['labeled'].apply(lambda x: 1 if x else 0)

    df['content'] = df['content'].apply(lambda x: str(x))

    df['content'] = df['content'].str.lower()
    df['content'] = df['content'].str.replace('[^\w\s]', '')

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    X = tfidf_vectorizer.fit_transform(df['content'])
    y = df['labeled']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Step 5: Build the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    model.tfidf_vectorizer = tfidf_vectorizer

    # Step 6: Train the model
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    # Extract values from confusion matrix
    tn, fp, fn, tp = cm.ravel()

    # Calculate TPR, TNR, F1-Score, and other metrics
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    f1_score = 2 * tp / (2 * tp + fp + fn)

    labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    categories = ['0', '1']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # Print results
    print(f'True Positive Rate (TPR): {tpr:.2%}')
    print(f'True Negative Rate (TNR): {tnr:.2%}')
    print(f'F1 Score: {f1_score:.2%}')

    print(f"Accuracy: {accuracy:.2f}")

    report = classification_report(y_test, y_pred)

    print("Classification Report:")
    print(report)


    return model


def extract(directory_path, data, spam):
    for filename in os.listdir(directory_path):
        eml_path = os.path.join(directory_path, filename)
        with open(eml_path, 'r', encoding='utf-8', errors='ignore') as file:
            file_content = file.read().strip()
            msg = email.message_from_string(file_content)
            if msg.get_payload() == "":
                data['content'].append(file_content)
                data['Subject'].append("")
            else:
                data['content'].append(msg.get_payload())
                data['Subject'].append(msg['Subject'])
            data['labeled'].append(spam)

    return data


def get_data():
    data = {'Subject': [], 'content': [], 'labeled': []}

    directory_path = "/Users/patrickweber/PycharmProjects/spam_classification/spam"

    data = extract(directory_path, data, 1)

    directory_path = "/Users/patrickweber/PycharmProjects/spam_classification/ham"

    data = extract(directory_path, data, 0)

    df = pd.DataFrame(data)

    print(len(df))

    return df


if __name__ == '__main__':
    df = get_data()
    model = train_model(df)
