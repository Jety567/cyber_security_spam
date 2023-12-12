import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, render_template, request, jsonify

print("Start")
df = pd.read_csv('mail_data.csv')
df['Category'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

df['Message'] = df['Message'].str.lower()  # Convert to lowercase
df['Message'] = df['Message'].str.replace('[^\w\s]', '')  # Remove punctuation

# Step 3: Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X = tfidf_vectorizer.fit_transform(df['Message'])
y = df['Category']  # Assuming 'label' is the column containing 0 for ham and 1 for spam

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 6: Train the model
y_pred = model.predict(X_test)

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_email = request.form['email']

        # Preprocess the user input
        user_input = pd.Series([user_email])
        user_input = user_input.str.lower()
        user_input = user_input.str.replace('[^\w\s]', '')

        # Transform the user input using the same TF-IDF vectorizer
        user_input_transformed = tfidf_vectorizer.transform(user_input)

        # Make a prediction
        prediction = model.predict(user_input_transformed)

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)