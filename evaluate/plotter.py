import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# Given data
data = {'ham': 750, 'ham_correct': 620, 'spam': 150, 'spam_correct': 130}

# Extract values
ham_total = data['ham']
ham_correct = data['ham_correct']
spam_total = data['spam']
spam_correct = data['spam_correct']

# Calculate confusion matrix
cm = confusion_matrix(['Ham'] * ham_correct + ['Spam'] * spam_correct +
                      ['Ham'] * (ham_total - ham_correct) + ['Spam'] * (spam_total - spam_correct),
                      ['Ham'] * ham_total + ['Spam'] * spam_total)

print(cm)

# Convert to DataFrame for visualization
labels = ['Ham', 'Spam']
df_cm = pd.DataFrame(cm, index=labels, columns=labels)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
