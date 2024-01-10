def calculate_metrics(data):
    ham = data['ham']
    ham_correct = data['ham_correct']
    spam = data['spam']
    spam_correct = data['spam_correct']

    true_positive = spam_correct
    false_positive = ham - ham_correct
    true_negative = ham_correct
    false_negative = spam - spam_correct

    true_positive_rate = true_positive / (true_positive + false_negative)
    true_negative_rate = true_negative / (true_negative + false_positive)

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive_rate

    f1_score = 2 * (precision * recall) / (precision + recall)

    return true_positive_rate, true_negative_rate, f1_score


# Example JSON data
json_data = {'ham': 750, 'ham_correct': 620, 'spam': 150, 'spam_correct': 130}

# Calculate metrics
tp_rate, tn_rate, f1 = calculate_metrics(json_data)

# Print the results
print(f'True Positive Rate: {tp_rate:.2%}')
print(f'True Negative Rate: {tn_rate:.2%}')
print(f'F1 Score: {f1:.2%}')

#17657
#19045

#36702