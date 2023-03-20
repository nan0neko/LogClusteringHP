import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Read the log file
with open(r"C:\Users\toutu\Downloads\samplelog.log", 'r') as f:
    logs = f.readlines()

# Remove the timestamp from each log entry
logs = [log.split(' ', 2)[2] for log in logs]

# Initialize CountVectorizer
cv = CountVectorizer()

# Tokenize and count the words in each log entry
counts = cv.fit_transform(logs)

# Get the feature names
feature_names = sorted(cv.vocabulary_.keys(), key=lambda x: cv.vocabulary_[x])

# Convert the matrix of counts to a DataFrame
df = pd.DataFrame(counts.toarray(), columns=feature_names)

# Write the DataFrame to a CSV file
df.to_csv(r"C:\Users\toutu\Downloads\ifinfobase10.csv", index=False)
