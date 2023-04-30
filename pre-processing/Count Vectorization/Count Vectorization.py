import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Get the absolute path of the current file
file_path = os.path.abspath(__file__)

# Extract the directory from the file path
dir_path = os.path.dirname(file_path)

# Specify the log file name
log_file_name = 'samplelog.log'

# Join the current directory with the log file name to get the full file path
log_file_path = os.path.join(dir_path, log_file_name)

# Read the log file
with open(log_file_path, 'r') as f:
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

# Specify the CSV file name
csv_file_name = 'logtable3.csv'

# Join the current directory with the CSV file name to get the full file path
csv_file_path = os.path.join(dir_path, csv_file_name)

# Write the DataFrame to a CSV file
df.to_csv(csv_file_path, index=False)
