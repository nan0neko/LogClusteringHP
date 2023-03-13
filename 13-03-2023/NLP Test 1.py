import re
import csv
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

metadata_regex = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - [\w-]+: [\w-]+:'

log_regex = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - ([\w-]+): ([\w-]+): (.*)'

stop_words = set(stopwords.words('english'))

log_data = []

with open(r"C:\Users\toutu\Downloads\samplelog.log", 'r') as f:
    for line in f:
        # remove metadata
        line = re.sub(metadata_regex, '', line)
        # parse log message and extract relevant information
        match = re.match(log_regex, line)
        if match:
            timestamp = match.group(1)
            module = match.group(2)
            function = match.group(3)
            message = match.group(4)
            # tokenize message
            words = word_tokenize(message)
            # remove stop words
            words = [word for word in words if not word in stop_words]
            # add data to log_data list
            log_data.append([timestamp, module, function, words])

with open(r"C:\Users\toutu\Downloads\ifinfobase7.csv", 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # write header row
    writer.writerow(['Timestamp', 'Module', 'Function', 'Message'])
    # write data rows
    for data in log_data:
        writer.writerow(data)
