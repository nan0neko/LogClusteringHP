import csv
import re
import os

header = ["Date", "Time","Software/Module", "User","Type_Log","Interface_Description", "Data"]
rows = []

# get the absolute path of the current file
file_path = os.path.abspath(__file__)

# extract the directory from the file path
dir_path = os.path.dirname(file_path)

# Specify the input file name
input_file_name = 'ifinfobase.csv'

# Specify the output file name
output_file_name = 'ifinfobase4.csv'

# Join the current directory with the input file name to get the full input file path
input_file_path = os.path.join(dir_path, input_file_name)

# Join the current directory with the output file name to get the full output file path
output_file_path = os.path.join(dir_path, output_file_name)

with open(input_file_path, "r") as file:
    for line in file:
        match = re.match(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) - (.*): %USER-(.*)-(.*): (.*): (.*)", line)
        if match:
            row = [match.group(1), match.group(2), match.group(3), match.group(4),match.group(5), match.group(6), match.group(7)]
            rows.append(row)

with open(output_file_path, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)
