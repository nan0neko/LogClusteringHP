import csv
import re

header = ["Date", "Time","Software/Module", "User","Type_Log","Interface_Description", "Data"]

rows = []

with open(r"C:\Users\toutu\Downloads\ifinfobase.csv", "r") as file:

    for line in file:

        match = re.match(r"(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}) - (.*): %USER-(.*)-(.*): (.*): (.*)", line)

        if match:
            row = [match.group(1), match.group(2), match.group(3), match.group(4),match.group(5), match.group(6), match.group(7)]
            rows.append(row)

with open(r"C:\Users\toutu\Downloads\ifinfobase4.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)
