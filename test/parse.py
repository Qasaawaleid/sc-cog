import re
import csv
from operator import itemgetter


def parse_logs(file_name):
    data = []
    with open(file_name, "r") as f:
        log_block = f.read()
        log_lines = log_block.strip().split("\n")

        for line in log_lines:
            match = re.search(
                r"🖥️ Generated in (\d+) ms - Model: (.+?) - Width: (.+?) - Height: (.+?) - Steps: (.+?) - Outputs: (.+?) 🖥️", line)
            if match:
                data.append(
                    [
                        match.group(3),
                        match.group(4),
                        match.group(5),
                        match.group(6),
                        match.group(2),
                        match.group(1),
                    ]
                )
    return data


def write_to_csv(data):
    with open("test/logs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for index, row in enumerate(data):
            # Skip the primer row
            if index == 0:
                continue
            writer.writerow(row)


data = parse_logs("test/logs.txt")
data.sort(key=itemgetter(0, 2, 3))
write_to_csv(data)
