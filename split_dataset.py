import csv
import re
from datetime import datetime
from urllib.parse import urlparse


def extract_urls(text):
    """Extract all URLs from text"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)


def parse_email_content(email_content):
    """Parse email content to extract sender, date, body, etc."""
    lines = email_content.strip().split('\n')

    sender = ""
    date_str = ""
    body_lines = []

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        line = line.lower()

        if line.startswith("from:"):
            sender = line.replace("from:", "").strip()
        elif line.startswith("date:"):
            date_str = line.replace("date:", "").strip()
        elif line.startswith("to:") or line.startswith("attachment:"):
            pass
        else:
            if line and not line.startswith("from:") and not line.startswith("to:") and not line.startswith("date:"):
                body_lines.extend(lines[i:])
                break
        i += 1

    body = '\n'.join(body_lines).strip()

    return sender, date_str, body


def parse_date(date_str):
    """Parse date string into day, month, year (no time in the example)"""
    try:
        dt = datetime.strptime(date_str.strip(), "%m/%d/%Y")
        return dt.day, dt.month, dt.year
    except:
        try:
            dt = datetime.strptime(date_str.strip(), "%d/%m/%Y")
            return dt.day, dt.month, dt.year
        except:
            return None, None, None


def extract_sender_domain(sender):
    """Extract domain from sender email"""
    match = re.search(r'@([a-zA-Z0-9.-]+)', sender)
    if match:
        return match.group(1)
    return ""


def process_csv(input_file, output_file, empty_rows_file):
    """Process the email CSV and create expanded output"""

    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8', newline='') as outfile, \
            open(empty_rows_file, 'w', encoding='utf-8', newline='') as emptyfile:

        reader = csv.DictReader(infile)

        fieldnames = ['day', 'month', 'year', 'urls', 'url_count',
                      'sender_domain', 'sender', 'title', 'body', 'label']

        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        empty_writer = csv.DictWriter(emptyfile, fieldnames=fieldnames)
        empty_writer.writeheader()

        row_number = 0
        rows_with_empty_cells = 0

        for row in reader:
            row_number += 1
            email_subject = row['Email_Subject']
            email_content = row['Email_Content']
            label = row['Label']

            sender, date_str, body = parse_email_content(email_content)
            day, month, year = parse_date(date_str)

            urls = extract_urls(email_content)
            url_list = '|'.join(urls) if urls else ""
            url_count = len(urls)

            sender_domain = extract_sender_domain(sender)

            title = email_subject

            output_row = {
                'day': day if day else '',
                'month': month if month else '',
                'year': year if year else '',
                'urls': url_list,
                'url_count': url_count,
                'sender_domain': sender_domain,
                'sender': sender,
                'title': title,
                'body': body,
                'label': label
            }

            has_empty_cell = any(str(value).strip() == '' for value in output_row.values())

            if has_empty_cell:
                rows_with_empty_cells += 1
                empty_writer.writerow(output_row)

            writer.writerow(output_row)

        print(f"Processing complete!")
        print(f"Main output saved to: {output_file}")
        print(f"Empty rows saved to: {empty_rows_file}")
        print(f"Total rows processed: {row_number}")
        print(f"Rows with empty cells: {rows_with_empty_cells}")


input_csv = "./datasets/dataset.csv"
output_csv = "./datasets/extracted_dataset.csv"
empty_rows_csv = "./datasets/rows_with_empty_cells.csv"

process_csv(input_csv, output_csv, empty_rows_csv)