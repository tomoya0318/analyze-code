import re


def extract_data(text):
    pattern = re.compile(r"^(.*?):\s+\(([^)]*)\)$")
    match = pattern.match(text)

    if match:
        key = match.group(1).strip()
        value = match.group(2).strip()
        return key, value
    else:
        return None, None
