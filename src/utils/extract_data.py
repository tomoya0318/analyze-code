import re


def extract_data(text):
    """正規表現にマッチするかの判定

    Args:
        text (str): マッチさせる文

    Returns:
        tuple: マッチしたキーと値のペア、または None, None
    """
    pattern = re.compile(r"^(.*?):\s+\(([^)]*)\)$")
    match = pattern.match(text)

    if match:
        key = match.group(1).strip()
        value = match.group(2).strip()
        return key, value
    else:
        return None, None
