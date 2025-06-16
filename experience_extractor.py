import re

def extract_years_of_experience(text):
    text = text.lower()

    # List of patterns to search
    patterns = [
        r'(\d+)\s+(?:years|yrs)\s+(?:of\s+)?experience',
        r'experience of\s+(\d+)\s+(?:years|yrs)',
        r'(\d+)\s*\+?\s*(?:years|yrs)\s*(?:experience)?',
        r'(\d+)\s+(?:years|yrs)\s+(?:working|in)?'
    ]

    years = []

    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                years.append(int(match))
            except:
                pass

    if years:
        return max(years)
    else:
        return 0
