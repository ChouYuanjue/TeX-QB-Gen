import requests
from bs4 import BeautifulSoup
import re


def fetch_url(url, timeout=10):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def extract_math_questions_from_html(html):
    """A naive extractor that finds sections containing 'question', 'exercise', or tags with class names often used on Q&A sites.
    Returns a list of dicts: {title, body, answers:[...]}
    """
    soup = BeautifulSoup(html, 'lxml')
    results = []

    # Try common selectors for Q&A platforms
    # StackExchange uses div.question and div.answer
    questions = soup.select('div.question')
    if questions:
        for q in questions:
            title_tag = q.select_one('h1') or q.select_one('.question-hyperlink')
            title = title_tag.get_text(strip=True) if title_tag else ''
            body_tag = q.select_one('.post-text') or q
            body = body_tag.get_text('\n', strip=True)
            answers = []
            parent = q.find_parent()
            # find answers in the page
            for a in soup.select('div.answer'):
                a_body = a.select_one('.post-text')
                if a_body:
                    answers.append(a_body.get_text('\n', strip=True))
            results.append({'title': title, 'body': body, 'answers': answers})
        return results

    # Fallback: search for headings containing Exercise/Problem keywords
    pattern = re.compile(r'(Exercise|Problem|Example|练习|习题)', re.I)
    for h in soup.find_all(re.compile('^h[1-6]$')):
        if pattern.search(h.get_text()):
            # gather sibling nodes until next heading
            body_parts = []
            for sib in h.next_siblings:
                if sib.name and sib.name.startswith('h'):
                    break
                body_parts.append(getattr(sib, 'get_text', lambda sep='\n', strip=True: str(sib))("\n", True))
            results.append({'title': h.get_text(strip=True), 'body': '\n'.join(body_parts), 'answers': []})
    return results
