import requests
from typing import Dict, List

STACK_API = 'https://api.stackexchange.com/2.3/search/advanced'
QUESTION_API = 'https://api.stackexchange.com/2.3/questions/{ids}'
ANSWER_API = 'https://api.stackexchange.com/2.3/questions/{ids}/answers'

def fetch_questions_by_keyword(keyword: str, site: str = 'math', pagesize: int = 10, key: str = None) -> List[Dict]:
    """Search StackExchange sites by keyword and return question IDs & titles.

    site: 'math' for math.stackexchange.com, 'mathoverflow' for mathoverflow.net
    """
    params = {
        'order': 'desc',
        'sort': 'relevance',
        'q': keyword,
        'site': site,
        'pagesize': pagesize,
    }
    if key:
        params['key'] = key
    r = requests.get(STACK_API, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    items = data.get('items', [])
    # return minimal info
    return [{'question_id': it['question_id'], 'title': it.get('title','')} for it in items]


def fetch_full_questions(ids: List[int], site: str = 'math', key: str = None) -> List[Dict]:
    """Fetch full question bodies and answers given a list of ids."""
    if not ids:
        return []
    ids_str = ';'.join(str(i) for i in ids)
    params = {
        'order': 'desc',
        'sort': 'activity',
        'site': site,
        'filter': 'withbody'
    }
    if key:
        params['key'] = key
    url = QUESTION_API.format(ids=ids_str)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    out = []
    for it in data.get('items', []):
        q = {
            'id': it.get('question_id'),
            'title': it.get('title'),
            'body': it.get('body', ''),
            'answers': []
        }
        for a in it.get('answers', []) if it.get('answers') else []:
            q['answers'].append(a.get('body', ''))
        out.append(q)
    missing_answers = [q['id'] for q in out if not q['answers']]
    if missing_answers:
        params = {
            'order': 'desc',
            'sort': 'votes',
            'site': site,
            'filter': 'withbody'
        }
        if key:
            params['key'] = key
        url = ANSWER_API.format(ids=';'.join(str(i) for i in missing_answers))
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        answer_data = resp.json().get('items', [])
        answer_map: Dict[int, List[str]] = {}
        for ans in answer_data:
            qid = ans.get('question_id')
            answer_map.setdefault(qid, []).append(ans.get('body', ''))
        for q in out:
            if q['id'] in answer_map:
                q['answers'].extend(answer_map[q['id']])
    return out
