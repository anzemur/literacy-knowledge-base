# Scrapper for Project Gutenberg website: https://www.gutenberg.org/

from bs4 import BeautifulSoup as bs
import requests
import json
from tqdm import tqdm

SUBJECT = '94'  # Short stories.
LIMIT = 1000
PAGINATION = 25

BASE_URL = f'https://www.gutenberg.org/ebooks/subject/{SUBJECT}?start_index='
BASE_META_URL = 'https://www.gutenberg.org/ebooks/'
BASE_FILE_URL = 'https://www.gutenberg.org/files/'


LICENSE = 'Public domain in the USA.'
LANGUAGE = 'English'


def get_stories_list():
    stories = []

    for skip in tqdm(range(1, LIMIT + PAGINATION, PAGINATION)):
        try:
            res = requests.get(BASE_URL + str(skip))

            if (res.status_code == 200):
                soup = bs(res.text, features='html.parser')

                for book_li in soup.find_all('li', class_='booklink'):
                    book_id = str(book_li.find('a').get('href')).split('/')[2]
                    book_title = str(book_li.find('span', class_='title').contents[0])

                    # Don't scrape stories that have multiple stories.
                    if 'stories' not in book_title.lower():
                        res_meta = requests.get(BASE_META_URL + book_id)
                        if (res_meta.status_code == 200):
                            soup = bs(res_meta.text, features='html.parser')

                            lang = soup.find('tr', {'property': 'dcterms:language'}).find('td').contents[0]
                            license = soup.find('td', {'property': 'dcterms:rights'}).contents[0]

                            # Check for english book and public license.
                            if (lang == LANGUAGE and license == LICENSE):
                                res_file = requests.get(BASE_FILE_URL + book_id)
                                if (res_file.status_code == 200):
                                    soup = bs(res_file.text, features='html.parser')

                                    book_txt = None
                                    for file_link in soup.find_all("a"):
                                        link = file_link.get("href")

                                        if link.endswith(".txt"):
                                            book_txt = link

                                    # Check for the version in `txt` format.
                                    if book_txt != None:
                                        stories.append({
                                            'id': book_id,
                                            'title': book_title,
                                            'file_url': f'{BASE_FILE_URL}{book_id}/{book_txt}'
                                        })
        except Exception as e:
            print(e)

    print(f'Scrapped {len(stories)} short stories.')
    out_file = open('scrapped_books.json', 'w')
    json.dump(stories, out_file, indent=2)
    out_file.close()


def get_stories_files():
    stories = json.load(open('scrapped_books.json', 'r'))
    successfull = len(stories)
    out_dir = '../../data/gutenberg/'

    for story in tqdm(stories):
        try:
            res = requests.get(story['file_url'])

            if (res.status_code == 200):
                with open(f'{out_dir}/{story["title"]}.txt', 'w') as f:
                    f.writelines(res.text)
            else:
                successfull -= 1

        except Exception as e:
            successfull -= 1
            print(e)

    print(f'Successfully downloaded {successfull}/{len(stories)} short stories.')


if __name__ == '__main__':
    get_stories_files()
