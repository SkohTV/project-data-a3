#!/usr/bin/env python3

import argparse
import requests as r
from bs4 import BeautifulSoup



'''
It took a while to find a website that didn't forbid scrapping in their robots.txt
Since we are good people, we don't scrap when forbidden :)
'''

parser = argparse.ArgumentParser(
  prog='Image retriever',
  description='Takes a vessel MMSI and find it\' picture',
)
parser.add_argument('mmsi')
args = parser.parse_args()

mmsi = args.mmsi
req = r.get(f'https://www.myshiptracking.com/vessels/mmsi-{mmsi}')
soup = BeautifulSoup(req.content, 'html.parser')

# The *magic* scrapper
try:
  image_link = [i.strip() for i in soup.find(id='asset-main-image').find('source')['srcset'].split(',')][-1].split(' ')[0]# type: ignore
except AttributeError:
    pass
else:
  print(image_link)
