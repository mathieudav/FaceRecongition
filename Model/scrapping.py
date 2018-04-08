# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 13:24:13 2018

@author: demo
"""

import requests
from bs4 import BeautifulSoup
import urllib.request
import csv

with open('people.csv', newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    anaonymous = True
    for row in reader:
        if anaonymous:
            anaonymous = False
        else:
            names = row[1].split( )
            link = "https://www.google.fr/search?tbm=isch&q="
            for name in names:
                link += name + "+"
            link += "movie"
    
            page = requests.get(link)
            soup = BeautifulSoup(page.content)
            i = 0
            for p in soup.find_all('img'):
                dest_file = "Data/"+row[0]+"/"
                name_file = dest_file+"scraped"+str(i)+".jpg"
                urllib.request.urlretrieve(p.attrs['src'], name_file)
                i += 1