import pandas as pd
from requests import get
from bs4 import BeautifulSoup
import os
import json
from os import path

def create_dictionary(url):
    
    # assign headers so that the codeup website allows us in
    headers = {'User-Agent': 'Codeup Bayes Data Science'}
    
    # creates response object
    response = get(url, headers=headers)
    
    # takes string of html and turns into a beautiful soup object so we can have access to the soup functions and properties
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # create a variable title. Can use select instead but it returns a list, so you need to step down into it with [0]
    title = soup.find('title').text
    
    # create a variable body.  Can also use select ie: soup.select('div.mk-single-content.clearfix')
    body = soup.find('div', class_='mk-single-content').get_text()
    
    # returning a dictionary literal
    return {'title': title, 'body': body}

#     Can also use this to return a dictionary.  Can loop through this, whereas you cant on the the one above
#     output = {}
#     output['title'] = title
#     output['body'] = body
#     return output


def get_blog_articles():
    
    articles = []
    
    urls = [
        "https://codeup.com/codeups-data-science-career-accelerator-is-here/",
        "https://codeup.com/data-science-myths/",
        "https://codeup.com/data-science-vs-data-analytics-whats-the-difference/",
        "https://codeup.com/10-tips-to-crush-it-at-the-sa-tech-job-fair/",
        "https://codeup.com/competitor-bootcamps-are-closing-is-the-model-in-danger/",
    ]
    
    for url in urls:
        # Can also use extend.  So if you bring in another list, it flattens it and adds it to the same list
        # whereas append throws the list inside of the original list
        articles.append(create_dictionary(url))
    
    df = pd.DataFrame(articles, columns=['title', 'body'])
    
    return articles, df