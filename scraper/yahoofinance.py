import requests
from bs4 import BeautifulSoup
from typing import List, Tuple
import csv

class Scraper:
    def __init__(self, url: str, symbol: str)-> None:
        self.url: str = url
        self.symbol: str = symbol
        self.stock_data:List[List[float]] = []

    def request(self) -> requests:
        self.headers: dict = {
            "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36",
            "From": "piya.prayag@yahoo.com"
        }
        req = requests.get(self.url, headers=self.headers)
        if req.status_code == 200:
            print("Website data stored succesfully")
        else:
            print("Make sure url is okay")
        return req
    
    def load_file(self) -> None:
        html_document: requests = self.request()
        with open(self.symbol+".html", "w", encoding="utf-8") as html_doc:
            html_doc.write(html_document.text)

    def parser(self, file_path: str) -> None:
        with open(file_path, 'r', encoding="utf-8") as html_doc:
            doc_path = BeautifulSoup(html_doc.read())
            tr_tag = doc_path.find_all("tr")
            for tags in tr_tag:
                data = []
                td_tags = tags.find_all("td")
                for text in td_tags:
                    data.append(text.text.strip())
                self.stock_data.append(data)
    
    def csvwriter(self):
        with open(self.symbol+".csv", "w", newline='') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            wr.writerows(self.stock_data)

# scraper = Scraper("https://finance.yahoo.com/quote/NVDA/history/?period1=917015400&period2=1728398759", "NVDA")
# scraper.request()
# scraper.load_file()
# scraper.parser("NVDA.html")
# scraper.csvwriter()