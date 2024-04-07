import scrapy
from ..items import ScrapItem
import json


class Scrap_ProdSpider(scrapy.Spider):
    name = 'prod'

    start_urls = [
        'https://www.woolworths.com.au/shop/browse/drinks/cordials-juices-iced-teas/iced-teas'
    ]

    def parse(self, response):
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:96.0) Gecko/20100101 Firefox/96.0",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.5",
            "Content-Type": "application/json"

        }
        url = "https://www.woolworths.com.au/apis/ui/browse/category"
        body = "{\"categoryId\":\"1_9573995\",\"pageNumber\":1,\"pageSize\":36,\"sortType\":\"TraderRelevance\",\"url\":\"/shop/browse/drinks/cordials-juices-iced-teas/iced-teas\",\"location\":\"/shop/browse/drinks/cordials-juices-iced-teas/iced-teas\",\"formatObject\":\"{\\\"name\\\":\\\"Iced Teas\\\"}\",\"isSpecial\":false,\"isBundle\":false,\"isMobile\":false,\"filters\":[],\"token\":\"\"}"
        yield scrapy.Request(url, callback=self.result, method="POST", body=body, headers=headers)

    def result(self, response):
        item = ScrapItem()
        data = json.loads(response.body)
        products = data['Bundles']
        for product in products:
            fields = product['Products']
            for i in fields:
                department_names = i['AdditionalAttributes']['piesdepartmentnamesjson'].replace('\"', '').replace('[','').replace(']', '')
                category_names = i['AdditionalAttributes']['piescategorynamesjson'].replace('\"', '').replace('\"','').replace('[', '').replace(']', '')
                subcategory_names = i['AdditionalAttributes']['piessubcategorynamesjson'].replace('\"', '').replace('\"', '').replace('[', '').replace(']', '')
                item['Products'] = product['Name']
                item['Breadcrumb'] = ['Home', department_names, category_names, subcategory_names]
                yield item

