import requests
import gzip
import json
import logging
import csv
import pandas as pd


class product_information(object):
    def __init__(self):
        self.important = ['Price', 'IsInStock', 'Name', 'Barcode']
        self.product_with_any_information = None
        self.dataframe = None
        self.load()

    def load(self):
        with gzip.open('./data/data.json.gz', "r") as f:
            j = json.loads(f.read().decode('utf-8'))
            # list comprehension with dictionary {"products"  : [informations...]} // replacement keys "Product" by "Products" for better analysis of our data
            js_list = [{k: v for k, v in j["Bundles"][i].items()} for i in range(len(j["Bundles"])) if "Product" not in j["Bundles"][i].keys()]
            js_list.insert(0, {"Products": j["Bundles"][0]["Product"]})
            # read data in a dataframe and drop the col 'Type'
            data = pd.read_json(json.dumps(js_list))
            data = data.drop(['Type'], axis=1)
            # dataframe of product without any information // name given but any information
            self.product_with_any_iformation = data[data["Products"].isnull()][data[data["Products"].isnull()]["Name"].notna()]["Name"]
            self.product_with_any_iformation = self.product_with_any_iformation.reset_index(drop=True)
            # print(product_with_any_iformation)
            # remove all line with NaN
            data = data.dropna(axis=0)
            data = data.reset_index(drop=True)
            # col that we need in our analysis
            important = self.important
            # list comprehension for our final dataframe // with just importants col
            list_dict = [{important[i]: data['Products'][j][0][important[i]] for i in range(len(important))} if len(data['Products'][j]) == 1 else {important[i]: data['Products'][j][1][important[i]] for i in range(len(important))} for j in range(len(data['Products']))]
            self.dataframe = pd.DataFrame(list_dict)
            # print(len(self.dataframe))

    def print_write_result(self):
        csv_file = open('data/product.csv', 'w')
        writer = csv.writer(csv_file)
        writer.writerow(['id', 'name', 'price'])
        for i in range(len(self.dataframe)):
            name = self.dataframe['Name'][i][:30]
            price = self.dataframe["Price"][i]
            id = self.dataframe['Barcode'][i]
            if price == None:
                price = 0
            price = '%.1f' % (price)
            if not self.dataframe["IsInStock"][i]:
                logging.warning("  id :{id}, name:{name} is unavailable".format(id=id, name=name))
            else:
                writer.writerow([id, name, price])
                print("You can buy " + str(name) + " at our store at " + str(price))
        for j in range(len(self.product_with_any_iformation)):
            logging.error("ERROR: {} availability can’t be found ".format(self.product_with_any_iformation[0]))


p = product_information()
print(p.print_write_result())