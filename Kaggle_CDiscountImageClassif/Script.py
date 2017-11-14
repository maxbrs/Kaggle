# coding: utf-8

import numpy as np
import pandas as pd
import io
from tqdm import tqdm
import bson
import matplotlib.pyplot as plt
from skimage.data import imread
import multiprocessing as mp



##########################
#                        #
#         KAGGLE         #
#  CDISCOUNT  CHALLENGE  #
#  IMAGE CLASSIFICATION  #
#                        #
##########################

# Cf. https://www.kaggle.com/inversion/processing-bson-files
# Cf. https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson



#----------
# Loading data :
#----------

train_len = 1768181
test_len = 1338268

data = bson.decode_file_iter(open('train.bson', 'rb'))


def extract(data):
    prod_to_category = dict()
    with tqdm(total=train_len) as pbar:
        for c, d in enumerate(data):
            #if c%1000 == 0:
            #    print('Progress : ' + str(round((c/train_len)*100, 2)) + '%')
            product_id = d['_id']
            category_id = d['category_id'] # This won't be in Test data
            prod_to_category[product_id] = category_id
            #for e, pic in enumerate(d['imgs']):
            #    picture = imread(io.BytesIO(pic['picture']))
                # do something with the picture, etc
            pbar.update()
    prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
    prod_to_category.index.name = '_id'
    prod_to_category.rename(columns={0: 'category_id'}, inplace=True)
    return prod_to_category

prod_to_category = extract(data)

prod_to_category.head()

#img = picture
#plt.imshow(img)
#plt.show()





##----------
## Multiprocessing :
##----------
#
#NCORE =  8
#
#prod_to_category = mp.Manager().dict() # note the difference
#
#def process(q, iolock):
#    while True:
#        d = q.get()
#        if d is None:
#            break
#        product_id = d['_id']
#        category_id = d['category_id']
#        prod_to_category[product_id] = category_id
#        for e, pic in enumerate(d['imgs']):
#            picture = imread(io.BytesIO(pic['picture']))
#            # do something with the picture, etc
#    
#q = mp.Queue(maxsize=NCORE)
#iolock = mp.Lock()
#pool = mp.Pool(NCORE, initializer=process, initargs=(q, iolock))
#
## process the file
#
#data = bson.decode_file_iter(open('../input/train_example.bson', 'rb'))
#for c, d in enumerate(data):
#    q.put(d)  # blocks until q below its max size
#
## tell workers we're done
#
#for _ in range(NCORE):  
#    q.put(None)
#pool.close()
#pool.join()
#
## convert back to normal dictionary
#prod_to_category = dict(prod_to_category)
#
#prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
#prod_to_category.index.name = '_id'
#prod_to_category.rename(columns={0: 'category_id'}, inplace=True)
#
#prod_to_category.head()








