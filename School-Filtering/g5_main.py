# -*- coding: utf-8 -*-
"""
============================================================================
Created on Tue Jan  10 15:45:27 2018
@group: Groupe 5 - Filtrage
Main Program
============================================================================
"""
import pickle
import json
from tqdm import tqdm


# Server
from functions.g5_import_json import import_daily_jsons
from functions.g5_integration import tag_text
from functions.g5_tfidf import get_tf_idf
"""
============================================================================
    links
============================================================================
"""

# LINK ON THE SERVER
#path_source = '/var/www/html/projet2018/data/clean/robot/data/clean/filtering'
#path_target_save = '/var/www/html/projet2018/data/clean/filtering_save'
#path_post_filt_target = '/var/www/html/projet2018/data/clean/temporary_filtering/post_filtering'
#path_post_tf_target = '/var/www/html/projet2018/data/clean/temporary_filtering/post_tfidf'
#stop_words = pickle.load(open('/var/www/html/projet2018/code/filtering/functions/stopwords.p', 'rb'))

# LINK ON GITHUB
path_source = './data/source_press_article'
path_target = './data/target_press_article'
path_target_save = './data/target_press_article_save'
path_post_filt_target = './data/post/filtering'
path_post_tf_target = './data/post/tfidf'
stop_words = pickle.load(open('./functions/stopwords.p', 'rb'))


articles = import_daily_jsons(path_source)

#articles = {key: articles[key] for key in list(articles)[0:5]}

with tqdm(desc='Filtering', total=len(articles)) as pbar:
    for item in articles:
        art = articles[item]
        data_post_content, filtered = tag_text(art, is_title=False)
        data_post_title = tag_text(art, is_title=True)
        data_post_title = list(data_post_title)
        for dic in range(len(data_post_title)):
            data_post_content["position_word"].append(data_post_title[dic])
        data_post_content["id_art"] = art["id_art"]
        data_post = []
        data_post.append(data_post_content)
#        data_post = json.dumps(data_post, ensure_ascii='False')
#        log_post_filt = post_filtering(data_post)
#        id_article = log_post_filt.json()[0][0]["message"]["id_article"]
        ifile = path_post_filt_target + '/' + item + '_post_filtered.json'
        with open(ifile, 'w', encoding='utf-8') as outfile:
            json.dump(data_post, outfile, ensure_ascii=False)
        tfidf = get_tf_idf(filtered['list_lemma'], art["id_art"])
#        tfidf = json.dumps(tfidf, ensure_ascii='False')
#        log_post_tf = post_tfidf(tfidf)
#        print('log_post_tf = '+str(log_post_tf))
        ifile = path_post_tf_target + '/' + item + '_post_tf.json'
        with open(ifile, 'w', encoding='utf-8') as outfile:
            json.dump(tfidf, outfile, ensure_ascii=False)
        art["content"] = filtered
        #art["id_article"] = id_article
        ifile = path_target + '/' + item + '_filtering.json'
        with open(ifile, 'w', encoding='utf-8') as outfile:
            json.dump(art, outfile, ensure_ascii=False)
        ifile = path_target_save + '/' + item + '_filtering.json'
        with open(ifile, 'w', encoding='utf-8') as outfile:
            json.dump(art, outfile, ensure_ascii=False)
        pbar.update()

#import insert_json_filtering
#import insert_json_tfidf

print('Opération journalière de filtrage et d\'intégration en Base de Données OK !')
