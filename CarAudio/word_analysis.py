#!/usr/bin/env python
# -*- coding: utf-8 -*-


import jieba
import jieba.analyse
from io import open


class WordAnalysis(object):
    def __init__(self, filepath, filename):
        self.filepath = filepath
        self.filename = filename
    
    def readfile(self):
        with open(self.filepath + "/" + self.filename, "r", encoding='utf8') as f:
            content = f.readlines()
            return content
        
    def stopwordslist(self):
        stopwords = [line.strip() for line in self.readfile()]
        add_stopwords = ["我","帮","请"]
        stopwords.extend(add_stopwords)
        return stopwords
    
    def words_seg(self,sentence):
        seg_list = jieba.cut(sentence, cut_all=False, HMM=False)
        for word in seg_list:
            yield word
    
    def fetch_keywords_tfidf(self, sentence):
        "TF_IDF to fetch keywords"
        keywords = jieba.analyse.extract_tags(sentence, topK=5, withWeight=False, allowPOS=())
        return keywords

    def fetch_keywords(self, sentence):
        """
        use TextRank algorithm 
        """
        keywords = jieba.analyse.textrank(sentence, topK=5, withWeight=False, allowPOS=('ns','n','v'))
        return keywords

    def removestopwords(self, words):
        word_list = list()
        stopwords = self.stopwordslist()
        # words = self.words_seg(sentence)
        for word in words:
            if word not in stopwords:
                word_list.append(word)

        return word_list

