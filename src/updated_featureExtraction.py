"""
	Construct account, fuct and review features given a review dataset.
"""
import nltk
import nltk.tokenize.punkt
from nltk.tokenize import word_tokenize

import bisect
import math

import numpy as np

import codecs
import re
import string
from datetime import datetime

import json
import pickle
import gzip
import sys
import copy
sys.path.insert(0, '../../Utils')
from iohelper import *


date_time_format_str = '%Y-%m-%d'

def MNR(data):
	"""
		Normalized maximum number of reviews in a day for a user/product
		Args:
			data is a dictionary with key=u_id or p_id and value = tuples of (neighbor id, rating, label, posting time)
		Return:
			dictionary with key = u_id or p_id and value = MNR
	"""
	# maximum number of reviews written in a day for user / product
	feature = {}
	for i, d in data.items():
		# key = posting date; value = number of reviews
		frequency = {}
		for t in d:
			if t[3] not in frequency:
				frequency[t[3]] = 1
			else:
				frequency[t[3]] += 1
		feature[i] = max(frequency.values())
	# normalize it
	normalizer = max(feature.values())

	for k in feature.keys():
		feature[k] /= normalizer
	return feature

def iMNR(data, new_data):
	""" Incremental version of MNR
	"""
	feature = {}
	for i, d in new_data.items():
		all_d = copy.deepcopy(d)
		if i in data:
			all_d += data[i]
		frequency = {}
		for t in all_d:
			if t[3] not in frequency:
				frequency[t[3]] = 1
			else:
				frequency[t[3]] += 1
		feature[i] = max(frequency.values())
	
	# normalize it
	normalizer = max(feature.values())

	for k in feature.keys():
		feature[k] /= normalizer
	return feature

def PR_NR(data):
	"""
		Ratio of positive and negative reviews of a user or product
		Args:
			data is a dictionary with key=u_id or p_id and value = tuples of (neighbor id, rating, label, posting time)
		Return:
			dictionary with key = u_id or p_id and value = (PR, NR)
	"""
	feature = {}

	for i, d in data.items():
		positives = [1 for t in d if t[1] > 3]
		negatives = [1 for t in d if t[1] < 3]
		feature[i] = (float(len(positives)) / len(d), float(len(negatives)) / len(d))
	return feature

def iPR_NR(data, new_data):
	feature = {}
	for i, d in new_data.items():
		all_d = copy.deepcopy(d)
		if i in data:
			all_d = all_d + data[i]
		positives = [1 for t in all_d if t[1] > 3]
		negatives = [1 for t in all_d if t[1] < 3]
		feature[i] = (float(len(positives)) / len(all_d), float(len(negatives)) / len(all_d))

	return feature

def avgRD_user(user_data, product_data):
    """
        Average rating deviation of each user / product.
        For a user i, avgRD(i) = average(r_ij - avg_j | for all r_ij of the user i)
        For a product j, avgRD(j) = average(r_ij - avg_j | for all r_ij of the user i) = 0!?
        Return:
            average rating deviation on users, as defined in the paper
            Detecting product review spammers using rating behaviors, CIKM, 2010
    """
    # find the average rating of each product
    p_avg = {}
    for i, d in product_data.items():
        p_avg[i] = np.mean(np.array([t[1] for t in d]))

    # find average rating deviation of each user
    u_avgRD = {}
    for i, d in user_data.items():
        u_avgRD[i] = np.mean(np.array([abs(t[1] - p_avg[t[0]]) for t in d]))

    return u_avgRD

def iavgRD_user(user_data, new_user_data, product_data, new_product_data):
    """
        Need to ensure those target products in new_user_data are also in new_product_data
    """
    # find averaged ratings of the newly added products
    p_avg = {}
    for i, d in new_product_data.items():
        all_d = copy.deepcopy(d)
        if i in product_data:
            all_d += product_data[i]

        p_avg[i] = np.mean(np.array([t[1] for t in all_d]))

    # find average rating deviation of each user
    u_avgRD = {}
    for i, d in new_user_data.items():
      all_d = copy.deepcopy(d)
# remember to include this user's previous reviews
      if i in user_data:
        all_d = all_d + user_data[i]
        # go thru all targets (including new and existing ones) of user i
        for r in all_d:
            product_id = r[0]
            if product_id not in p_avg:
                product_reviews = product_data[product_id]
                p_avg[product_id] = np.mean(np.array([t[1] for t in product_reviews]))

      u_avgRD[i] = np.mean(np.array([abs(t[1] - p_avg[t[0]]) for t in all_d]))

    return u_avgRD

def avgRD_prod(product_data):
    """
        Average rating deviation of each user / product.
        For a user i, avgRD(i) = average(r_ij - avg_j | for all r_ij of the product i)
        For a product j, avgRD(j) = average(r_ij - avg_j | for all r_ij of the product i) = 0!?
        Return:
            average rating deviation on products, as defined in the paper
            collective opinion spam detection: bridging review networks and metadata, KDD, 2015
    """
    
    # find the average rating of each product
    p_avg = {}
    for i, d in product_data.items():
        p_avg[i] = np.mean(np.array([t[1] for t in d]))
        
    # find average rating deviation of each product
    p_avgRD = {}
    for i, d in product_data.items():
        p_avgRD[i] = np.mean(np.array([abs(t[1] - p_avg[i]) for t in d]))
        
    return p_avgRD

def iavgRD_prod(product_data, new_product_data):
    
    # find the average rating of each product
    p_avg = {}
    p_avgRD = {}
    for i, d in new_product_data.items():
        all_d = copy.deepcopy(d)
        if i == '1':
            print (len(all_d))
        if i in product_data:
			# this will modify the contents of all_d and also the new_product_data!
            all_d += product_data[i]
        p_avg[i] = np.mean(np.array([t[1] for t in all_d]))
        p_avgRD[i] = np.mean(np.array([abs(t[1] - p_avg[i]) for t in all_d]))
	#    if i == '1':
    #        print (len(all_d))
    #        print (p_avg[i])
        
    # find average rating deviation of each product
	#p_avgRD = {}
    #for i, d in new_product_data.items():
    #    all_d = d
    #    if i == '1':
    #        print (len(all_d))
    #    if i in product_data:
    #        all_d += product_data[i]
    #    if i == '1':
    #        print (len(all_d))
    #    p_avgRD[i] = np.mean(np.array([abs(t[1] - p_avg[i]) for t in all_d]))
        
    return p_avgRD

def BST(user_data):
	""" Burstiness of reviews by users. Spammers are often short term
		members of the site: so BST(i) = 0, if L(i) - F(i) > tau else BST(i) = 1 -
		(L(i) - F(i))/tau, where, L(i) - F(i) are number of days between first and
		last review of i, tau = 28 days
		
		Args:
			user_data is a dictionary with key=u_id value = tuples of (prod_id, rating, label, posting time)
		Return:
			dictionary with key = u_id and value = BST
	"""
	bst = {}
	tau = 28.0	# 28 days
	for i, d in user_data.items():
		post_dates = sorted([datetime.strptime(t[3], date_time_format_str) for t in d])
		delta_days = (post_dates[-1] - post_dates[0]).days
		if delta_days > tau:
			bst[i] = 0.0
		else:
			bst[i] = 1.0 - (delta_days / tau)
	return bst

def iBST(user_data, new_user_data):
	bst = {}
	tau = 28.0	# 28 days
	for i, d in new_user_data.items():
		all_d = copy.deepcopy(d)
		if i in user_data:
			all_d += user_data[i]
		post_dates = sorted([datetime.strptime(t[3], date_time_format_str) for t in all_d])
		delta_days = (post_dates[-1] - post_dates[0]).days
		if delta_days > tau:
			bst[i] = 0.0
		else:
			bst[i] = 1.0 - (delta_days / tau)
	return bst

def ERD(data):
	"""
		Entropy of the rating distribution of each user (product)
	"""
	erd = {}
	for i, d in data.items():
		ratings = [t[1] for t in d]
		h, _ = np.histogram(ratings, bins = np.arange(1,7))
		h = h / h.sum()
		h = h[np.nonzero(h)]
		erd[i] = (- h * np.log2(h)).sum()
	return erd

def iERD(data, new_data):
	erd = {}
	for i, d in new_data.items():
		all_d = copy.deepcopy(d)
		if i in data:
			all_d += data[i]
		ratings = [t[1] for t in all_d]
		h, _ = np.histogram(ratings, bins = np.arange(1,7))
		h = h / h.sum()
		h = h[np.nonzero(h)]
		erd[i] = (- h * np.log2(h)).sum()
	return erd

def ETG(data):
	"""
		Entropy of the gaps between any two consecutive ratings.
	"""
	etg = {}
# [0,1) -> 1
# [1,3) -> 2
# ...
# [17, 33) -> 6
# anything larger than 33 will be discarded.

	edges = [0, 0.5, 1, 4, 7, 13, 33]
	for i, d in data.items():

		# if there is only one posting time, then entropy = 0
		if len(d) <= 1:
			etg[i] = 0
			continue
		# sort posting dates from the past to the future
		posting_dates = sorted([datetime.strptime(t[3], date_time_format_str) for t in d])
		
		# find the difference in days between two consecutive dates
		delta_days = [(posting_dates[i+1] - posting_dates[i]).days for i in range(len(posting_dates) - 1)]
		delta_days = [d for d in delta_days if d < 33]
		
		# bin to the 6 bins, discarding any differences that are greater than 33 days
		h = []
		for delta in delta_days:
			j = 0
			while j < len(edges) and delta > edges[j]:
				j += 1
			h.append(j)
		_, h = np.unique(h, return_counts=True)
		if h.sum() == 0:
			etg[i] = 0
			continue
		h = h / h.sum()
		h = h[np.nonzero(h)]
		etg[i] = np.sum(- h * np.log2(h))
	return etg

def iETG(data, new_data):
	etg = {}
	edges = [0, 0.5, 1, 4, 7, 13, 33]
	for i, d in new_data.items():
		all_d = copy.deepcopy(d)
		if i in data:
			all_d += data[i]
		# if there is only one posting time, then entropy = 0
		if len(all_d) <= 1:
			etg[i] = 0
			continue
		# sort posting dates from the past to the future
		posting_dates = sorted([datetime.strptime(t[3], date_time_format_str) for t in all_d])
		
		# find the difference in days between two consecutive dates
		delta_days = [(posting_dates[i+1] - posting_dates[i]).days for i in range(len(posting_dates) - 1)]
		delta_days = [d for d in delta_days if d < 33]
		
		# bin to the 6 bins, discarding any differences that are greater than 33 days
		h = []
		for delta in delta_days:
			j = 0
			while j < len(edges) and delta > edges[j]:
				j += 1
			h.append(j)
		_, h = np.unique(h, return_counts=True)
		if h.sum() == 0:
			etg[i] = 0
			continue
		h = h / h.sum()
		h = h[np.nonzero(h)]
		etg[i] = np.sum(- h * np.log2(h))
	return etg

def RL(data, text_features, isUser = True):
	"""
		Average review length of each user / product
		Args:
			data: a dictionary with key=u_id or p_id and value = tuples of (neighbor id, rating, label, posting time)
			text_features: dictionary with key=(u_id, p_id) and value = dict of feature_name:feature_value
			isUser: is dealing with user data or product data
		Return:
			dictionary with key = u_id or p_id and value = (RL)
	"""
	rl = {}
	for i, d in data.items():
		if isUser:
			# d[0] is product id
			lengths = [text_features[(i, t[0])]['L'] for t in d]
		else:
			lengths = [text_features[(t[0], i)]['L'] for t in d]
		rl[i] = np.mean(lengths)
	return rl

def RD(product_data):
	"""Calculate the deviation of the review ratings to the product average.
		
		Args:
			prod_data:
		Return:
			a dictionary with key = (u_id, p_id), value = deviation of the rating of this review to the average rating of the target product
	"""
	rd = {}
	for i, d in product_data.items():
		avg = np.mean(np.array([t[1] for t in d]))
		for t in d:
			rd[(t[0], i)] = abs(t[1] - avg)
	return rd

def iRD(product_data, new_product_data):
	rd = {}
	for i, d in new_product_data.items():
		all_d = copy.deepcopy(d)
		if i in product_data:
			all_d = d + product_data[i]
		avg = np.mean(np.array([t[1] for t in all_d]))
		for t in all_d:
			rd[(t[0], i)] = abs(t[1] - avg)
	return rd

def EXT(product_data):
	"""
		Whether a rating is extreme or not
		Args:
			product_data is a dictionary with key=p_id and value = tuples of (u_id, rating, label, posting time)
		Return:
			a dictionary with key = (u_id, p_id) and value = 0 (not extreme) / 1 (extreme)
	"""
	ext = {}
	for i, d in product_data.items():
		for t in d:
			if int(t[1]) == 5 or int(t[1]) == 1:
				ext[(t[0], i)] = 1
			else:
				ext[(t[0], i)] = 0
	return ext

def iEXT(product_data, new_product_data):
	ext = {}
	for i, d in new_product_data.items():
		all_d = copy.deepcopy(d)
		if i in product_data:
			all_d = d + product_data[i]
		for t in all_d:
			if int(t[1]) == 5 or int(t[1]) == 1:
				ext[(t[0], i)] = 1
			else:
				ext[(t[0], i)] = 0
	return ext

def DEV(product_data):
	"""
		Deviation of each rating from the average rating of the target product.
		Need to use "recursive minimal entropy partitioning" to find beta_1
		Args:
			product_data is a dictionary with key=p_id and value = tuples of (neighbor id, rating, label, posting time)
		Return:
			a dictionary with key = (u_id, p_id) and value = (RD_ij, RD_ij / 4 > 0.63 ? 1: 0)
			where RD_ij = |r_ij - average rating of product j|
	"""
	beta_1 = 0.63
	dev = {}
# i is a product id
	for i, d in product_data.items():
		# find the average rating of each product
		p_avg_rating = np.mean(np.array([t[1] for t in d]))
		for t in d:
			u_id = t[0]	# user id
			if (abs(p_avg_rating - t[1]) / 4.0 > 0.63):
				dev[(u_id, i)] = 1	# absolute difference between current rating and product average rating
			else:
				dev[(u_id, i)] = 0	# absolute difference between current rating and product average rating
	return dev

def iDEV(product_data, new_product_data):
	beta_1 = 0.63
	dev = {}
# i is a product id
	for i, d in new_product_data.items():
		all_d = copy.deepcopy(d)
		if i in product_data:
			all_d = d + product_data[i]
		# find the average rating of each product
		p_avg_rating = np.mean(np.array([t[1] for t in all_d]))
		for t in all_d:
			u_id = t[0]	# user id
			if (abs(p_avg_rating - t[1]) / 4.0 > 0.63):
				dev[(u_id, i)] = 1	# absolute difference between current rating and product average rating
			else:
				dev[(u_id, i)] = 0	# absolute difference between current rating and product average rating
	return dev

def ETF(product_data):
	"""
		Binary feature: 0 if ETF_prime <= beta_3, 1 otherwise.
		Need to use "recursive minimal entropy partitioning" to find beta_1
		ETF_prime = 1 - (date of last review of user i on product p from the date of the first review of the product / 7 months)
	"""

	beta_3 = 0.69

# for each product j, find the time of the earliest review F(j)
	first_time_product = {}
	for i,d in product_data.items():
		for t in d:
			if i not in first_time_product:
				first_time_product[i] = datetime.strptime(t[3], date_time_format_str)
			elif datetime.strptime(t[3], date_time_format_str) < first_time_product[i]:
				first_time_product[i] = datetime.strptime(t[3], date_time_format_str)

	etf = {}	# key = (u_id, p_id), value = maximum difference between reviews (u_id, p_id) and first review of the product
	for i, d in product_data.items():
		for t in d:
			td = datetime.strptime(t[3], date_time_format_str) - first_time_product[i]
			if (t[0], i) not in etf:
				etf[(t[0], i)] = td
# find the largest td for the review
			elif td > etf[(t[0], i)]:
				etf[(t[0], i)] = td
	
	for k, v in etf.items():
		if v.days > 7*30:
			etf[k] = 0
		elif 1.0 - v.days / (7 * 30) > beta_3:
			etf[k] = 1
		else:
			etf[k] = 0
	return etf

def iETF(product_data, new_product_data):
	beta_3 = 0.69
# for each product j, find the time of the earliest review F(j)
	first_time_product = {}
	for i,d in new_product_data.items():
		all_d = copy.deepcopy(d)
		if i in product_data:
			all_d = d + product_data[i]
		for t in all_d:
			if i not in first_time_product:
				first_time_product[i] = datetime.strptime(t[3], date_time_format_str)
			elif datetime.strptime(t[3], date_time_format_str) < first_time_product[i]:
				first_time_product[i] = datetime.strptime(t[3], date_time_format_str)

	etf = {}	# key = (u_id, p_id), value = maximum difference between reviews (u_id, p_id) and first review of the product
	for i, d in new_product_data.items():
		all_d = copy.deepcopy(d)
		if i in product_data:
			all_d = d + product_data[i]
		for t in all_d:
			td = datetime.strptime(t[3], date_time_format_str) - first_time_product[i]
			if (t[0], i) not in etf:
				etf[(t[0], i)] = td
# find the largest td for the review
			elif td > etf[(t[0], i)]:
				etf[(t[0], i)] = td
	
	for k, v in etf.items():
		if v.days > 7*30:
			etf[k] = 0
		elif 1.0 - v.days / (7 * 30) > beta_3:
			etf[k] = 1
		else:
			etf[k] = 0
	return etf

def ISR(user_data):
	"""
		Check if a user posts only one review
	"""
	isr = {}
	for i, d in user_data.items():
		# go through all review of this user
		for t in d:
			if len(d) == 1:
				isr[(i, t[0])] = 1
			else:
				isr[(i, t[0])] = 0
	return isr

def iISR(user_data, new_user_data):
	isr = {}
	for i, d in new_user_data.items():
		all_d = copy.deepcopy(d)
		if i in user_data:
			all_d = d + user_data[i]
		# go through all review of this user
		for t in all_d:
			if len(all_d) == 1:
				isr[(i, t[0])] = 1
			else:
				isr[(i, t[0])] = 0
	return isr

def low_level_text_features(review_filename, output_filename):
	"""
		Since extracting low level text feature can take some time, we extract and write the features to files for later use.
		Output to file lines of review features, in the same order as the review text file.

		List of features:
			1. word counts in reviews
			2. ratio of all Capital words to all words
			3. ratio of capital letters to all letters
			4. PP1: number of first person names.
			5. RES: ratio of sentences with exclamations.
		Args:
			review_filename: the review text file
			output_filename: the json file that contains the dictionary = {key = (uid, pid): value = dict(key = text feature name, value = feature value)}
	"""
# prepare various lexicons
	PP1_filename = '1PP-words.txt'

	PP1 = set()

	with open(PP1_filename, 'r') as f:
		for line in f:
			PP1.add(line.strip())
    
	tag = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'TO', 'UH', 'PDT', 'SYM', 'RP']
	noun = ['NN', 'NNS', 'NP', 'NPS']
	adj = ['JJ', 'JJR', 'JJS']
	pronoun = ['PP', 'PP$', 'WP', 'WP$']
	verb = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
	adverb = ['RB', 'RBR', 'RBS', 'WRB']
    
	swn_filename= 'SentiWordNet_3.0.0_20100705.txt'
	swn = SentiWordNetCorpusReader(swn_filename)

	review_text_features = dict()

# use the rt mode to read strings instead of binary
	# infile = gzip.open(review_filename, 'rt')
	line_num = 0
	with codecs.open(review_filename, 'r', encoding='utf8') as infile:
		for line in infile:
			line_num += 1
			if line_num % 10000 == 0:
				print (line_num)

			items = re.split("\s+", line, 3)
			u_id = items[0]
			p_id = items[1]
			time = items[2]
			texts = items[3]

			words = nltk.word_tokenize(texts)
			
			lower_words = texts.lower()

			wordcount = float(len(words))
			
			allCapitalCount = 0
			
			PP1Count = 0

	# count the number of capital letters
			countCapital = sum(1 for c in line if c.isupper())
			countAllChars = sum(1 for c in line)

			for w in words:
				# count the numebr of words that have all capital letters
				if w.isupper():
					allCapitalCount += 1
				# count the number of 1st person words
				if w in PP1:
					PP1Count += 1

	# find ratios of subjective and objective words
			
			# objCount = 0
			# subCount = 0
			# for w in word_tag:
			# 	word = w[0]
			# 	pos_tag = w[1]
			# 	if not pos_tag in tag:
			# 		if pos_tag in noun:
			# 			pos_Char = 'n'
			# 		if pos_tag in adj:
			# 			pos_Char = 'a'
			# 		if pos_tag in pronoun:
			# 			pos_Char = 'p'
			# 		if pos_tag in verb:
			# 			pos_Char = 'v'
			# 		if pos_tag in adverb:
			# 			pos_Char = 'r'
			# 		else:
			# 			pos_Char = 'none'
			# 	try:
			# 		if pos_Char == 'none':
			# 			s = swn.senti_synsets(word)
			# 		else:
			# 			s = swn.senti_synsets(word, pos_Char)
			# 		scores = list(s)[0]
			# 		if score.obj_score > 0.5:
			# 			objCount += 1
			# 		elif score.neg_score + score.pos_score > 0.5:
			# 			subCount += 1
			# 	except:
			# 		pass
			
			# if objCount + subCount > 0:
			# 	ratioObj = float(objCount) / (objCount + subCount)
			# 	ratioSub = float(subCount) / (objCount + subCount)
			# else:
			# 	ratioObj = 0.0
			# 	ratioSub = 0.0

			sents = nltk.sent_tokenize(texts)
			countExc = 0
			countSent = 0
			for sentence in sents:
			    countSent += 1
			    if '!' in sentence:
			        countExc += 1
			if countSent > 0:
			    ratioExcSent = float(countExc)/countSent
			else:
			    ratioExcSent = 0.0

			# feature vector for a review: number of words, ratio of allCap words, ratio of allCap Chars, number of 1st persons, number of 2nd persons, ratio of sentences end with exclamations
			review_text_features[(u_id, p_id)] = {"L":wordcount,
				"PCW" : allCapitalCount / wordcount,
				"PC" : float(countCapital) / countAllChars,
				"PP1" : PP1Count,
				"RES" : ratioExcSent}
	
	with open(output_filename, 'wb') as out_f:
		pickle.dump(review_text_features, out_f)

	print ('%d lines processed.' % line_num)
	print ('Text features written to %s' % output_filename)


def add_feature(existing_features, new_features, feature_names):
	"""
		Add or update feature(s) of a set of nodes of the same type to the existing feature(s).
		If a feature of a node is already is existing_features, then the new values will replace the existing ones.
		Args:
			existing_features: a dictionary {node_id:dict{feature_name:feature_value}}
			new_features: new feature(s) to be added. A dict {node_id: list of feature values}
			feature_names: the name of the new feature. A list of feature names, in the same order of the list of feature values in new_features
	"""

	for k, v in new_features.items():
		# k is the node id and v is the feature value
		if k not in existing_features:
			existing_features[k] = dict()
		# add the new feature to the dict of the node
		for i in range(len(feature_names)):
			if len(feature_names) > 1:
				existing_features[k][feature_names[i]] = v[i]
			else:
				existing_features[k][feature_names[i]] = v

def construct_all_features(user_data, prod_data, text_features):
	"""
	    Main entry to feature construction.
	    Args:
	        metadata_filename:
	        text_feature_filename:
	    Return:
	        user, product and review features
	"""

	# key = user id, value = dict of {feature_name: feature_value}
	UserFeatures={}
	# key = product id, value = dict of {feature_name: feature_value}
	ProdFeatures={}

	# go through feature functions
	#print ('\nadding user and product features......\n')
#new feature
	uf = MNR(user_data)
	add_feature(UserFeatures, uf, ["MNR"])
	pf = MNR(prod_data)
	add_feature(ProdFeatures, pf, ["MNR"])

	uf = PR_NR(user_data)
	add_feature(UserFeatures, uf, ["PR", "NR"])
	pf = PR_NR(prod_data)
	add_feature(ProdFeatures, pf, ["PR", "NR"])

	uf = avgRD_user(user_data, prod_data)
	add_feature(UserFeatures, uf, ["avgRD"])
	pf = avgRD_prod(prod_data)
	add_feature(ProdFeatures, pf, ["avgRD"])
#new feature
	uf = BST(user_data)
	add_feature(UserFeatures, uf, ["BST"])

	uf = ERD(user_data)
	add_feature(UserFeatures, uf, ["ERD"])
	pf = ERD(prod_data)
	add_feature(ProdFeatures, pf, ["ERD"])
#new feature
	uf = ETG(user_data)
	add_feature(UserFeatures, uf, ["ETG"])
	pf = ETG(prod_data)
	add_feature(ProdFeatures, pf, ["ETG"])

	#MN: Jan 7, 2018 - we don't deal with text-based features
	# uf = RL(user_data, text_features)
	# add_feature(UserFeatures, uf, ['RL'])
	# pf = RL(prod_data, text_features, isUser = False)
	# add_feature(ProdFeatures, pf, ['RL'])

# go through review features
	#print ('\nadding review features......\n')
	ReviewFeatures = {}
	rf = RD(prod_data)
	add_feature(ReviewFeatures, rf, ['RD'])

	rf = EXT(prod_data)
	add_feature(ReviewFeatures, rf, ['EXT'])

	rf = DEV(prod_data)
	add_feature(ReviewFeatures, rf, ['DEV'])

	rf = ETF(prod_data)
	add_feature(ReviewFeatures, rf, ['ETF'])

	rf = ISR(user_data)
	add_feature(ReviewFeatures, rf, ['ISR'])

# # add low level text features
# 	print ('\nadding low level review features......\n')
# 	for k, v in text_features.items():
# 		# k is a tuple (u_id, p_id)
# 		# v is a dict (key=feature name, value = feature value)
# 		names = [name for name, value in v.items()]
# 		values = [value for name, value in v.items()]

# # recall that the arguments to add_feature
# 		add_feature(ReviewFeatures, {k:values}, names)

	return UserFeatures, ProdFeatures, ReviewFeatures

def update_all_features(user_data, new_user_data, prod_data, new_product_data, text_features, UserFeatures, ProdFeatures, ReviewFeatures):
	""" Construct features using the new data (new_user_data, new_product_data) and update them to UserFeatures, ProdFeatures and ReviewFeatures
	"""
	# go through feature functions
	uf = iMNR(user_data, new_user_data)
	add_feature(UserFeatures, uf, ["MNR"])
	pf = iMNR(prod_data, new_product_data)
	add_feature(ProdFeatures, pf, ["MNR"])

	uf = iPR_NR(user_data, new_user_data)
	add_feature(UserFeatures, uf, ["PR", "NR"])
	pf = iPR_NR(prod_data, new_product_data)
	add_feature(ProdFeatures, pf, ["PR", "NR"])
	
	uf = iavgRD_user(user_data, new_user_data, prod_data, new_product_data)
	add_feature(UserFeatures, uf, ["avgRD"])
	pf = iavgRD_prod(prod_data, new_product_data)
	add_feature(ProdFeatures, pf, ["avgRD"])

	uf = iBST(user_data, new_user_data)
	add_feature(UserFeatures, uf, ["BST"])

	uf = iERD(user_data, new_user_data)
	add_feature(UserFeatures, uf, ["ERD"])
	pf = iERD(prod_data, new_product_data)
	add_feature(ProdFeatures, pf, ["ERD"])

	uf = iETG(user_data, new_user_data)
	add_feature(UserFeatures, uf, ["ETG"])
	pf = iETG(prod_data, new_product_data)
	add_feature(ProdFeatures, pf, ["ETG"])

	rf = iRD(prod_data, new_product_data)
	add_feature(ReviewFeatures, rf, ['RD'])

	rf = iEXT(prod_data, new_product_data)
	add_feature(ReviewFeatures, rf, ['EXT'])

	rf = iDEV(prod_data, new_product_data)
	add_feature(ReviewFeatures, rf, ['DEV'])

	rf = iETF(prod_data, new_product_data)
	add_feature(ReviewFeatures, rf, ['ETF'])

	rf = iISR(user_data, new_user_data)
	add_feature(ReviewFeatures, rf, ['ISR'])
	
	return UserFeatures, ProdFeatures, ReviewFeatures

def calculateNodePriors(feature_names, features_py, when_suspicious):
    """
        Calculate priors of nodes P(y=1|node) using node features.
        Args:
            feature_names: a list of feature names for a particular node type.
			features_py: a dictionary with key = node_id and value = dict of feature_name:feature_value
            when_suspicious: a dictionary with key = feature name and value = 'H' (the higher the more suspicious) or 'L' (the opposite)
        Return:
            A dictionary with key = node_id and value = S score (see the SpEagle paper for the definition)
    """
    
    priors = {}
    for node_id,v in features_py.items():
        priors[node_id] = 0
        
    for f_idx, fn in enumerate(feature_names):
        
        fv_py = []
        for node_id,v in features_py.items():
            if fn not in v:
                fv_py.append((node_id, -1))
            else:
                fv_py.append((node_id, v[fn]))
        fv_py = sorted(fv_py, key=lambda x:x[1])
       
        i = 0
        while i < len(fv_py):
            start = i
            end = i + 1
            while end < len(fv_py) and fv_py[start][1] == fv_py[end][1]:
                end += 1
            i = end

            for j in range(start, end):
                node_id = fv_py[j][0]
                if fv_py[j][0] == -1:
                    priors[node_id] += pow(0.5, 2)
                    continue
                if when_suspicious[fn] == '+1':
                    priors[node_id] += pow( (1.0 - float(start + 1) / len(fv_py)), 2)
                else:
                    priors[node_id] += pow( float(end) / len(fv_py), 2)
            
    for node_id,v in features_py.items():
        priors[node_id] = 1.0 - math.sqrt(priors[node_id] / len(feature_names))
        if priors[node_id] > 0.999:
            priors[node_id] = 0.999
        elif priors[node_id] < 0.001:
            priors[node_id] = 0.001
    return priors

def construct_user_features(user_data, prod_data):

	UserFeatures = {}

	uf = MNR(user_data)
	add_feature(UserFeatures, uf, ["MNR"])

	uf = PR_NR(user_data)
	add_feature(UserFeatures, uf, ["PR", "NR"])

	uf = avgRD_user(user_data, prod_data)
	add_feature(UserFeatures, uf, ["avgRD"])

	uf = BST(user_data)
	add_feature(UserFeatures, uf, ["BST"])

	uf = ERD(user_data)
	add_feature(UserFeatures, uf, ["ERD"])

	uf = ETG(user_data)
	add_feature(UserFeatures, uf, ["ETG"])

	return UserFeatures

def construct_review_features(user_data, prod_data):

	ReviewFeatures = {}

	rf = RD(prod_data)
	add_feature(ReviewFeatures, rf, ['RD'])

	rf = EXT(prod_data)
	add_feature(ReviewFeatures, rf, ['EXT'])

	rf = DEV(prod_data)
	add_feature(ReviewFeatures, rf, ['DEV'])

	rf = ETF(prod_data)
	add_feature(ReviewFeatures, rf, ['ETF'])

	rf = ISR(user_data)
	add_feature(ReviewFeatures, rf, ['ISR'])

	return ReviewFeatures

def new_priors(user_data, prod_data, feature_config):

	UserPriors = {}
	ProdPriors = {}
	ReviewPriors = {}

	review_feature_list = ['RD', 'EXT', 'EXT', 'DEV', 'ETF', 'ISR']
	user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']
	product_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'ERD', 'ETG']

	text_features = []
	UserFeatures, ProdFeatures, ReviewFeatures = construct_all_features(user_data, prod_data, text_features)

	ReviewPriors = calculateNodePriors(review_feature_list, ReviewFeatures, feature_config)
	UserPriors = calculateNodePriors(user_feature_list, UserFeatures, feature_config)
	ProdPriors = calculateNodePriors(product_feature_list, ProdFeatures, feature_config)

	return [UserPriors, ProdPriors, ReviewPriors]
if __name__ == '__main__':
	
	review_feature_list = ['RD', 'EXT', 'EXT', 'DEV', 'ETF', 'ISR']
	user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']
	product_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'ERD', 'ETG']

# path to the folder containing the files
	prefix = '/Users/fgreichert/Documents/Lehigh/D4I/data/'

	UserPriors = {}
	ProdPriors = {}
	ReviewPriors = {}

	# raw data file names
	metadata_filename = prefix + 'metadata.gz'
	review_filename = prefix + 'reviewContent'

	# feature file names
	user_feature_filename = prefix + 'UserFeatures.pickle'
	prod_feature_filename = prefix + 'ProdFeatures.pickle'
	review_feature_filename = prefix + 'ReviewFeatures.pickle'

	#model file names
	prod_model = prefix + 'prod_model.pkl'
	user_model = prefix + 'user_model.pkl'
	review_model = prefix + 'review_model.pkl'

	# prior file names
	user_prior_filename = prefix + 'UserPriors.pickle'
	prod_prior_filename = prefix + 'ProdPriors.pickle'
	review_prior_filename = prefix + 'ReviewPriors.pickle'

	# output file names
	text_feature_filename = prefix + 'text_features.pickle'

	# feature configuration
	feature_suspicious_filename = 'feature_configuration.txt'
	
	# uid2row = {}
	# with open(prefix + 'user_id_list.txt', 'r') as f:
	# 	i = 0
	# 	for line in f:
	# 		uid2row[line.strip()] = i
	# 		i += 1

# low level text features
	# # just need to construct text features once
	# print ('Starting constructing low level text features\n')
	# low_level_text_features(review_filename, text_feature_filename)

	# # can just load the constructed low level text features
	# tf = load_text_feature(text_feature_filename)
	# print ('Finished constructing low level text features\n')

# all high level features
	# print ('Starting constructing high level user, product and review features\n')
	user_data, prod_data = read_graph_data(metadata_filename)

	# missing_user_id = []
	# for k,v in user_data.items():
	# 	if k not in uid2row:
	# 		missing_user_id.append(k)
	# print (missing_user_id)
	
	# text_features = []
	# text_features = load_text_feature(text_feature_filename)
	# UserFeatures, ProdFeatures, ReviewFeatures = construct_all_features(user_data, prod_data, text_features)

	# print(user_feature_filename)
	# with open(user_feature_filename, 'wb') as f:
	# 	pickle.dump(UserFeatures, f)

	# with open(prod_feature_filename, 'wb') as f:
	# 	pickle.dump(ProdFeatures, f)

	# with open(review_feature_filename, 'wb') as f:
	# 	pickle.dump(ReviewFeatures, f)
	# print ('Finished constructing high level user, product and review features\n')


# use regression models to predict priors from feature vectors
	#with open(user_model, 'rb') as f:
	#	user_model = pickle.load(f)
	#with open(prod_model, 'rb') as f:
	#	prod_model = pickle.load(f)
	#with open(review_model, 'rb') as f:
	#	review_model = pickle.load(f)
	#for key, value in UserFeatures.items():
	#	feature_vector = [n for m, n in value.items()]
	#	feature_vector = np.reshape(feature_vector, (-1, 7))
	#	Priors = user_model.predict(feature_vector)
	#	UserPriors[key] = Priors[0]

	#for key, value in ProdFeatures.items():
	#	feature_vector = [n for m, n in value.items()]
	#	feature_vector = np.reshape(feature_vector, (-1, 6))
	#	Priors = prod_model.predict(feature_vector)
	#	ProdPriors[key] = Priors[0]

	#for key, value in ReviewFeatures.items():
	#	feature_vector = [n for m, n in value.items()]
	#	feature_vector = np.reshape(feature_vector, (-1, 5))
	#	U_Priors = review_model.predict(feature_vector)
	#	ReviewPriors[key] = Priors[0]

	#print(UserPriors)
	
# Priors
	# # print ('Start calculating user, product and review priors.\n')

	# feature_config = load_feature_config('/Users/fgreichert/Documents/Lehigh/D4I/data/', feature_suspicious_filename)
	# #print (feature_config)

	# with open(user_feature_filename, 'rb') as f:
	# 	user_features = pickle.load(f)
	
	# # Replace this line to do the regressions
	# user_priors = calculateNodePriors(review_feature_list, user_features, feature_config)
	# with open(user_prior_filename, 'wb') as f:
	# 	pickle.dump(user_priors, f)

	# with open(prod_feature_filename, 'rb') as f:
	# 	prod_features = pickle.load(f)
	# prod_priors = calculateNodePriors(product_feature_list, prod_features, feature_config)
	# with open(prod_prior_filename, 'wb') as f:
	# 	pickle.dump(prod_priors, f)

	# with open(review_feature_filename, 'rb') as f:
	# 	review_features = pickle.load(f)
	# review_priors = calculateNodePriors(review_feature_list, review_features, feature_config)
	# with open(review_prior_filename, 'wb') as f:
	# 	pickle.dump(review_priors, f)
	# print ('Finished calculating user, product and review priors.\n')


# Extract label data (Create this function)
	# x  - features of reviews and accounts (each is a row)
	# y is labeled spam not spam (should be balanced)
	# 

	# import user and review features
	with open(user_feature_filename, 'rb') as f:
		user_features = pickle.load(f)
	
	with open(review_feature_filename, 'rb') as f:
		review_features = pickle.load(f)

	# x denotes features
	review_X = review_features.copy()
	user_X 	 = user_features.copy()

	# y denotes labels
	review_Y = {}
	user_Y = {}

	# Initialize arrays to hold indexes of spam and not spam to ensure training sets are balanced
	user_spam_idx = []
	user_real_idx = []

	# reviews use a tuple as the key, so must use two arrays for each spam and non spam
	# idx1 holds user key
	review_spam_idx1 = []
	# idx2 holds product key
	review_spam_idx2 = []
	review_real_idx1 = []
	review_real_idx2 = []

	# Define limit of how many rows to print
	limit = 10
	
	print('Labeling users and reviews')
	#i = 0
	for k, v in user_data.items():
		# default to user not being a spammer
		user_Y[k] = 0
		# iterate through all reviews associated with each user
		for r in v:
			# print(str(k) + ': ' + str(v))
			# print(str(review_X[str(k), r[0]]))
			
			# if review is spam, set review_Y label to 1 to indicate spam
			if r[2] == 1:
				review_Y[str(k), r[0]] = 1
				review_spam_idx1.append(str(k))
				review_spam_idx2.append(r[0])
				
				# if user makes any review that is labeled spam, label user as a spammer
				if user_Y[k] == 0:
					user_Y[k] = 1
					user_spam_idx.append(k)

			# if review is non spam, set label to 0
			else:
				review_Y[str(k), r[0]] = 0
				review_real_idx1.append(str(k))
				review_real_idx2.append(r[0])

		# if user is not a spammer
		if user_Y[k] == 0:
			user_real_idx.append(k)
		# i += 1
		# if i >= limit:
		# 	break


	# print('Review x & y:')
	# i = 0
	# for k, v in review_X.items():
	# 	i += 1
	# 	if i <= limit:
	# 		print(str(k) + '  X: ' + str(v) + '  Y: ' + str(review_Y[k]))

	# print('User x & y:')
	# i = 0
	# for k, v in user_X.items():
	# 	i += 1
	# 	if i <= limit:
	# 		print(str(k) + '  X: ' + str(v) + '  Y: ' + str(user_Y[k]))

	# assert length of vectors is the same
	if (len(user_X) != len(user_Y)) | (len(review_X) != len(review_Y)):
		print('Error: lengths of features and labels are not equal')
		print('user_X: ' + str(len(user_X)))
		print('user_Y: ' + str(len(user_Y)))
		print('Review_X: ' + str(len(review_X)))
		print('Review_Y: ' + str(len(review_Y)))
	else:
		print('Features and labels created successfully')
		print('Users: ' + str(len(user_X)))
		print('Reviews: ' + str(len(review_X)))

		# Define size of spammers and spam reviews (Training data will be twice the size since it will have spam and non spam)
		spammers = 200
		spams = 500

		# randomly get a sequence of user keys for 200 spammers and non spammers 
		np.random.seed(22)
		user_spam_idx = np.random.choice(user_spam_idx, spammers, False)
		np.random.seed(22)
		user_real_idx = np.random.choice(user_real_idx, spammers, False)

		# randomly get a sequence of review keys for 500 spam and non spam reviews
		# since the review key is a tuple, it is important to set seed so that the tuples stay together when they are randomized
		np.random.seed(22)
		review_spam_idx1 = np.random.choice(review_spam_idx1, spams, False)
		np.random.seed(22)
		review_spam_idx2 = np.random.choice(review_spam_idx2, spams, False)
		np.random.seed(22)
		review_real_idx1 = np.random.choice(review_real_idx1, spams, False)
		np.random.seed(22)
		review_real_idx2 = np.random.choice(review_real_idx2, spams, False)

	#----------------------------------------------

	# 	# Define new dictionaries to hold training  data
	# 	user_train_X = {}
	# 	user_train_Y = {}
	# 	review_train_X = {}
	# 	review_train_Y = {}
		
	# 	# Get (spammers) ammount of spam and non spam users for the training data
	# 	for i in range(spammers):
	# 		k = user_spam_idx[i]
	# 		user_train_X[k] = user_X.pop(k)
	# 		user_train_Y[k] = user_Y.pop(k)
	# 		k = user_real_idx[i]
	# 		user_train_X[k] = user_X.pop(k)
	# 		user_train_Y[k] = user_Y.pop(k)
		
	# 	# test data is the remaining records
	# 	user_test_X = user_X.copy()
	# 	user_test_Y = user_Y.copy()

	# 	# Get (spams) ammount of spam and non spam reviews for the training data
	# 	for i in range(spams):
	# 		uk = review_spam_idx1[i]
	# 		pk =  review_spam_idx2[i]
	# 		review_train_X[uk, pk] = review_X.pop(uk, pk)
	# 		review_train_Y[uk, pk] = review_Y.pop(uk, pk)
	# 		uk = review_real_idx1[i]
	# 		pk =  review_real_idx2[i]
	# 		review_train_X[uk, pk] = review_X.pop(uk, pk)
	# 		review_train_Y[uk, pk] = review_Y.pop(uk, pk)

	# 	# test data is the remaining records
	# 	review_test_X = review_X.copy()
	# 	review_test_Y = review_Y.copy()
	# 	print('Data divided into train and test')

	# Define new dictionaries to hold training  data
		user_train_X = [] #initialized to arrays
		user_train_Y = []
		user_test_X = []
		user_test_Y = []
		
		review_train_X = []
		review_train_Y = []
		review_test_X = []
		review_test_Y = []
		
		
		user_feature_names = {'MNR': 0.05555555555555555, 'PR': 1.0, 'NR': 0.0, 'avgRD': 0.36013986013986, 'BST': 1.0, 'ERD': 0.0, 'ETG': 0}
		review_feature_names = {'RD': 2.0, 'EXT': 1, 'DEV': 0, 'ETF': 0, 'ISR': 1}

		# Get (spammers) ammount of spam and non spam users for the training data
		for i in range(spammers):
			k = user_spam_idx[i]
			# user_train_X[k] = user_X.pop(k)
			user_feature_vector = [user_X[k][name] for name in user_feature_names.keys()]
			user_train_X.append(user_feature_vector)
			# user_train_Y[k] = user_Y.pop(k)
			user_train_Y.append(user_Y[k])
		
		#for real users
		for i in range(spammers):
			k = user_real_idx[i]
			user_feature_vector = [user_X[k][name] for name in user_feature_names.keys()]
			user_train_X.append(user_feature_vector)
			# user_train_X[k] = user_X.pop(k)
			user_train_Y.append(user_Y[k])
			# user_train_Y[k] = user_Y.pop(k)
		
		#populate test cases
		for k in user_features.keys():
			if k not in user_spam_idx and k not in user_real_idx:
				user_feature_vector = [user_X[k][name] for name in user_feature_names.keys()]
				user_test_X.append(user_feature_vector)
				user_test_Y.append(user_Y[k])
		
		user_test_X = np.array(user_test_X)
		user_test_Y = np.array(user_test_Y)
		user_train_X = np.array(user_train_X)
		user_train_Y = np.array(user_train_Y)
		# print(user_train_X.shape)
		# print(user_train_Y.shape)

		# Get (spams) ammount of spam and non spam reviews for the training data
		#for fake reviews
		for i in range(spams):
			uk = review_spam_idx1[i]
			pk =  review_spam_idx2[i]
			review_feature_vector = [review_X[(uk, pk)][name] for name in review_feature_names.keys()]
			#review_train_X[uk, pk] = review_X.pop(uk, pk) #Read Only structure, not necessary to modify
			review_train_X.append(review_feature_vector)
			#review_train_Y[uk, pk] = review_Y.pop(uk, pk)
			review_train_Y.append(review_Y[(uk, pk)])
		
		#for real reviews
		for i in range(spams):
			uk = review_real_idx1[i]
			pk =  review_real_idx2[i]
			review_feature_vector = [review_X[(uk, pk)][name] for name in review_feature_names.keys()]
			review_train_X.append(review_feature_vector)
			review_train_Y.append(review_Y[(uk, pk)])

		#populate test cases
		for (uk, pk) in review_features.keys():
			if uk not in review_spam_idx1 and uk not in review_real_idx1 and pk not in review_spam_idx2 and pk not in review_real_idx2:
				review_feature_vector = [review_X[(uk, pk)][name] for name in review_feature_names.keys()]
				review_test_X.append(review_feature_vector)
				review_test_Y.append(review_Y[(uk, pk)])

		review_test_X = np.array(review_test_X)
		review_test_Y = np.array(review_test_Y)
		review_train_X = np.array(review_train_X)
		review_train_Y = np.array(review_train_Y)

		print('Data divided into train and test')

		
		# assert lengths of vectors are the same
		if (len(review_test_X) != len(review_test_Y)) | (len(review_train_X) != len(review_train_Y)) | (len(user_train_X) != len(user_train_Y)) | (len(user_test_X) != len(user_test_Y)):
			print("Error, train and test dictionaries are not the same lengths")
		else:
			# add logistic regression files to separate folder
			prefix += 'log_reg/'
			# initialize filenames
			user_train_X_filename = prefix + 'user_train_X.pickle'
			user_train_Y_filename = prefix + 'user_train_Y.pickle'
			user_test_X_filename  = prefix + 'user_test_X.pickle'
			user_test_Y_filename  = prefix + 'user_test_Y.pickle'

			review_train_X_filename = prefix + 'review_train_X.picke'
			review_train_Y_filename = prefix + 'review_train_Y.picke'
			review_test_X_filename  = prefix + 'review_test_X.picke '
			review_test_Y_filename  = prefix + 'review_test_Y.picke'
		
			# create files
			with open(user_train_X_filename, 'wb') as f:
				pickle.dump(user_train_X, f)
			with open(user_train_Y_filename, 'wb') as f:
				pickle.dump(user_train_Y, f)
			with open(user_test_X_filename, 'wb') as f:
				pickle.dump(user_test_X, f)
			with open(user_test_Y_filename, 'wb') as f:
				pickle.dump(user_test_Y, f)
			with open(review_train_X_filename, 'wb') as f:
				pickle.dump(review_train_X, f)
			with open(review_train_Y_filename, 'wb') as f:
				pickle.dump(review_train_Y, f)
			with open(review_test_X_filename, 'wb') as f:
				pickle.dump(review_test_X, f)
			with open(review_test_Y_filename, 'wb') as f:
				pickle.dump(review_test_Y, f)
			
			print('Successfully created pickle files')

		

	# # Print out user and review features to examine keys
	# print('\nUser Features:')
	# i = 0
	# for k, v in user_features.items():
	# 	print(str(k) + ': ' + str(v))
	# 	i += 1
	# 	if i >= 10:
	# 		break
	# print('\nReview Features:')
	# i = 0
	# for k, v in review_features.items():
	# 	print(str(k) + ': ' + str(v))
	# 	i += 1
	# 	if i >= 10:
	# 		break

	# # Index into review feature
	# print(str(review_features['201','0']))
	
# Get reviews with more diverse numbers
	# import user and review features
	
	diverse_reviews = {}
	diverse_users = {}

	size = 0
	g = 0
	while size < 20:
		mnr_dict = {}

		for k, v in user_features.items():
			
			g += 1

			#print('Entered for loop')
			mnr = v.get('MNR')
			
			if mnr not in mnr_dict.keys():
				#print(str(mnr))
				#print(mnr_dict.keys())
				# print('New mnr value')
				if k not in diverse_users.keys():
					mnr_dict[mnr] = 1
					# print('New user')
					diverse_users[k] = v
					size += 1 
					print('Size: ' + str(size))
			if size == 20:
				break

	i = 0
	for k, v in review_features.items():

		user = k[0]

		if user in diverse_users.keys():
			diverse_reviews[k] = v


		
	# review key will relate the review to the labeled review in review_Y 
	# use that to pick diverse reviews


	print('\nDiverse Reviews')
	for k, v in diverse_reviews.items():
		print(str(k) + ' ' + str(v))

	# print('\nDiverse users')
	# for k, v in diverse_users.items():
	# 	print(str(k) + ' ' + str(v))


	# get 20 reviews
	# diverse mnr values
	# 10 spam 10 not spam