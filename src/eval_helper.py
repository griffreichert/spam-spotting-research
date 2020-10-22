from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import numpy as np

def create_ground_truth(user_data):
    """Given user data, return a dictionary of labels of users and reviews
    Args:
        user_data: key = user_id, value = list of review tuples.

    Return:
        user_ground_truth: key = user id (not prefixed), value = 0 (non-spam) /1 (spam)
        review_ground_truth: review id (not prefixed), value = 0 (non-spam) /1 (spam)
        review_ground_truth: product id (not prefixed), value = 0 (non-spam) /1 (spam)
    """
    user_ground_truth = {}
    review_ground_truth = {}
    product_ground_truth = {}

    for user_id, reviews in user_data.items():

        # initial user labels are negative
        user_ground_truth[user_id] = 0

        for r in reviews:
            prod_id = r[0]

            # initial product labels are negative
            if prod_id not in product_ground_truth:
                product_ground_truth[prod_id] = 0
            label = r[2]

            if label == -1:
                review_ground_truth[(user_id, prod_id)] = 1
                user_ground_truth[user_id] = 1
                product_ground_truth[prod_id] = 1
            else:
                review_ground_truth[(user_id, prod_id)] = 0

    return user_ground_truth, review_ground_truth, product_ground_truth


def create_ground_truth_with_labeled_reviews(user_data, labeled_reviews):
    """Given user data, return a dictionary of labels of users and reviews
    Args:
        user_data: key = user_id, value = list of review tuples.
		labeled_reviews: key = review_id, value = 1 (spam) / 0 (non-spam)
    Return:
        user_ground_truth: key = user id (not prefixed), value = 0 (non-spam) /1 (spam)
        review_ground_truth: review id (not prefixed), value = 0 (non-spam) /1 (spam)
    """
    user_ground_truth = {}
    review_ground_truth = {}

    for user_id, reviews in user_data.items():

        user_ground_truth[user_id] = 0

        for r in reviews:
            prod_id = r[0]
            label = r[2]

# skip labeled ones
            if (user_id, prod_id) in labeled_reviews:
                continue

            if label == -1:
                review_ground_truth[(user_id, prod_id)] = 1
                user_ground_truth[user_id] = 1
            else:
                review_ground_truth[(user_id, prod_id)] = 0

    return user_ground_truth, review_ground_truth

def create_evasion_ground_truth(user_data, evasive_spams):
    """Assign label 1 to evasive spams and 0 to all existing reviews; Assign labels to accounts accordingly
    Args:
        user_data: key = user_id, value = list of review tuples.
            user_data can contain only a subset of reviews
            (for example, if some of the reviews are used for training)
            
        evasive_spams: key = product_id, value = list of review tuples

    Return:
        user_ground_truth: key = user id (not prefixed), value = 0 (non-spam) /1 (spam)
        review_ground_truth: review id (not prefixed), value = 0 (non-spam) /1 (spam)
    """
    old_spammers = set()
    old_spams = set()

    user_ground_truth = {}
    review_ground_truth = {}

    # assign label 0 to all existing reviews and users
    for user_id, reviews in user_data.items():
        user_ground_truth[user_id] = 0

        for r in reviews:
            prod_id = r[0]
            label = r[2]
            review_ground_truth[(user_id, prod_id)] = 0

            if label == -1:
                old_spams.add((user_id, prod_id))
                old_spammers.add(user_id)

    # exclude previous spams and spammers, since the controlled accounts are selcted from the normal accounts.
    for r_id in old_spams:
        review_ground_truth.pop(r_id)
    for u_id in old_spammers:
        user_ground_truth.pop(u_id)

    # add label 1 to the evasive spams
    for prod_id, spams in evasive_spams.items():

        for r in spams:
            user_id = r[0]

            review_ground_truth[(user_id, prod_id)] = 1
            # this user now has posted at least one spam, so set its label to 1
            user_ground_truth[user_id] = 1

    return user_ground_truth, review_ground_truth

def create_review_features(userFeatures, prodFeatures, reviewFeatures, userFeatureNames, prodFeatureNames, reviewFeatureNames):
    """
    Concatenate product and user features to each review, as the review's features
    :param userFeatures:
    :param prodFeatures:
    :param reviewFeatures:
    :return:
    """
    review_mat = []
    for r, rf in reviewFeatures.items():
        u_id = r[0]
        p_id = r[1]
        uf = userFeatures[u_id]
        pf = prodFeatures[p_id]

        review_feature_vector = []
        for fn in reviewFeatureNames:
            if fn in rf:
                review_feature_vector.append(rf[fn])
            else:
                review_feature_vector.append(np.inf)

        for fn in prodFeatureNames:
            if fn in pf:
                review_feature_vector.append(pf[fn])
            else:
                review_feature_vector.append(np.inf)

        for fn in userFeatureNames:
            if fn in uf:
                review_feature_vector.append(uf[fn])
            else:
                review_feature_vector.append(np.inf)
        review_mat.append(review_feature_vector)

    review_mat = np.array(review_mat)
    for col in range(review_mat.shape[1]):
        non_inf = np.logical_not(np.isinf(review_mat[:, col]))
        m = np.mean(review_mat[non_inf, col])
        # replace inf with mean
        review_mat[np.isinf(review_mat[:, col]), col] = m

    review_feature_dict = {}
    i = 0
    for r, _ in reviewFeatures.items():
        review_feature_dict[r] = review_mat[i,:]
        i+=1
    return review_feature_dict, review_mat.shape[1]

def evaluate(y, pred_y):
    """
    Revise: test when a key is a review/account.
    Evaluate the prediction of account and review by SpEagle
    Args:
        y: dictionary with key = user_id/review_id and value = ground truth (1 means spam, 0 means non-spam)

        pred_y: dictionary with key = user_id/review_id and value = p(y=spam | x) produced by SpEagle.
                the keys in pred_y must be a subset of the keys in y
    """
    posteriors = []
    ground_truth = []

    for k, v in pred_y.items():
        if k in y:
            posteriors.append(v)
            ground_truth.append(y[k])

    #     print ('number of test reviews: %d' % len(review_ground_truth))
    #     print ('number of test users: %d' % len(user_ground_truth))

    auc = roc_auc_score(ground_truth, posteriors)

    return auc

def roc(y, pred_y):
    """
    Revise: test when a key is a review/account.
    Evaluate the prediction of account and review by SpEagle
    Args:
        y: dictionary with key = user_id/review_id and value = ground truth (1 means spam, 0 means non-spam)

        pred_y: dictionary with key = user_id/review_id and value = p(y=spam | x) produced by SpEagle.
                the keys in pred_y must be a subset of the keys in y
    """
    posteriors = []
    ground_truth = []

    for k, v in pred_y.items():
        if k in y:
            posteriors.append(v)
            ground_truth.append(y[k])

    #     print ('number of test reviews: %d' % len(review_ground_truth))
    #     print ('number of test users: %d' % len(user_ground_truth))

    fpr, tpr, threshold = roc_curve(ground_truth, posteriors)

    return fpr, tpr, threshold

def precision_recall(y, pred_y):
    """
    Revise: test when a key is a review/account.
    Evaluate the prediction of account and review by SpEagle
    Args:
        y: dictionary with key = user_id/review_id and value = ground truth (1 means spam, 0 means non-spam)

        pred_y: dictionary with key = user_id/review_id and value = p(y=spam | x) produced by SpEagle.
                the keys in pred_y must be a subset of the keys in y
    """
    posteriors = []
    ground_truth = []

    for k, v in pred_y.items():
        if k in y:
            posteriors.append(v)
            ground_truth.append(y[k])

    #     print ('number of test reviews: %d' % len(review_ground_truth))
    #     print ('number of test users: %d' % len(user_ground_truth))

    precision, recall, threshold = precision_recall_curve(ground_truth, posteriors)

    return precision, recall, threshold

def precision_top_k(y, pred_y, k):
    """
    Compute the top-k precision, along with the top k items with their true labels.
    Args:
        y: dictionary with key = user_id/review_id and value = ground truth (1 means spam, 0 means non-spam)

        pred_y: dictionary with key = user_id/review_id and value = p(y=spam | x) produced by SpEagle.
                the keys in pred_y must be a subset of the keys in y
    Return:
        topK precision
        items in the top k list
    """
    sorted_list = sorted([(k,v) for k,v in pred_y.items()], key = lambda x:x[1])
    top_k_items = [k for k,v in sorted_list[:k]]
    top_k_labels = [y[k] for k in top_k_items]
    return float(sum(top_k_labels)) / k, top_k_items

def sample_labeled_reviews(user_data, percentage):
    """Sample some reviews as labeled data
    Note that the user_id and product_id may duplicate: there is no u or p prefix to them.
    """
    spams = []
    non_spams = []
    for user_id, reviews in user_data.items():
        for r in reviews:
            label = r[2]
            prod_id = r[0]

            if label == -1:
                spams.append((user_id, prod_id))
            else:
                non_spams.append((user_id, prod_id))

    idx = np.random.choice(len(spams), int(len(spams) * percentage), replace=False)
    labeled_spams = [spams[i] for i in idx]
    idx = np.random.choice(len(non_spams), int(len(non_spams) * percentage), replace=False)
    labeled_non_spams = [non_spams[i] for i in idx]

    return labeled_spams, labeled_non_spams


def reset_priors_with_labels(priors, node_labels):
    """Set node priors (of a single type: user, product or review) according to the given node labels
    Args:
        priors: original node priors
        node_labels: a dictionary with key = node id and value = label (0 non-spam, 1 spam)
    """
    for node_id, label in node_labels.items():
        assert (node_id in priors), 'Review %s not in priors' % node_id
        if label == 1:
            priors[node_id] = 0.999
        elif label == 0:
            priors[node_id] = 0.001


def reset_priors_with_priors(priors, node_priors):
    """Set node priors (of a single type: user, product or review) according to the given node priors
    Args:
        priors: original node priors
        node_priors: a dictionary with key = node id and value = p(y=spam|node)
    """
    for node_id, label in node_priors.items():
        assert (node_id in priors), 'Review %s not in priors' % node_id
        priors[node_id] = node_priors[node_id]


def create_retraining_set(model, node_feature, user_data):
    """Create the training set for the retraining model based on the generated evasive spams
        Args:
        model: the converged SpEagle model 
        node_feature: [UserFeatures, ProdFeatures, ReviewFeatures]
        user_data: key = user_id, value = list of review tuples.
            user_data can contain only a subset of reviews
            (for example, if some of the reviews are used for training)

    Return:
        input_data: key: = review_id, value: a list of feature value
        ground_truth: key = review_id, value = 0 (non-spam) /1 (spam)
    """

    input_data = {}
    ground_truth = {}
    for user, reviews in user_data.items():

        user_feature_list = []
        for name, value in node_feature[0][user].items():
            user_feature_list.append(value)

        for review in reviews:
            feature_list = []
            user_id = 'u'+user
            product_id = 'p'+review[0]
            review_id = (user_id, product_id)
            message1 = model._nodes[review_id]._outgoing[user_id]
            message2 = model._nodes[review_id]._outgoing[product_id]
            message3 = model._nodes[user_id]._outgoing[review_id]
            message4 = model._nodes[product_id]._outgoing[review_id]


            prod_feature_list = []
            for name, value in node_feature[1][review[0]].items():
                prod_feature_list.append(value)

            review_feature_list = []
            for name, value in node_feature[2][(user, review[0])].items():
                review_feature_list.append(value)

            feature_list.append(message1[0])
            feature_list.append(message1[1])
            feature_list.append(message2[0])
            feature_list.append(message2[1])
            feature_list.append(message3[0])
            feature_list.append(message3[1])
            feature_list.append(message4[0])
            feature_list.append(message4[1])

            feature_list = feature_list + user_feature_list + prod_feature_list + review_feature_list

            input_data[(user, review[0])] = feature_list
            if review[2] == -1:
                ground_truth[(user, review[0])] = 1
            else:
                ground_truth[(user, review[0])] = 0

    return input_data, ground_truth
