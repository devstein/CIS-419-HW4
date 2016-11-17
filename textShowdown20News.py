# Your program must output a table of the following metrics 
# for both classifiers: 
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import svm
from sklearn.metrics.pairwise import linear_kernel
from pprint import pprint
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc


newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle = True)
countVectorizer = CountVectorizer(stop_words='english', lowercase=True)

x_train = countVectorizer.fit_transform(newsgroups_train.data)
x_test = countVectorizer.transform(newsgroups_test.data)

tfidfVectorizer = TfidfTransformer(norm='l2', sublinear_tf=True )

x_train = tfidfVectorizer.fit_transform(x_train)
x_test = tfidfVectorizer.transform(x_test)

#Bayes Shit
bayes = MultinomialNB()
start_time_bayes = time.clock()
bayes = bayes.fit(x_train, newsgroups_train.target)
end_time_bayes = time.clock()
training_time_bayes = end_time_bayes - start_time_bayes 
print "---------------------------"
print "Bayes Training Time: ", training_time_bayes
print "---------------------------"

predictedTrain_bayes = bayes.predict(x_train)
accuracyTrain_bayes = metrics.accuracy_score(newsgroups_train.target, predictedTrain_bayes)
precisionTrain_bayes = metrics.precision_score(newsgroups_train.target, predictedTrain_bayes, average = 'macro')
recallTrain_bayes = metrics.recall_score(newsgroups_train.target, predictedTrain_bayes, average = 'macro')
print "Bayes Train Accuracy Score: ", accuracyTrain_bayes
print "Bayes Train Precision Score: ", precisionTrain_bayes
print "Bayes Train Recall Score: ", recallTrain_bayes
print "---------------------------"

predictedTest_bayes = bayes.predict(x_test)
accuracyTest_bayes = metrics.accuracy_score(newsgroups_test.target, predictedTest_bayes)
precisionTest_bayes = metrics.precision_score(newsgroups_test.target, predictedTest_bayes, average = 'macro')
recallTest_bayes = metrics.recall_score(newsgroups_test.target, predictedTest_bayes, average = 'macro')
print "Bayes Test Accuracy Score: ", accuracyTest_bayes
print "Bayes Test Precision Score: ", precisionTest_bayes
print "Bayes Test Recall Score: ", recallTest_bayes
print "---------------------------"

#SVM Shit
svc = svm.SVC(kernel=linear_kernel, probability=True)

start_time_svc = time.clock()
svc = svc.fit(x_train, newsgroups_train.target)
end_time_svc = time.clock()
training_time_svc = end_time_svc - start_time_svc 
print "SVC Training Time: ", training_time_svc
print "---------------------------"

predictedTrain_svc = svc.predict(x_train)
accuracyTrain_svc = metrics.accuracy_score(newsgroups_train.target, predictedTrain_svc)
precisionTrain_svc = metrics.precision_score(newsgroups_train.target, predictedTrain_svc, average = 'macro')
recallTrain_svc = metrics.recall_score(newsgroups_train.target, predictedTrain_svc, average = 'macro')
print "SVC Train Accuracy Score: ", accuracyTrain_svc
print "SVC Train Precision Score: ", precisionTrain_svc
print "SVC Train Recall Score: ", recallTrain_svc

print "---------------------------"
predictedTest_svc = svc.predict(x_test)
accuracyTest_svc = metrics.accuracy_score(newsgroups_test.target, predictedTest_svc)
precisionTest_svc = metrics.precision_score(newsgroups_test.target, predictedTest_svc, average = 'macro')
recallTest_svc = metrics.recall_score(newsgroups_test.target, predictedTest_svc, average = 'macro')
print "SVC Test Accuracy Score: ", accuracyTest_svc
print "SVC Test Precision Score: ", precisionTest_svc
print "SVC Test Recall Score: ", recallTest_svc
print "---------------------------"

# (a) train & test accuracy
# (b) train & test precision 
# (c) train & test recall 
# (d) training time. Ensure the table is neat and clear.

roc_categories = [ 'comp.graphics', 'comp.sys.mac.hardware', 'rec.motorcycles', 'sci.space', 'talk.politics.mideast']
roc_indexes = np.asarray([newsgroups_train.target_names.index(i) for i in roc_categories])

binarized = label_binarize(newsgroups_test.target, classes = np.unique(newsgroups_test.target))

probs_bayes  = bayes.predict_proba(x_test)
probs_svc = svc.predict_proba(x_test)

fpr_bayes = dict()
tpr_bayes = dict()
roc_auc_bayes = dict()

fpr_svc = dict()
tpr_svc = dict()
roc_auc_svc = dict()


for i in range(len(roc_indexes)):
    # Naive Bayes
    fpr_bayes[i], tpr_bayes[i], _ = roc_curve(binarized[:, roc_indexes[i]], probs_bayes[:, roc_indexes[i]])
    roc_auc_bayes[i] = auc(fpr_bayes[i], tpr_bayes[i])

    # SVM
    fpr_svc[i], tpr_svc[i], _ = roc_curve(binarized[:, roc_indexes[i]], probs_svc[:, roc_indexes[i]])
    roc_auc_svc[i] = auc(fpr_svc[i], tpr_svc[i])

with PdfPages('graphTextClassifierROC.pdf') as pdf:
     plt.figure()
     for i in range(len(roc_indexes)):
         plt.plot(fpr_bayes[i], tpr_bayes[i], label='{0} Bayes curve (Area = {1:0.4f})'
			                                    ''.format(roc_categories[i], roc_auc_bayes[i]))
         plt.plot(fpr_svc[i], tpr_svc[i], label='{0} ROC SVC curve (Area = {1:0.4f})'
			                                    ''.format(roc_categories[i], roc_auc_svc[i]))
	 
     plt.plot([0, 1], [0, 1], 'k--')
     plt.xlim([0.0, 1.0])
     plt.ylim([0.0, 1.05])
     plt.xlabel('False Positive Rate')
     plt.ylabel('True Positive Rate')
     plt.title('ROC Plot for Bayes and SVC Classifiers')
     plt.legend(loc="lower right", prop={'size':6})
     pdf.savefig()
     plt.close()
