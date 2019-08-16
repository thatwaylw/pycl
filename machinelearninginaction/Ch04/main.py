'''
@author: laiwei
'''
import bayes
listOPosts,listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
print(myVocabList)
print(listOPosts)
print(listClasses)
print(bayes.setOfWords2Vec(myVocabList, listOPosts[0]))
print(bayes.bagOfWords2VecMN(myVocabList, listOPosts[0]))