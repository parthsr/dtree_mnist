from sklearn import tree
import matplotlib.pyplot as plt
import StringIO
import pydotplus
from loader import MNIST

mndata = MNIST('./Datasets')
trainingImages, trainingLabels = mndata.load_training()
testImages, testLabels = mndata.load_testing()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainingImages[:1000], trainingLabels[:1000])

scores = clf.score(testImages,testLabels.tolist())
print "Accuracy: %f " % scores

importances = clf.feature_importances_
importances = importances.reshape((28, 28))

plt.matshow(importances, cmap=plt.cm.hot)
plt.title("Pixel importances for decision tree")
plt.show()

dot_data = StringIO.StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("Dtree.pdf")
print "The Decision Tree was saved!"


