import bentoml

from sklearn import svm, datasets

# Load the dataset 
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train the model
clf = svm.SVC(gamma='scale')
clf.fit(X, y)

# Save model to the bentoML local store
saved_model = bentoml.sklearn.save_model('iris_clf', clf)
print(f'Model saved: {saved_model}')