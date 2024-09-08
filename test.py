import bentoml

# model latest
# iris_clf:ld5f75tn5wz6vjfr
iris_clf_runner = bentoml.sklearn.get('iris_clf:latest').to_runner()
iris_clf_runner.init_local()
print(iris_clf_runner.predict.run([[5.9, 1.2, 3, 2.5]]))