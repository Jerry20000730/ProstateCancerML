from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class LinearRegressionHelper:
    def __init__(self, C=1.0, solver="liblinear", max_iter=100):
        """
        Initialize the LinearRegressionHelper with parameters for the logistic regression model.
        """
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver=solver)

    def train(self, X_train, y_train):
        """
        Train the logistic regression model using the training data.
        :param X_train: Training features
        :param y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        print("[INFO] LR Model training completed successfully.")

    def predict(self, X_test):
        """
        Predict the labels for the test data using the trained model.
        :param X_test: Test features
        :return: Predicted labels
        """
        predictions = self.model.predict(X_test)
        print("[INFO] LR Model Predictions made successfully, returning results.")
        return predictions
    
    def predict_proba(self, X):
        """
        Predict probability estimates for positive class (label=1), used for ROC/DCA.
        :param X: Features to predict
        :return: Probability estimates
        """
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using the test data.
        :param X_test: Test features
        :param y_test: True labels for the test data
        :return: Accuracy of the model on the test data
        """
        accuracy = self.model.score(X_test, y_test)
        print("[INFO] LR Model evaluation completed with accuracy: {:.2f}%".format(accuracy * 100))
        return accuracy
    
class SVMHelper:
    def __init__(self, kernel='linear', C=1.0, gamma='auto'):
        """
        Initialize the SVMHelper with parameters for the SVM model.
        """
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)

    def train(self, X_train, y_train):
        """
        Train the SVM model using the training data.
        :param X_train: Training features
        :param y_train: Training labels
        """
        self.model.fit(X_train, y_train)
        print("[INFO] SVM Model training completed successfully.")

    def predict(self, X_test):
        """
        Predict the labels for the test data using the trained model.
        :param X_test: Test features
        :return: Predicted labels
        """
        predictions = self.model.predict(X_test)
        print("[INFO] SVM Predictions made successfully, returning results.")
        return predictions
    
    def predict_proba(self, X):
        """
        Predict probability estimates for positive class (label=1), used for ROC/DCA.
        :param X: Features to predict
        :return: Probability estimates
        """
        return self.model.predict_proba(X)[:, 1]
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model using the test data.
        :param X_test: Test features
        :param y_test: True labels for the test data
        :return: Accuracy of the model on the test data
        """
        accuracy = self.model.score(X_test, y_test)
        print("[INFO] SVM Model evaluation completed with accuracy: {:.2f}%".format(accuracy * 100))
        return accuracy