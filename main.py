import sys
from util import util
from const.global_const import *
from dao.minio import MinioProxyClient
from logic.preprocessing import Preprocessor
from logic.plot import plot_roc_train, plot_roc_test, plot_dca
from model.model import *

if __name__ == "__main__":
    """
    step 1: 检查源文件是否存在于源目录中
            a. 如果不存在，则从 Minio 下载源文件
            b. 如果存在，则直接使用源文件
    """
    print("[INFO] Pre-starting the training process, check training file...")
    # Check whether the source is in the source directory
    if not util.check_file_exists(SOURCE_FILE):
        print("[INFO] Training file '{}' does not exist in the source directory, start downloading...".format(SOURCE_FILE))
        try:
            # Initialize the Minio client
            minio_client = MinioProxyClient(conf_path="conf/app.yaml", is_https=False)
            # Download the file from Minio
            minio_client.download_file(BUCKET_NAME, OBJECT_NAME, SOURCE_FILE)
            print("[INFO] Training file '{}' downloaded successfully.".format(SOURCE_FILE))
        except Exception as e:
            print("[ERROR] Error downloading source file: {}".format(e))
            sys.exit(1)
    else:
        print("[INFO] Training file '{}' already exists in the source directory, skip downloading.".format(SOURCE_FILE))

    """
    step 2: 开始预处理数据源，切分数据集，25% 测试集，75% 训练集
    """
    preprocessor = Preprocessor(SOURCE_FILE)
    print("[INFO] Starting preprocessing...")
    try:
        X_train, X_test, y_train, y_test = preprocessor.preprocess()
        print("[INFO] Preprocessing completed successfully.")
        print("[INFO] Training data shape: {}".format(X_train.shape))
        print("[INFO] Test data shape: {}".format(X_test.shape))
    except Exception as e:
        print("[ERROR] Preprocessing failed: {}".format(e))
        sys.exit(1)

    """
    step 3: 开始训练模型
    """
    lr_model = LinearRegressionHelper(max_iter=200)
    svm_model = SVMHelper()
    print("[INFO] Starting lr model training...")
    try:
        # Train the logistic regression model
        lr_model.train(X_train, y_train)
        print("[INFO] LR Model training completed successfully.")
    except Exception as e:
        print("[ERROR] LR Model training failed: {}".format(e))
        sys.exit(1)

    print("[INFO] Starting SVM model training...")
    try:
        # Train the SVM model
        svm_model.train(X_train, y_train)
        print("[INFO] SVM Model training completed successfully.")
    except Exception as e:
        print("[ERROR] SVM Model training failed: {}".format(e))
        sys.exit(1)

    """
    step 4: 模型评估
    """
    print("[INFO] Starting model evaluation...")
    try:
        lr_accuracy = lr_model.evaluate(X_test, y_test)
        svm_accuracy = svm_model.evaluate(X_test, y_test)
        print("[INFO] LR Model accuracy: {:.2f}%".format(lr_accuracy * 100))
        print("[INFO] SVM Model accuracy: {:.2f}%".format(svm_accuracy * 100))
    except Exception as e:
        print("[ERROR] Model evaluation failed: {}".format(e))
        sys.exit(1)
    
    """
    step 5: 画图
    """
    print("[INFO] Starting to plot train ROC...")
    try:
        train_probs = lr_model.predict_proba(X_train)
        plot_roc_train(y_train, train_probs)
        print("[INFO] Train ROC plot completed successfully.")
    except Exception as e:
        print("[ERROR] Train ROC plot failed: {}".format(e))
        sys.exit(1)
    
    print("[INFO] Starting to plot test ROC...")
    try:
        test_probs = lr_model.predict_proba(X_test)
        plot_roc_test(y_test, test_probs)
        print("[INFO] Test ROC plot completed successfully.")
    except Exception as e:
        print("[ERROR] Test ROC plot failed: {}".format(e))
        sys.exit(1)
    
    print("[INFO] Starting to draw DCA...")
    try:
        plot_dca(y_test, test_probs)
        print("[INFO] DCA plot completed successfully.")
    except Exception as e:
        print("[ERROR] DCA plot failed: {}".format(e))
        sys.exit(1)
    
    print("[INFO] All steps completed successfully!")