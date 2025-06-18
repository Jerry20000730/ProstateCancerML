import sys
from util import util
from const.global_const import *
from dao.minio import MinioProxyClient
from logic.preprocessing import Preprocessor
from logic.plot import plot_roc_train, plot_roc_test, plot_dca
from logic.calculate_metrics import calculate_metrics
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
    svm_model = SVMHelper(kernel="linear", gamma="scale")
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
    step 5: 画图(LR model)
    """
    print("[INFO] Starting to plot LR model train ROC...")
    try:
        train_probs = lr_model.predict_proba(X_train)
        plot_roc_train(y_train, train_probs, title="ROC Curve for LR model\n(train)", file_path="output/roc_train_lr.png")
        print("[INFO] Train ROC plot completed successfully.")
    except Exception as e:
        print("[ERROR] Train ROC plot failed: {}".format(e))
        sys.exit(1)
    
    print("[INFO] Starting to plot LR model test ROC...")
    try:
        test_probs = lr_model.predict_proba(X_test)
        plot_roc_test(y_test, test_probs, title="ROC Curve for LR model\n(test)", file_path="output/roc_test_lr.png")
        print("[INFO] Test ROC plot completed successfully.")
    except Exception as e:
        print("[ERROR] Test ROC plot failed: {}".format(e))
        sys.exit(1)
    
    print("[INFO] Starting to draw LR model DCA...")
    try:
        plot_dca(y_test, test_probs, title="DCA Plot for LR model", file_path="output/dca_plot_lr.png")
        print("[INFO] DCA plot completed successfully.")
    except Exception as e:
        print("[ERROR] DCA plot failed: {}".format(e))
        sys.exit(1)
    
    """
    step 6: 保存模型评估结果到 Excel
    """
    print("[INFO] Starting to save LR model evaluation metrics to Excel...")
    try:
        lr_train_metrics = calculate_metrics(y_train, train_probs, lr_model.predict(X_train), dataset_type="train")
        lr_test_metrics = calculate_metrics(y_test, test_probs, lr_model.predict(X_test), dataset_type="test")
        util.save_metrics_to_excel(metrics=[lr_train_metrics, lr_test_metrics], filepath="output/result/metrics_summary_lr.xlsx")
        print("[INFO] LR model evaluation metrics saved successfully.")
    except Exception as e:
        print("[ERROR] Saving LR model evaluation metrics failed: {}".format(e))
        sys.exit(1)

    """
    step 7: 画图(SVM model)
    """
    print("[INFO] Starting to plot SVM model train ROC...")
    try:
        train_probs_svm = svm_model.predict_proba(X_train)
        plot_roc_train(y_train, train_probs_svm, title="ROC Curve for SVM model\n(train)", file_path="output/roc_train_svm.png")
        print("[INFO] Train ROC plot for SVM completed successfully.")
    except Exception as e:
        print("[ERROR] Train ROC plot for SVM failed: {}".format(e))
        sys.exit(1)
    
    print("[INFO] Starting to plot SVM model test ROC...")
    try:
        test_probs_svm = svm_model.predict_proba(X_test)
        plot_roc_test(y_test, test_probs_svm, title="ROC Curve for SVM model\n(test)", file_path="output/roc_test_svm.png")
        print("[INFO] Test ROC plot for SVM completed successfully.")
    except Exception as e:
        print("[ERROR] Test ROC plot for SVM failed: {}".format(e))
        sys.exit(1)
    
    print("[INFO] Starting to draw SVM model DCA...")
    try:
        plot_dca(y_test, test_probs_svm, title="DCA Plot for SVM model", file_path="output/dca_plot_svm.png")
        print("[INFO] DCA plot for SVM completed successfully.")
    except Exception as e:
        print("[ERROR] DCA plot for SVM failed: {}".format(e))
        sys.exit(1)
    
    """
    step 8: 保存模型评估结果到 Excel
    """
    print("[INFO] Starting to save SVM model evaluation metrics to Excel...")
    try:
        svm_train_metrics = calculate_metrics(y_train, train_probs_svm, svm_model.predict(X_train), dataset_type="train")
        svm_test_metrics = calculate_metrics(y_test, test_probs_svm, svm_model.predict(X_test), dataset_type="test")
        util.save_metrics_to_excel(metrics=[svm_train_metrics, svm_test_metrics], filepath="output/result/metrics_summary_svm.xlsx")
        print("[INFO] SVM model evaluation metrics saved successfully.")
    except Exception as e:
        print("[ERROR] Saving SVM model evaluation metrics failed: {}".format(e))
        sys.exit(1)
    
    print("[INFO] All steps completed successfully!")