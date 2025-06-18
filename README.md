# ProstateCancerML
Prostate Cancer Machine Learning Model 基于机器学习的前列腺癌症预测模型

## 如何运行项目
1. 将minio的秘钥放入conf文件夹下，作为minio的秘钥，格式如下
```yaml
minio:
  endpoint: "minioapi.tragicmaster.space"
  access_key: "XXX"
  secret_key: "XXX"
```

2. 安装python依赖
```shell
pip install -r requirements.txt
```

3. 运行python文件
```shell
python3 main.py
```