# Chinese NER

基于Bi-GRU + CRF 的中文机构名、人名识别
集成GOOGLE BERT模型

# 下载bert模型
     
    wget -c https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip

放到根目录 **bert_model** 下

# 用法

    # 训练
    # 使用bert模型
    python3 model.py -e train -m bert
    # 使用一般模型
    python3 model.py -e train
    
    
    # 预测
    # 使用bert模型
    python3 model.py -e predict -m bert
    # 使用一般模型
    python3 model.py -e predict


# 例子
    > 金树良先生，董事，硕士。现任北方国际信托股份有限公司总经济师。曾任职于北京大学经济学院国际经济系。1992年7月起历任海南省证券公司副总裁、北京华宇世纪投资有限公司副总裁、昆仑证券有限责任公司总裁、北方国际信托股份有限公司资产管理部总经理及公司总经理助理兼资产管理部总经理、渤海财产保险股份有限公司常务副总经理及总经理、北方国际信托股份有限公司总经理助理。
    >   [
            {
              "begin": 14,
              "end": 26,
              "entity": "北方国际信托股份有限公司",
              "type": "ORG"
            },
            {
              "begin": 70,
              "end": 82,
              "entity": "北京华宇世纪投资有限公司",
              "type": "ORG"
            },
            {
              "begin": 99,
              "end": 111,
              "entity": "北方国际信托股份有限公司",
              "type": "ORG"
            },
            {
              "begin": 160,
              "end": 172,
              "entity": "北方国际信托股份有限公司",
              "type": "ORG"
            },
            {
              "begin": 0,
              "end": 3,
              "entity": "金树良",
              "type": "PER"
            }
        ]