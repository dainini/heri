import os
import json
import pandas as pd
from datasets import Dataset

class RawDataset(object):
    def __init__(self, conf, logging):
        self.path_rawdata_train = os.path.join(conf.path.dataset, f"{conf.dataprep.raw_dataset}_train.json")
        self.path_rawdata_dev = os.path.join(conf.path.dataset, f"{conf.dataprep.raw_dataset}_dev.json")
        # self.path_rawdata_test = os.path.join(conf.path.dataset, f"{conf.dataprep.raw_dataset}_test.json")
        self.path_ftdata  = os.path.join(conf.path.dataset, conf.dataprep.finetuning_dataset)
        self.logging  = logging
        self.subsample = conf.dataprep.subsample

        self.logging.info(f"[SUBSAMPLE]: {self.subsample}")

    def process_and_save(self, rawdata_path, split_name):
        COLUMNS = ['question', 'explanation']  # 데이터 컬럼

        with open(rawdata_path, 'r') as f:
            raw_dataset = json.load(f)

        data_instances = []

        for item in raw_dataset:
            dict_data = dict()
            dict_data['question'] = item['질문']
            dict_data['explanation'] = item['해설']
            data_instances.append(dict_data)

        data = Dataset.from_pandas(pd.DataFrame(data=data_instances, columns=COLUMNS).sample(frac=self.subsample))
        data = data.map(
            lambda x: {'text':
        f"""당신은 한국의 문화유산 큐레이터입니다. 박물관 관람객의 질문에 친절한 말투로 답하시오. 단, 유물에 관한 허위사실 또는 주어진 정보의 확대해석이 포함되면 안 되며 가치판단을 최대한 배제하십시오.
        ---
        질문: {x['question']}
        
        해설: {x['explanation']}"""})

        data.save_to_disk(f"{self.path_ftdata}/ft_{split_name}_{self.subsample}.hf")

    def make_ft_dataset(self):
        self.process_and_save(self.path_rawdata_train, 'train')
        self.process_and_save(self.path_rawdata_dev, 'dev')
        # self.process_and_save(self.path_rawdata_test, 'test')

