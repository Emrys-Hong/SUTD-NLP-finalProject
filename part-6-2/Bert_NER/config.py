import torch


class Config:
    def __init__(self) -> None:
        # data hyperparameter
        self.batch_size = 32
        self.shuffle_train_data = True
        self.train_file = "data/partial/train"
        self.dev_file = "data/partial/dev.out"

        self.num_workers = 1  ## workers

        # optimizer hyperparameter
        self.learning_rate = 2e-5
        self.max_grad_norm = 1

        # training
        self.device = torch.device("cuda:0")
        self.num_epochs = 10
        self.seed = 0

        # model
        self.model_folder = "NER"
        self.bert_model_name = "bert-base-cased"
        self.bert_folder = "bert_model"
        self.labels = ['O', 'I-org', 'I-geo', 'B-geo', 'B-art', 'B-per', 'I-gpe', 'B-nat', 'B-eve', 'I-tim', 'I-art', 'B-tim', 'B-org', 'B-gpe', 'I-per', 'I-eve']
        self.label_map = {'O': 0,
                          'I-org': 1,
                          'I-geo': 2,
                          'B-geo': 3,
                          'B-art': 4,
                          'B-per': 5,
                          'I-gpe': 6,
                          'B-nat': 7,
                          'B-eve': 8,
                          'I-tim': 9,
                          'I-art': 10,
                          'B-tim': 11,
                          'B-org': 12,
                          'B-gpe': 13,
                          'I-per': 14,
                          'I-eve': 15}
