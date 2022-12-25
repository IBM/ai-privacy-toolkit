import numpy as np
from torch import nn, optim

from apt.utils.datasets.datasets import PytorchData
from apt.utils.models import ModelOutputType
from apt.utils.models.pytorch_model import PyTorchClassifier
from art.utils import load_nursery


class pytorch_model(nn.Module):

    def __init__(self, num_classes, num_features):
        super(pytorch_model, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.Tanh(), )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Tanh(), )

        self.fc3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Tanh(), )

        self.fc4 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        return self.classifier(out)


def test_nursery_pytorch_state_dict():
    (x_train, y_train), (x_test, y_test), _, _ = load_nursery(test_set=0.5)
    # reduce size of training set to make attack slightly better
    train_set_size = 500
    x_train = x_train[:train_set_size]
    y_train = y_train[:train_set_size]
    x_test = x_test[:train_set_size]
    y_test = y_test[:train_set_size]

    inner_model = pytorch_model(4, 24)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(inner_model.parameters(), lr=0.01)

    model = PyTorchClassifier(model=inner_model, output_type=ModelOutputType.CLASSIFIER_LOGITS, loss=criterion,
                              optimizer=optimizer, input_shape=(24,),
                              nb_classes=4)
    model.fit(PytorchData(x_train.astype(np.float32), y_train), save_entire_model=False, nb_epochs=10)
    model.load_latest_state_dict_checkpoint()
    score = model.score(PytorchData(x_test.astype(np.float32), y_test))
    print('Base model accuracy: ', score)
    assert (0 <= score <= 1)
    # python pytorch numpy
    model.load_best_state_dict_checkpoint()
    score = model.score(PytorchData(x_test.astype(np.float32), y_test))
    print('best model accuracy: ', score)
    assert (0 <= score <= 1)


def test_nursery_pytorch_save_entire_model():

    (x_train, y_train), (x_test, y_test), _, _ = load_nursery(test_set=0.5)
    # reduce size of training set to make attack slightly better
    train_set_size = 500
    x_train = x_train[:train_set_size]
    y_train = y_train[:train_set_size]
    x_test = x_test[:train_set_size]
    y_test = y_test[:train_set_size]

    model = pytorch_model(4, 24)
    # model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    art_model = PyTorchClassifier(model=model, output_type=ModelOutputType.CLASSIFIER_LOGITS, loss=criterion,
                                  optimizer=optimizer, input_shape=(24,),
                                  nb_classes=4)
    art_model.fit(PytorchData(x_train.astype(np.float32), y_train), save_entire_model=True, nb_epochs=10)

    score = art_model.score(PytorchData(x_test.astype(np.float32), y_test))
    print('Base model accuracy: ', score)
    assert (0 <= score <= 1)
    art_model.load_best_model_checkpoint()
    score = art_model.score(PytorchData(x_test.astype(np.float32), y_test))
    print('best model accuracy: ', score)
    assert (0 <= score <= 1)
