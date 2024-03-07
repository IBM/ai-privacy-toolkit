import numpy as np
from torch import nn, optim, sigmoid, where, from_numpy
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import expit

from apt.utils.datasets.datasets import PytorchData
from apt.utils.models import ModelOutputType
from apt.utils.models.pytorch_model import PyTorchClassifier
from art.utils import load_nursery
from apt.utils import dataset_utils


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


def test_pytorch_nursery_state_dict():
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

    model = PyTorchClassifier(model=inner_model,
                              output_type=ModelOutputType.CLASSIFIER_SINGLE_OUTPUT_CLASS_LOGITS,
                              loss=criterion,
                              optimizer=optimizer,
                              input_shape=(24,),
                              nb_classes=4)
    model.fit(PytorchData(x_train.astype(np.float32), y_train), save_entire_model=False, nb_epochs=10)
    model.load_latest_state_dict_checkpoint()
    score = model.score(PytorchData(x_test.astype(np.float32), y_test))
    print('Base model accuracy: ', score)
    assert (0 <= score <= 1)
    # python pytorch numpy
    model.load_best_state_dict_checkpoint()
    score = model.score(PytorchData(x_test.astype(np.float32), y_test), apply_non_linearity=expit)
    print('best model accuracy: ', score)
    assert (0 <= score <= 1)


def test_pytorch_nursery_save_entire_model():

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

    art_model = PyTorchClassifier(model=model,
                                  output_type=ModelOutputType.CLASSIFIER_SINGLE_OUTPUT_CLASS_LOGITS,
                                  loss=criterion,
                                  optimizer=optimizer,
                                  input_shape=(24,),
                                  nb_classes=4)
    art_model.fit(PytorchData(x_train.astype(np.float32), y_train), save_entire_model=True, nb_epochs=10)

    score = art_model.score(PytorchData(x_test.astype(np.float32), y_test))
    print('Base model accuracy: ', score)
    assert (0 <= score <= 1)
    art_model.load_best_model_checkpoint()
    score = art_model.score(PytorchData(x_test.astype(np.float32), y_test), apply_non_linearity=expit)
    print('best model accuracy: ', score)
    assert (0 <= score <= 1)


# def test_pytorch_predictions_multi_label_cat():
#     # This kind of model requires special training and will not be supported using the 'fit' method.
#     class multi_label_cat_model(nn.Module):
#
#         def __init__(self, num_classes, num_features):
#             super(multi_label_cat_model, self).__init__()
#
#             self.fc1 = nn.Sequential(
#                 nn.Linear(num_features, 256),
#                 nn.Tanh(), )
#
#             self.classifier1 = nn.Linear(256, num_classes)
#             self.classifier2 = nn.Linear(256, num_classes)
#             self.classifier3 = nn.Linear(256, num_classes)
#
#         def forward(self, x):
#             out1 = self.classifier1(self.fc1(x))
#             out2 = self.classifier2(self.fc1(x))
#             out3 = self.classifier3(self.fc1(x))
#             return out1, out2, out3
#
#     (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset_np()
#
#     # make multi-label categorical
#     y_train = np.column_stack((y_train, y_train, y_train))
#     y_test = np.column_stack((y_test, y_test, y_test))
#     test = PytorchData(x_test.astype(np.float32), y_test.astype(np.float32))
#
#     model = multi_label_cat_model(3, 4)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.01)
#
#     # train model
#     train_dataset = TensorDataset(from_numpy(x_train.astype(np.float32)), from_numpy(y_train.astype(np.float32)))
#     train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
#
#     for epoch in range(5):
#         # Train for one epoch
#         for inputs, targets in train_loader:
#             # Zero the parameter gradients
#             optimizer.zero_grad()
#
#             # Perform prediction
#             model_outputs = model(inputs)[-1]
#
#             # Form the loss function
#             loss = 0
#             for i, o in enumerate(model_outputs):
#                 loss += criterion(o, targets[i])
#
#             loss.backward()
#
#             optimizer.step()
#
#     art_model = PyTorchClassifier(model=model,
#                                   output_type=ModelOutputType.CLASSIFIER_MULTI_OUTPUT_CLASS_LOGITS,
#                                   loss=criterion,
#                                   optimizer=optimizer,
#                                   input_shape=(24,),
#                                   nb_classes=3)
#
#     pred = art_model.predict(test)
#     assert (pred.shape[0] == x_test.shape[0])
#
#     score = art_model.score(test, apply_non_linearity=expit)
#     assert (score == 1.0)


def test_pytorch_predictions_multi_label_binary():
    class multi_label_binary_model(nn.Module):
        def __init__(self, num_labels, num_features):
            super(multi_label_binary_model, self).__init__()

            self.fc1 = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.Tanh(), )

            self.classifier1 = nn.Linear(256, num_labels)

        def forward(self, x):
            return self.classifier1(self.fc1(x))
            # missing sigmoid on each output

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2, alpha=0.5):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, input, target):
            bce_loss = functional.binary_cross_entropy_with_logits(input, target, reduction='none')

            p = sigmoid(input)
            p = where(target >= 0.5, p, 1-p)

            modulating_factor = (1 - p)**self.gamma
            alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha * modulating_factor * bce_loss

            return focal_loss.mean()

    (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset_np()

    # make multi-label binary
    y_train = np.column_stack((y_train, y_train, y_train))
    y_train[y_train > 1] = 1
    y_test = np.column_stack((y_test, y_test, y_test))
    y_test[y_test > 1] = 1
    test = PytorchData(x_test.astype(np.float32), y_test)

    model = multi_label_binary_model(3, 4)
    criterion = FocalLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=0.01)

    art_model = PyTorchClassifier(model=model,
                                  output_type=ModelOutputType.CLASSIFIER_MULTI_OUTPUT_BINARY_LOGITS,
                                  loss=criterion,
                                  optimizer=optimizer,
                                  input_shape=(24,),
                                  nb_classes=3)
    art_model.fit(PytorchData(x_train.astype(np.float32), y_train.astype(np.float32)), save_entire_model=False,
                  nb_epochs=10)
    pred = art_model.predict(test)
    assert (pred.shape[0] == x_test.shape[0])

    score = art_model.score(test, apply_non_linearity=expit)
    assert (score == 1.0)
