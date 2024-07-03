import numpy as np
from torch import nn, optim, sigmoid, where, from_numpy
from torch.nn import functional
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import expit

from art.utils import check_and_transform_label_format
from apt.utils.datasets.datasets import PytorchData
from apt.utils.models import CLASSIFIER_SINGLE_OUTPUT_CLASS_LOGITS, CLASSIFIER_SINGLE_OUTPUT_BINARY_LOGITS, \
    CLASSIFIER_SINGLE_OUTPUT_BINARY_PROBABILITIES, CLASSIFIER_MULTI_OUTPUT_CLASS_LOGITS, \
    CLASSIFIER_MULTI_OUTPUT_BINARY_LOGITS
from apt.utils.models.pytorch_model import PyTorchClassifier
from art.utils import load_nursery
from apt.utils import dataset_utils


class PytorchModel(nn.Module):

    def __init__(self, num_classes, num_features):
        super(PytorchModel, self).__init__()

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


class PytorchModelBinary(nn.Module):

    def __init__(self, num_features):
        super(PytorchModelBinary, self).__init__()

        self.fc2 = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.Tanh(), )

        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(), )

        self.fc4 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        out = self.fc2(x)
        out = self.fc3(out)
        return self.fc4(out)


class PytorchModelBinarySigmoid(nn.Module):

    def __init__(self, num_features):
        super(PytorchModelBinarySigmoid, self).__init__()

        self.fc2 = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.Tanh(), )

        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(), )

        self.fc4 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Tanh(),
        )

        self.classifier = nn.Sigmoid()

    def forward(self, x):
        out = self.fc2(x)
        out = self.fc3(out)
        out = self.fc4(out)
        return self.classifier(out)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        bce_loss = functional.binary_cross_entropy_with_logits(input, target, reduction='none')

        p = sigmoid(input)
        p = where(target >= 0.5, p, 1 - p)

        modulating_factor = (1 - p) ** self.gamma
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * modulating_factor * bce_loss

        return focal_loss.mean()


def test_pytorch_nursery_state_dict():
    (x_train, y_train), (x_test, y_test), _, _ = load_nursery(test_set=0.5)
    # reduce size of training set to make attack slightly better
    train_set_size = 500
    x_train = x_train[:train_set_size]
    y_train = y_train[:train_set_size]
    x_test = x_test[:train_set_size]
    y_test = y_test[:train_set_size]

    inner_model = PytorchModel(4, 24)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(inner_model.parameters(), lr=0.01)

    model = PyTorchClassifier(model=inner_model,
                              output_type=CLASSIFIER_SINGLE_OUTPUT_CLASS_LOGITS,
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

    inner_model = PytorchModel(4, 24)
    # model = torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(inner_model.parameters(), lr=0.01)

    model = PyTorchClassifier(model=inner_model,
                              output_type=CLASSIFIER_SINGLE_OUTPUT_CLASS_LOGITS,
                              loss=criterion,
                              optimizer=optimizer,
                              input_shape=(24,),
                              nb_classes=4)
    model.fit(PytorchData(x_train.astype(np.float32), y_train), save_entire_model=True, nb_epochs=10)

    score = model.score(PytorchData(x_test.astype(np.float32), y_test))
    print('Base model accuracy: ', score)
    assert (0 <= score <= 1)
    model.load_best_model_checkpoint()
    score = model.score(PytorchData(x_test.astype(np.float32), y_test), apply_non_linearity=expit)
    print('best model accuracy: ', score)
    assert (0 <= score <= 1)


def test_pytorch_predictions_single_label_binary():
    x = np.array([[23, 165, 70, 10],
                  [45, 158, 67, 11],
                  [56, 123, 65, 58],
                  [67, 154, 90, 12],
                  [45, 149, 67, 56],
                  [42, 166, 58, 50],
                  [73, 172, 68, 10],
                  [94, 168, 69, 11],
                  [69, 175, 80, 61],
                  [24, 181, 95, 10],
                  [18, 190, 102, 53],
                  [22, 161, 95, 10],
                  [24, 181, 103, 10],
                  [28, 184, 108, 10]])
    x = from_numpy(x)
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    y = from_numpy(y)
    data = PytorchData(x, y)

    inner_model = PytorchModelBinary(4)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(inner_model.parameters(), lr=0.01)

    model = PyTorchClassifier(model=inner_model, output_type=CLASSIFIER_SINGLE_OUTPUT_BINARY_LOGITS,
                              loss=criterion,
                              optimizer=optimizer, input_shape=(4,),
                              nb_classes=2)
    model.fit(data, save_entire_model=False, nb_epochs=1)

    pred = model.predict(data)
    assert (pred.shape[0] == x.shape[0])
    score = model.score(data)
    assert (0 < score <= 1.0)


def test_pytorch_predictions_single_label_binary_prob():
    x = np.array([[23, 165, 70, 10],
                  [45, 158, 67, 11],
                  [56, 123, 65, 58],
                  [67, 154, 90, 12],
                  [45, 149, 67, 56],
                  [42, 166, 58, 50],
                  [73, 172, 68, 10],
                  [94, 168, 69, 11],
                  [69, 175, 80, 61],
                  [24, 181, 95, 10],
                  [18, 190, 102, 53],
                  [22, 161, 95, 10],
                  [24, 181, 103, 10],
                  [28, 184, 108, 10]])
    x = from_numpy(x)
    y = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1])
    y = from_numpy(y)
    data = PytorchData(x, y)

    inner_model = PytorchModelBinarySigmoid(4)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(inner_model.parameters(), lr=0.01)

    model = PyTorchClassifier(model=inner_model,
                              output_type=CLASSIFIER_SINGLE_OUTPUT_BINARY_PROBABILITIES,
                              loss=criterion,
                              optimizer=optimizer, input_shape=(4,),
                              nb_classes=2)
    model.fit(data, save_entire_model=False, nb_epochs=1)

    pred = model.predict(data)
    assert (pred.shape[0] == x.shape[0])
    score = model.score(data)
    assert (0 < score <= 1.0)


def test_pytorch_predictions_multi_label_cat():
    # This kind of model requires special training and will not be supported using the 'fit' method.
    class MultiLabelCatModel(nn.Module):

        def __init__(self, num_classes, num_features):
            super(MultiLabelCatModel, self).__init__()

            self.fc1 = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.Tanh(), )

            self.classifier1 = nn.Linear(256, num_classes)
            self.classifier2 = nn.Linear(256, num_classes)

        def forward(self, x):
            out1 = self.classifier1(self.fc1(x))
            out2 = self.classifier2(self.fc1(x))
            return out1, out2

    (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset_np()

    # make multi-label categorical
    num_classes = 3
    y_train = check_and_transform_label_format(y_train, nb_classes=num_classes)
    y_test = check_and_transform_label_format(y_test, nb_classes=num_classes)
    y_train = np.column_stack((y_train, y_train))
    y_test = np.stack([y_test, y_test], axis=1)
    test = PytorchData(x_test.astype(np.float32), y_test.astype(np.float32))

    inner_model = MultiLabelCatModel(num_classes, 4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(inner_model.parameters(), lr=0.01)

    # train model
    train_dataset = TensorDataset(from_numpy(x_train.astype(np.float32)), from_numpy(y_train.astype(np.float32)))
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)

    for epoch in range(5):
        # Train for one epoch
        for inputs, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Perform prediction
            model_outputs = inner_model(inputs)

            # Form the loss function
            loss = 0
            for i, o in enumerate(model_outputs):
                t = targets[:, i * num_classes:(i + 1) * num_classes]
                loss += criterion(o, t)

            loss.backward()

            optimizer.step()

    model = PyTorchClassifier(model=inner_model,
                              output_type=CLASSIFIER_MULTI_OUTPUT_CLASS_LOGITS,
                              loss=criterion,
                              optimizer=optimizer,
                              input_shape=(24,),
                              nb_classes=3)

    pred = model.predict(test)
    assert (pred.shape[0] == x_test.shape[0])

    score = model.score(test, apply_non_linearity=expit)
    assert (0 < score <= 1.0)


def test_pytorch_predictions_multi_label_binary():
    class MultiLabelBinaryModel(nn.Module):
        def __init__(self, num_labels, num_features):
            super(MultiLabelBinaryModel, self).__init__()

            self.fc1 = nn.Sequential(
                nn.Linear(num_features, 256),
                nn.Tanh(), )

            self.classifier1 = nn.Linear(256, num_labels)

        def forward(self, x):
            return self.classifier1(self.fc1(x))

    (x_train, y_train), (x_test, y_test) = dataset_utils.get_iris_dataset_np()

    # make multi-label binary
    y_train = np.column_stack((y_train, y_train, y_train))
    y_train[y_train > 1] = 1
    y_test = np.column_stack((y_test, y_test, y_test))
    y_test[y_test > 1] = 1
    test = PytorchData(x_test.astype(np.float32), y_test)

    inner_model = MultiLabelBinaryModel(3, 4)
    criterion = FocalLoss()
    optimizer = optim.RMSprop(inner_model.parameters(), lr=0.01)

    model = PyTorchClassifier(model=inner_model,
                              output_type=CLASSIFIER_MULTI_OUTPUT_BINARY_LOGITS,
                              loss=criterion,
                              optimizer=optimizer,
                              input_shape=(24,),
                              nb_classes=3)
    model.fit(PytorchData(x_train.astype(np.float32), y_train.astype(np.float32)), save_entire_model=False,
              nb_epochs=10)
    pred = model.predict(test)
    assert (pred.shape[0] == x_test.shape[0])

    score = model.score(test, apply_non_linearity=expit)
    assert (score == 1.0)
