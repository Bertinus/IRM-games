from torch import nn
from tensorflow import keras


class TorchModel(nn.Module):
    def __init__(self, length, width, height, n_classes):
        super().__init__()

        self.layer1 = nn.Linear(length * width * height, 390)
        self.layer2 = nn.Linear(390, 390)
        self.layer3 = nn.Linear(390, n_classes)
        self.dropout = nn.Dropout(p=0.75)
        self.activation = nn.ELU()

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten the input

        x = self.dropout(x)
        x = self.layer1(x)
        x = self.activation(x)

        x = self.dropout(x)
        x = self.layer2(x)
        x = self.activation(x)

        x = self.layer3(x)

        return x


def keras_model(length, width, height, n_classes):
    return keras.Sequential([keras.layers.Flatten(input_shape=(length, width, height)),
                             keras.layers.Dense(390,
                                                activation='elu',
                                                kernel_regularizer=keras.regularizers.l2(0.00125)),
                             keras.layers.Dropout(0.75),
                             keras.layers.Dense(390,
                                                activation='elu',
                                                kernel_regularizer=keras.regularizers.l2(0.00125)),
                             keras.layers.Dropout(0.75),
                             keras.layers.Dense(n_classes)])


def create_models(module_name, variable_phi, n_env, length, width, height, n_classes):
    assert module_name in ["tensorflow", "pytorch"]
    n = n_env

    if module_name == "tensorflow":
        return [keras_model(length=length, width=width, height=height, n_classes=n_classes) for _ in range(n)]

    else:
        return [TorchModel(length=length, width=width, height=height, n_classes=n_classes) for _ in range(n)]
