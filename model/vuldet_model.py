from keras import Input, Model, Sequential
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Concatenate
from stellargraph.layer.gcn import GraphConvolution
import tensorflow as tf
from sklearn.metrics import accuracy_score as acc, precision_score as pr, recall_score as rc, f1_score as f1

import model.load_data as load_data


def gcn_model(n_node_features, n_node_max):
    x_features = Input(shape=(n_node_features, n_node_max))
    x_adjacency = Input(shape=(n_node_features, n_node_max))

    out = Dropout(0.5)(x_features)
    out = GraphConvolution(32, activation='relu', use_bias=True)([out, x_adjacency])

    out = Dropout(0.5)(out)
    out = GraphConvolution(32, activation='relu', use_bias=True)([out, x_adjacency])
    out = Flatten()(out)

    out = Dense(64, activation='relu', use_bias=True)(out)
    out = Dropout(0.5)(out)

    model = Model(inputs=[x_features, x_adjacency], outputs=[out])
    return model


def cnn_model(n_node_max):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu',
                     input_shape=(n_node_max, 134, 1)))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(1, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.5))

    model.add(Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu'))
    model.add(Flatten())

    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.25))

    return model


def mlp_model(n_node_max):
    model = Sequential()
    model.add(Input(shape=(n_node_max, n_node_max, 1)))
    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.2))

    return model


def decisionTree ():
    pass


def randomForest():
    pass

def concatenate_model_mlp_gcn(n_node_max):
    model_gcn = gcn_model(n_node_max, n_node_max=134)
    model_mlp = mlp_model(n_node_max)

    model_out = Concatenate()([model_gcn.output, model_mlp.output])
    model_out = Dense(64, activation='relu')(model_out)
    model_out = Dropout(.5)(model_out)
    model_out = Dense(134, activation='relu')(model_out)
    model_out = Dense(units=1, activation='softmax')(model_out)

    model_out = Model([model_mlp.input, model_gcn.input], model_out)
    model_out.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model_out.summary()
    return model_out


def concatenate_model_mlp_cnn(n_node_max):
    model_mlp = mlp_model(n_node_max)
    model_cnn = cnn_model(n_node_max)

    model_concatenate = Concatenate()([model_mlp.output, model_cnn.output])
    model_concatenate = Dense(units=1, activation='relu')(model_concatenate)
    model_concatenate = Model([model_mlp.input, model_cnn.input], model_concatenate)
    model_concatenate.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam())
    model_concatenate.summary()
    return model_concatenate


if __name__ == '__main__':
    X_test, y_test, X_train, y_train = load_data.load_dataset()

    total = len(y_test)
    neg = (y_test == 0).sum()
    pos = (y_test == 1).sum()
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    # mlp_gcn
    model_concate = concatenate_model_mlp_gcn(n_node_max=134)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy',
        verbose=1,
        patience=5,
        mode='max',
        restore_best_weights=True)

    model_concate.compile(optimizer="adam", loss='categorical_crossentropy')
    model_concate.fit(x=[X_train[0],X_train[1]], y=y_train, validation_data=([X_test[0],X_test[1]], y_test),
              batch_size=256, epochs=1, class_weight=class_weight)
    y_pred = model_concate.predict(X_test)

    # mlp_cnn
    # model = concatenate_model_mlp_cnn(n_node_max=112)
    # model.fit(x=[X_train[0], X_train[1]], y=y_train, validation_data=([X_test[0], X_test[1]], y_test), batch_size=256, epochs=1)
    # y_pred = model.predict([X_test[0], X_test[1]])
    # y_pred = y_pred.argmax(axis=1)

    accuracy = acc(y_test, y_pred)
    precision = pr(y_test, y_pred)
    recall = rc(y_test, y_pred)
    f1 = f1(y_test, y_pred)

    print('accuracy:' + accuracy, 'precision: ' + precision, 'recall: ' + recall, 'f1: ' + f1, sep='\t', flush=True)
    print(accuracy, precision, recall, f1, sep='\t', flush=True, end=('\n' + '=' * 100 + '\n'))
