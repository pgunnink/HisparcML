from keras.layers import Permute, MaxPooling2D, LocallyConnected2D, AveragePooling2D, \
    GlobalAveragePooling2D, concatenate, Input, Conv2D, Reshape, Dense, Flatten, \
    Dropout
from keras import Model
from keras.optimizers import Adam
from CustomMetrics import metric_degrees_difference

length_trace = 80
features = 2
N_stations = 1


def baseModelComplicated():
    """
    Dit is het model zoals beschrevn in mijn verslag. De gecommente dingen zijn
    probeersels en voorbeelden van andere manieren om een NN te maken.
    """
    input_traces = Input(shape=(N_stations * 4, length_trace), dtype='float32',
                         name='trace_input')
    reshape_traces = Reshape((N_stations * 4, length_trace, 1))(input_traces)
    input_metadata = Input(shape=(N_stations * 4, features), dtype='float32',
                           name='metadata_input')

    Trace_1 = Conv2D(64, (1, 3), strides=(1, 2), padding='valid',
                     activation='relu', data_format='channels_last',
                     kernel_initializer='he_normal')(reshape_traces)
    Trace_1 = Conv2D(128, (1, 3), strides=(1, 2), padding='valid',
                     activation='relu', data_format='channels_last',
                     kernel_initializer='he_normal')(Trace_1)
    Trace_1_conv = Conv2D(128, (1, 3), strides=(1, 1), padding='valid',
                          activation='relu', kernel_initializer='he_normal')(Trace_1)
    MaxPool1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 1), padding='valid')(Trace_1)

    combined = concatenate([MaxPool1, Trace_1_conv])

    Trace_1 = Conv2D(64, (1, 1), strides=(1, 1), padding='valid',
                     activation='relu', kernel_initializer='he_normal')(combined)
    Trace_1 = Conv2D(128, (1, 3), strides=(1, 1), padding='valid',
                     activation='relu', kernel_initializer='he_normal')(Trace_1)

    Trace_2 = Conv2D(32, (1, 1), strides=(1, 1), padding='valid',
                     activation='relu', kernel_initializer='he_normal')(combined)
    Trace_2 = Conv2D(64, (1, 7), strides=(1, 1), padding='same',
                     activation='relu', kernel_initializer='he_normal')(Trace_2)
    Trace_2 = Conv2D(128, (1, 3), strides=(1, 1), padding='valid',
                     activation='relu', kernel_initializer='he_normal')(Trace_2)

    combined_2 = concatenate([Trace_1, Trace_2])
    MaxPool2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 1))(combined_2)
    Trace_3 = Conv2D(64, (1, 3), strides=(1, 1), padding='valid',
                     activation='relu', kernel_initializer='he_normal')(combined_2)
    combined_3 = concatenate([MaxPool2, Trace_3])

    avg_pooling = AveragePooling2D(pool_size=(1, 3), strides=(1, 1), padding='same')(
        combined_3)
    avg_pooling = Conv2D(64, (1, 1), strides=(1, 1), padding='same',
                         activation='relu', kernel_initializer='he_normal')(avg_pooling)

    path_1 = Conv2D(64, (1, 1), strides=(1, 1), padding='valid',
                    activation='relu', kernel_initializer='he_normal')(combined_3)

    path_2 = Conv2D(100, (1, 1), strides=(1, 1), padding='valid',
                    activation='relu', kernel_initializer='he_normal')(combined_3)
    path_2 = Conv2D(55, (1, 3), strides=(1, 1), padding='same',
                    activation='relu', kernel_initializer='he_normal')(path_2)

    path_3 = Conv2D(32, (1, 1), strides=(1, 1), padding='valid',
                    activation='relu', kernel_initializer='he_normal')(combined_3)
    path_3 = Conv2D(45, (1, 3), strides=(1, 1), padding='same',
                    activation='relu', kernel_initializer='he_normal')(path_3)
    path_3 = Conv2D(80, (1, 3), strides=(1, 1), padding='same',
                    activation='relu', kernel_initializer='he_normal')(path_3)

    result = concatenate([path_1, path_2, path_3, avg_pooling])
    """
    maxpool1 = MaxPooling2D(pool_size=(1,3), strides=(1,2))(result)
    conv_1 = Conv2D(32, (1, 3), strides=(1, 2), padding='valid',
                   activation='relu', kernel_initializer='he_normal')(result)
    block_1 = Conv2D(80, (1, 1), strides=(1, 1), padding='same',
                   activation='relu', kernel_initializer='he_normal')(result)
    block_1 = Conv2D(40, (1, 3), strides=(1, 1), padding='same',
                   activation='relu', kernel_initializer='he_normal')(block_1)
    block_1 = Conv2D(32, (1, 3), strides=(1, 2), padding='valid',
                   activation='relu', kernel_initializer='he_normal')(block_1)

    result = concatenate([maxpool1, conv_1, block_1])

    avg_pooling = AveragePooling2D(pool_size=(1, 3), strides=(1,1), padding='same')(result)
    avg_pooling = Conv2D(64, (1, 1), strides=(1, 1), padding='same',
                   activation='relu', kernel_initializer='he_normal')(avg_pooling)



    path_1 = Conv2D(128, (1, 1), strides=(1, 1), padding='valid',
                   activation='relu', kernel_initializer='he_normal')(result)

    path_2 = Conv2D(64, (1, 1), strides=(1, 1), padding='valid',
                   activation='relu', kernel_initializer='he_normal')(result)
    path_2 = Conv2D(128, (1, 7), strides=(1, 1), padding='same',
                   activation='relu', kernel_initializer='he_normal')(path_2)

    path_3 = Conv2D(20, (1, 1), strides=(1, 1), padding='valid',
                   activation='relu', kernel_initializer='he_normal')(result)
    path_3 = Conv2D(30, (1, 7), strides=(1, 1), padding='same',
                   activation='relu', kernel_initializer='he_normal')(path_3)
    path_3 = Conv2D(100, (1, 7), strides=(1, 1), padding='same',
                   activation='relu', kernel_initializer='he_normal')(path_3)

    result = concatenate([path_1, path_2, path_3, avg_pooling])

    path_1 = Conv2D(64, (1, 1), strides=(1, 1), padding='valid',
                   activation='relu', kernel_initializer='he_normal')(result)

    path_2 = Conv2D(64, (1, 1), strides=(1, 1), padding='valid',
                   activation='relu', kernel_initializer='he_normal')(result)
    path_2 = Conv2D(128, (1, 7), strides=(1, 1), padding='same',
                   activation='relu', kernel_initializer='he_normal')(path_2)

    path_3 = Conv2D(32, (1, 1), strides=(1, 1), padding='valid',
                   activation='relu', kernel_initializer='he_normal')(result)
    path_3 = Conv2D(64, (1, 7), strides=(1, 1), padding='same',
                   activation='relu', kernel_initializer='he_normal')(path_3)
    path_3 = Conv2D(80, (1, 7), strides=(1, 1), padding='same',
                   activation='relu', kernel_initializer='he_normal')(path_3)
    avg_pooling = AveragePooling2D(pool_size=(1, 3), strides=(1,1), padding='same')(result)
    avg_pooling = Conv2D(64, (1, 1), strides=(1, 1), padding='same',
                   activation='relu', kernel_initializer='he_normal')(avg_pooling)
    result = concatenate([path_1, path_2, path_3, avg_pooling])
    result = Conv2D(10, (1,1), strides=(1,1), padding='valid',
                   activation='relu', kernel_initializer='he_normal')(result)
    result = AveragePooling2D(pool_size=(1, 14))(result)
    """
    TraceResult = Reshape((4, -1))(result)
    input_labels = Input(shape=(3,), dtype='float32')
    x = concatenate([TraceResult, input_metadata])

    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(100, kernel_initializer='he_normal', activation='relu')(x)
    x = Dense(50, kernel_initializer='he_normal', activation='relu')(x)

    # x = Flatten()(x)

    # x = concatenate([input_labels, x])
    Output = Dense(3)(x)

    model = Model(inputs=[input_traces, input_metadata], outputs=Output)

    A = Adam(lr=0.001)
    model.compile(optimizer=A, loss='mse', metrics=[metric_degrees_difference])

    return model


model = baseModelComplicated()
model.summary()