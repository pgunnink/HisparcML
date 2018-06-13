from keras.layers import Conv2D, Input, Reshape, concatenate, Conv1D, Flatten, \
    Dense, PReLU, LSTM, Conv3D, SeparableConv2D, ELU, SeparableConv1D, Dropout, \
    BatchNormalization, LocallyConnected2D, MaxPooling2D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D
from .CustomMetrics import metric_degrees_difference
from keras.models import Model
from keras.optimizers import Adam, SGD, Adamax, Nadam, RMSprop, Adadelta
from keras.callbacks import TensorBoard,ReduceLROnPlateau,EarlyStopping

def baseModel(N_stations, features, length_trace=80, trace_filter_1=64,
              trace_filter_2=32,
              nfilter=6,
              N_sep_layers=4):
    '''
    The basic model as described by https://arxiv.org/pdf/1708.00647.pdf, but flattened
    out
    '''
    input_traces = Input(shape=(N_stations * 4, length_trace, 1), dtype='float32',
                         name='trace_input')
    input_metadata = Input(shape=(N_stations * 4, features), dtype='float32',
                           name='metadata_input')

    process_metadata = Reshape((N_stations, 4, features))(input_metadata)

    Trace = Conv2D(trace_filter_1, (1, 7), strides=(1, 4), padding='valid',
                   activation='relu', data_format='channels_last',
                   kernel_initializer='he_normal', )(input_traces)
    Trace = Conv2D(trace_filter_2, (1, 7), strides=(1, 4), padding='valid',
                   activation='relu', kernel_initializer='he_normal', )(Trace)
    Trace = Conv2D(10, (1, 4), strides=(1, 1), padding='valid', activation='relu',
                   kernel_initializer='he_normal', )(Trace)
    TraceResult = Reshape((N_stations, 4, 10))(Trace)

    x = concatenate([TraceResult, process_metadata])

    for i in range(N_sep_layers):
        c = SeparableConv2D(nfilter, (3, 3), padding='same',
                            kernel_initializer='he_normal', activation='relu',
                            depth_multiplier=1)(x)
        x = concatenate([c, x])
        nfilter = nfilter * 2

    x = Flatten()(x)
    Output = Dense(3)(x)

    model = Model(inputs=[input_traces, input_metadata], outputs=Output)

    A = Adam(lr=0.001)
    model.compile(optimizer=A, loss='mse', metrics=[metric_degrees_difference])

    return model


def baseModelNoTraces(N_stations, features, length_trace=80, trace_filter_1=64,
                      trace_filter_2=32,
                      nfilter=6, N_sep_layers=4):
    '''
    The basic model as described by https://arxiv.org/pdf/1708.00647.pdf, but flattened
    out
    '''
    input_traces = Input(shape=(N_stations * 4, length_trace, 1), dtype='float32',
                         name='trace_input')
    input_metadata = Input(shape=(N_stations * 4, features), dtype='float32',
                           name='metadata_input')

    process_metadata = Reshape((N_stations, 4, features))(input_metadata)

    x = process_metadata

    for i in range(N_sep_layers):
        c = SeparableConv2D(nfilter, (3, 3), padding='same',
                            kernel_initializer='he_normal', activation='relu',
                            depth_multiplier=1)(x)
        x = concatenate([c, x])
        nfilter = nfilter * 2

    x = Flatten()(x)
    Output = Dense(3)(x)

    model = Model(inputs=[input_traces, input_metadata], outputs=Output)

    A = Adam(lr=0.001)
    model.compile(optimizer=A, loss='mse', metrics=[metric_degrees_difference])

    return model
