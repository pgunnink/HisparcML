from keras.layers import Conv2D, Input, Reshape, concatenate, Conv1D, Flatten, \
    Dense, PReLU, LSTM, Conv3D, SeparableConv2D, ELU, SeparableConv1D, Dropout, \
    BatchNormalization, LocallyConnected2D, MaxPooling2D, GlobalAveragePooling2D, \
    GlobalMaxPooling2D
from .CustomMetrics import metric_degrees_difference
from keras.models import Model
from keras.optimizers import Adam, SGD, Adamax, Nadam, RMSprop, Adadelta
from keras.callbacks import TensorBoard,ReduceLROnPlateau,EarlyStopping
import numpy as np
from .DegRad import *
from matplotlib import pyplot as plt

def baseModel(N_stations, features, length_trace=80, trace_filter_1=64,
              trace_filter_2=32,
              nfilter=6,
              N_sep_layers=4):
    '''
    The basic model as described by https://arxiv.org/pdf/1708.00647.pdf, but flattened
    out
    '''
    input_traces = Input(shape=(N_stations * 4, length_trace), dtype='float32',
                         name='trace_input')
    reshape_traces = Reshape((N_stations * 4, length_trace, 1))(input_traces)
    input_metadata = Input(shape=(N_stations * 4, features), dtype='float32',
                           name='metadata_input')

    process_metadata = Reshape((N_stations, 4, features))(input_metadata)

    Trace = Conv2D(trace_filter_1, (1, 7), strides=(1, 4), padding='valid',
                   activation='relu', data_format='channels_last',
                   kernel_initializer='he_normal', name='first_trace_conv')(
        reshape_traces)
    Trace = Conv2D(trace_filter_2, (1, 7), strides=(1, 4), padding='valid',
                   activation='relu', kernel_initializer='he_normal',
                   name='second_trace_conv' )(Trace)
    Trace = Conv2D(10, (1, 4), strides=(1, 1), padding='valid', activation='relu',
                   kernel_initializer='he_normal', name='third_trace_conv')(Trace)
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

def baseModelDense(N_stations, features, length_trace=80, trace_filter_1=64,
              trace_filter_2=32,
              nfilter=6,
              N_sep_layers=4):
    '''
    The basic model as described by https://arxiv.org/pdf/1708.00647.pdf, but flattened
    out
    '''
    input_traces = Input(shape=(N_stations * 4, length_trace), dtype='float32',
                         name='trace_input')
    reshape_traces = Reshape((N_stations * 4, length_trace, 1))(input_traces)
    input_metadata = Input(shape=(N_stations * 4, features), dtype='float32',
                           name='metadata_input')

    process_metadata = Reshape((N_stations, 4, features))(input_metadata)

    Trace = Conv2D(trace_filter_1, (1, 7), strides=(1, 4), padding='valid',
                   activation='relu', data_format='channels_last',
                   kernel_initializer='he_normal', name='first_trace_conv')(
        reshape_traces)
    Trace = Conv2D(trace_filter_2, (1, 7), strides=(1, 4), padding='valid',
                   activation='relu', kernel_initializer='he_normal',
                   name='second_trace_conv' )(Trace)
    Trace = Conv2D(10, (1, 4), strides=(1, 1), padding='valid', activation='relu',
                   kernel_initializer='he_normal', name='third_trace_conv')(Trace)
    TraceResult = Reshape((N_stations, 4, 10))(Trace)

    x = concatenate([TraceResult, process_metadata])

    x = Flatten()(x)
    x = Dense(20*N_stations, kernel_initializer='he_normal', activation='relu')(x)
    x = Dense(10*N_stations, kernel_initializer='he_normal', activation='relu')(x)
    x = Dense(5*N_stations, kernel_initializer='he_normal', activation='relu')(x)

    Output = Dense(3)(x)

    model = Model(inputs=[input_traces, input_metadata], outputs=Output)

    A = Adam(lr=0.001)
    model.compile(optimizer=A, loss='mse', metrics=[metric_degrees_difference])

    return model

def reusedBaseModel(N_stations, features, prev_model, length_trace=80, trace_filter_1=64,
             trace_filter_2=32,
             nfilter=6,
             N_sep_layers=4):
    '''
    A basic model reusing previous weights
    '''
    input_traces = Input(shape=(N_stations * 4, length_trace), dtype='float32',
                         name='trace_input')
    reshape_traces = Reshape((N_stations * 4, length_trace, 1))(input_traces)
    input_metadata = Input(shape=(N_stations * 4, features), dtype='float32',
                           name='metadata_input')

    process_metadata = Reshape((N_stations, 4, features))(input_metadata)

    Trace_1 = Conv2D(trace_filter_1, (1, 7), strides=(1, 4), padding='valid',
                     activation='relu', data_format='channels_last',
                     weights=prev_model.layers[2].get_weights(), trainable=False)(
        reshape_traces)
    Trace_2 = Conv2D(trace_filter_2, (1, 7), strides=(1, 4), padding='valid',
                     activation='relu',
                     weights=prev_model.layers[3].get_weights(), trainable=False)(
        Trace_1)
    Trace_3 = Conv2D(10, (1, 4), strides=(1, 1), padding='valid', activation='relu',
                     weights=prev_model.layers[4].get_weights(), trainable=False)(
        Trace_2)
    TraceResult = Reshape((N_stations, 4, 10))(Trace_3)

    x = concatenate([TraceResult, process_metadata])

    x = Flatten()(x)
    x = Dense(100, kernel_initializer='he_normal', activation='relu')(x)
    x = Dense(50, kernel_initializer='he_normal', activation='relu')(x)
    x = Dense(50, kernel_initializer='he_normal', activation='relu')(x)

    Output = Dense(3)(x)

    model = Model(inputs=[input_traces, input_metadata], outputs=Output)

    A = Adam(lr=0.01)
    model.compile(optimizer=A, loss='mse', metrics=[metric_degrees_difference])

    return model

def baseModelNoTraces(N_stations, features, length_trace=80, trace_filter_1=64,
                      trace_filter_2=32,
                      nfilter=6, N_sep_layers=4):
    '''
    The basic model as described by https://arxiv.org/pdf/1708.00647.pdf, but flattened
    out
    '''
    input_traces = Input(shape=(N_stations * 4, length_trace), dtype='float32',
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


def performanceSapphire(model_prediction, sapphire_reconstruction, actual_direction):
    error_sapphire = angle_between_two_vectors(actual_direction, sapphire_reconstruction)
    error_sapphire = np.compress(~np.isnan(error_sapphire), error_sapphire)

    print('Sapphire reconstruction mean angle between  %.4f, std %.4f' % ( np.mean(
        error_sapphire), np.std(error_sapphire)))

    z_true, a_true = cartestian_to_azimuth_zenith(actual_direction[:, 0],
                                                  actual_direction[:, 1],
                                                  actual_direction[:, 2])
    rec_z_test, rec_a_test = cartestian_to_azimuth_zenith(sapphire_reconstruction[:, 0],
                                                          sapphire_reconstruction[:, 1],
                                                          sapphire_reconstruction[:, 2])

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    diff_z = np.degrees(z_true - rec_z_test[:])
    diff_z = np.compress(~np.isnan(diff_z), diff_z)
    h = plt.hist(diff_z, bins=np.degrees(np.linspace(-0.1 * np.pi, 0.1 * np.pi, 60)))
    plt.title('Difference zenith using Sapphire reconstruction')
    plt.ylabel('Occurence')
    plt.xlabel('Difference in degrees')
    plt.subplot(122)
    diff_a = np.degrees(a_true - rec_a_test[:])
    diff_a = np.compress(~np.isnan(diff_a), diff_a)
    h = plt.hist(diff_a, bins=np.degrees(np.linspace(-.2 * np.pi, .2 * np.pi, 80)))
    plt.ylabel('Occurence')
    plt.xlabel('Difference in degrees')
    plt.title('Difference azimuth using Sapphire reconstruction')


def plotGeneral(model_prediction, actual_direction, plot_zenith_bins=True):
    z_true, a_true = cartestian_to_azimuth_zenith(actual_direction[:, 0],
                                                  actual_direction[:, 1],
                                                  actual_direction[:, 2])
    z, a = cartestian_to_azimuth_zenith(model_prediction[:, 0],
                                        model_prediction[:, 1],
                                        model_prediction[:, 2])

    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    h = plt.hist(np.degrees(z_true - z),
                 bins=np.degrees(np.linspace(-0.1 * np.pi, 0.1 * np.pi, 60)))
    plt.title('Difference zenith')
    plt.subplot(122)
    h = plt.hist(np.degrees(a_true - a),
                 bins=np.degrees(np.linspace(-.2 * np.pi, .2 * np.pi, 80)))
    plt.title('Difference azimuth')

    plt.figure(figsize=(15, 10))
    plt.subplot(221)
    h = plt.hist(np.degrees(z_true), bins=np.degrees(np.linspace(0, 0.5 * np.pi, 30)))
    plt.title('actual zenith')
    plt.subplot(223)
    h = plt.hist(np.degrees(z), bins=np.degrees(np.linspace(0, 0.5 * np.pi, 30)))
    plt.title('calculated zenith')
    if plot_zenith_bins:
        available_zeniths = np.linspace(0., 60., 17, dtype=np.float32)
        for av_z in available_zeniths:
            plt.axvline(av_z, color='r')
    plt.subplot(222)
    h = plt.hist(np.degrees(a_true), bins=np.degrees(np.linspace(-np.pi, np.pi, 30)))
    plt.title('actual azimuth')
    plt.subplot(224)
    h = plt.hist(np.degrees(a), bins=np.degrees(np.linspace(-np.pi, np.pi, 30)))
    plt.title('calculated azimuth')


def performanceAngle(model_prediction, sapphire_reconstruction, actual_direction):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)

    a = angle_between_two_vectors(model_prediction, actual_direction)
    a = np.compress(~np.isnan(a), a)
    h = plt.hist(a, bins=np.degrees(np.linspace(0, 0.1 * np.pi, 50)))
    plt.title('NN with trace information')
    plt.xlabel('Angle between actual and predicted shower direction in degrees')
    plt.ylabel('Occurence')

    plt.subplot(122)
    a = angle_between_two_vectors(actual_direction, sapphire_reconstruction)
    a = np.compress(~np.isnan(a), a)
    h = plt.hist(a, bins=np.degrees(np.linspace(0, 0.1 * np.pi, 50)))
    plt.title('Sapphire reconstruction')
    plt.xlabel('Angle between actual and reconstructed direction in degrees')
    plt.ylabel('Occurence')

def errorPerAngle(model_prediction, actual_direction):
  error = angle_between_two_vectors(model_prediction,actual_direction)
  z_true, a_true = cartestian_to_azimuth_zenith(actual_direction[:, 0],
                                                  actual_direction[:, 1],
                                                  actual_direction[:, 2])
  z_box = np.linspace(0., 60., 17, dtype=np.float32)
  avg_z = np.zeros(len(z_box))
  std_z = np.zeros(len(z_box))
  i = 0
  for a in z_box:
    in_box = (z_true>np.radians(a-1))&(z_true<np.radians(a+1))
    if np.count_nonzero(in_box):
      avg_z[i] = np.mean(error[in_box])
      std_z[i] = np.std(error[in_box])
    i += 1

  a_box = np.linspace(-np.pi,np.pi, 20)
  avg_a = np.zeros(len(a_box)-1)
  std_a = np.zeros(len(a_box)-1)
  for i in range(len(std_a)):
    in_box = (a_true>a_box[i])&(a_true<a_box[i+1])
    if np.count_nonzero(in_box):
      avg_a[i] = np.mean(error[in_box])
      std_a[i] = np.std(error[in_box])

  plt.figure(figsize=(15,5))
  plt.subplot(121)
  plt.plot(z_box,avg_z,'*')
  plt.errorbar(z_box,avg_z,yerr=std_z)
  plt.title('Error as function of the zenith angle')
  plt.xlabel('Zenith angle in degrees')
  plt.ylabel('Mean error')
  plt.subplot(122)
  plt.plot(np.degrees(a_box[:-1]),avg_a,'*')
  plt.errorbar(np.degrees(a_box[:-1]),avg_a,yerr=std_a)
  plt.title('Error as function of the azimuth angle')
  plt.xlabel('Azimuth angle in degrees')
  plt.ylabel('Mean error')


def findMips(mips, highest_mip = 10):
  total_mips = np.floor(np.sum(mips,axis=1))
  idx_per_mip = []
  for i in range(highest_mip):
    idx_per_mip.append([])
  i = 0
  for m in total_mips:
    if m<highest_mip:
      idx_per_mip[int(m)].append(i)
    i += 1
  return idx_per_mip

def plotPerMips(model_prediction, actual_direction, idx_per_mip, highest_mip=10):
  mips = 0
  error_per_mip = []
  std_per_mip = []
  error_per_mip_sapphire = []
  std_per_mip_sapphire = []
  for idx in idx_per_mip:
    angle = angle_between_two_vectors(model_prediction[idx],actual_direction[idx,])
    angle = np.reshape(angle,(-1))
    angle = np.compress(~np.isnan(angle),angle)
    error_per_mip.append(np.mean(angle))
    std_per_mip.append(np.std(angle))

    mips += 1
  fig = plt.figure(figsize=(10,5))
  axes = plt.axes()
  plt.errorbar(range(highest_mip),error_per_mip, yerr=std_per_mip, label='Neural network')
  axes.legend()
  plt.title('Error as function of MiP')
  plt.xlabel('MiP')
  plt.ylabel('Mean error in degrees')


def comparePerMips(model_prediction, sapphire_prediction, actual_direction, idx_per_mip,
                   highest_mip=10):
    mips = 0
    error_per_mip = []
    std_per_mip = []
    error_per_mip_sapphire = []
    std_per_mip_sapphire = []
    for idx in idx_per_mip:
        angle = angle_between_two_vectors(model_prediction[idx], actual_direction[idx,])
        angle = np.reshape(angle, (-1))
        angle = np.compress(~np.isnan(angle), angle)
        error_per_mip.append(np.mean(angle))
        std_per_mip.append(np.std(angle))

        angle = angle_between_two_vectors(sapphire_prediction[idx],
                                          actual_direction[idx,])
        angle = np.reshape(angle, (-1))
        angle = np.compress(~np.isnan(angle), angle)
        error_per_mip_sapphire.append(np.mean(angle))
        std_per_mip_sapphire.append(np.std(angle))

        mips += 1
    fig = plt.figure(figsize=(10, 5))
    axes = plt.axes()
    plt.errorbar(range(highest_mip), error_per_mip, yerr=std_per_mip,
                 label='Neural network')
    plt.errorbar(range(highest_mip), error_per_mip_sapphire, yerr=std_per_mip_sapphire,
                 label='Sapphire')

    axes.legend()
    plt.title('Error as function of MiP')
    plt.xlabel('MiP')
    plt.ylabel('Mean error in degrees')

def compareTwoModels(model1_prediction, model2_prediction, actual_prediction):
  plt.figure(figsize=(15,5))
  plt.subplot(121)
  a = angle_between_two_vectors(model1_prediction,actual_prediction)
  a = np.compress(~np.isnan(a),a)
  h = plt.hist(a,bins=np.degrees(np.linspace(0,0.1*np.pi,50)))
  plt.title('NN with trace information')
  plt.xlabel('Angle between actual and predicted shower direction in degrees')
  plt.ylabel('Occurence')

  plt.subplot(122)
  a = angle_between_two_vectors(model2_prediction,actual_prediction)
  a = np.compress(~np.isnan(a),a)
  h = plt.hist(a,bins=np.degrees(np.linspace(0,0.1*np.pi,50)))
  plt.title('NN without trace information')
  plt.xlabel('Angle between actual and predicted shower direction in degrees')
  plt.ylabel('Occurence')