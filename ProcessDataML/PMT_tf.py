from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf
from scipy import stats


class PreProcPMT(Layer):
    def __init__(self, **kwargs):
        """
        This is a tensorflow implementation of my PMT model. It it much quicker then a
        plain python implementation, especially when you have access to a GPU. Late
        pulsing is however a bit poorly implemented and slows it down considerably. If
        you turn off late pulsing it can process 200,000 events in a minute or so,
        with late pulsing it takes 27 (!) minutes. This is due to a weird way of
        working with indivual chances for every trace, but I could not reall think of a
        better way.

        Also note that it probably is a bit pointless to invest more into this late
        pulsing method, becauce it is better to just do a poisson chance for every
        photon. This is also much easier to implement in TF.

        Parameters for this function are passed here. Every parameters is described by
        a truncated normal distribution, where values that lie more than 2 sigma away
        from the mean are re taken. Parameters are given as (mean, std).

        Simplest usage:
        from keras.models import Sequential

        determine_mip = Sequential([
            PreProcPMT(input_shape=(4,80))
        ])
        traces_train = determine_mip.predict(photontimes_train, verbose=1, batch_size=2**8)

        :param kwargs: these are all passed onto the core layer of Keras
        """
        self.GR = [10., 2.] # conversion factor to mV
        self.tau = np.array([7.419741616764922, 1.5])
        self.sigma = np.array([0.8, .06])

        self.chance_1 = 0.75 # chance of a late pulse happening per trace

        self.scale_1 = np.array(
            [28.3 / 290., 10.6 / 290.])  # mean, std, as percentage of main peak size
        self.loc_1 = np.array([21, 7]) # shift relative to main peak

        self.timings = tf.constant(np.arange(1.25, 80 * 2.5 + 1.25, 2.5),
                                   dtype='float32',
                                   shape=(80,),
                                   name='timings_25_ns')

        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        """
        This is the function that gets called when you include this layer in a Keras
        model. We expect x to contain a list of photon arrivals, binned in 2.5 ns bins
        over 200 ns (so an array of 80 units with integer numbers in every one of
        them). Also, the layer assumes 4 detectors, so the input should be (-1, 4,
        80) where -1 represents the batch size, which is set by Keras. These 4
        detectors are hard-coded in, but you could just fix this (just fix the return
        statement and you are done)
        """
        x = tf.reshape(x, [-1, 80]) # flatten out the array so we are just operating on
        #  N traces
        batch_size = tf.shape(x)
        to_return = tf.zeros_like(x, dtype='float32')

        # quantum efficiency of photocathode
        x = tf.random_poisson(x * .25, [1])
        x = x[0, :, :]

        # create a loop, which applies the parametrized function to every 2.5 ns and
        # adds the result up. Loops work a bit weird in TF, so maybe look at the
        # documentation if you want to know more.
        i = tf.constant(0, dtype='int32')
        condition = lambda i, x, exp_scaled: tf.less(i, 80)
        loop_vars = [i, x, to_return]

        # pick a gain for every trace, and then tile that gain such that you get an
        # array with (N, 80), where for every N the 80 entries are the same.
        gain = tf.truncated_normal((batch_size[0],), mean=self.GR[0], stddev=self.GR[1])
        gain = tf.expand_dims(gain, 0)
        gain = tf.transpose(tf.tile(gain, (80, 1)))
        # create a parametrized pulse for one single photon, using a different tau and
        # sigma for every trace. This response will get scaled and moved for every photon.
        tau = tf.truncated_normal((batch_size[0],), mean=self.tau[0], stddev=self.tau[1])
        t_x_scaled = tau
        t_x_scaled = tf.expand_dims(t_x_scaled, 0)
        t_x_scaled = tf.transpose(tf.tile(t_x_scaled, (80, 1)))
        ln = tf.log(self.timings / t_x_scaled)
        sigma = tf.truncated_normal((batch_size[0],), mean=self.sigma[0],
                                    stddev=self.sigma[1])
        sigma = tf.expand_dims(sigma, 0)
        sigma = tf.transpose(tf.tile(sigma, (80, 1)))
        exp_scaled = tf.exp(-0.5 * (ln / sigma) ** 2)

        # this gives a total constant part
        total_constant_part = gain * exp_scaled

        # this is the main loop
        def single_run_loop(i, x, to_return):
            # pick the i'th photon from every trace
            temp_x = tf.expand_dims(x[:, i], 0)
            # now repeat this over the entire 80 units, giving a scaling factor
            temp_x = tf.tile(temp_x, (80, 1))
            temp_x = tf.transpose(temp_x)
            # multiple this by the standard pulse
            result = temp_x * total_constant_part

            # pad the traces on the left with enough zeros to shift into the correct
            # position
            paddings = tf.stack((tf.zeros(2, dtype='int32'),
                                 tf.stack((i, tf.constant(0, dtype='int32')))))
            result = tf.pad(result, paddings, "CONSTANT")
            # because of padding we have too many entries on the 'right' side of the
            # trace, so keep only the first 80 units
            to_return = to_return + result[:, :80]
            i = i + 1
            return i, x, to_return

        _, _, res = tf.while_loop(condition, single_run_loop, loop_vars)

        # this function adds a late pulse to a trace, with a new tau and sigma
        def add_late_pulses(traces, chance, pos, scale, tau, sigma):
            # used to determine the position the main peak (note that since there is no
            #  noice we define the main peak as the first non-zero entry)
            def first_rise(t):
                for i, x in enumerate(t):
                    if x > 0:
                        return i
                else:
                    return None
            # actual parametrization
            def f(t, b, tau, sigma):
                res = np.zeros(t.shape)
                t_calc = t[t > b]
                res[t > b] = np.exp(-.5 * (np.log((t_calc - b) / tau) / sigma) ** 2)
                return res
            # picks entries and repicks if not within two sigma of the mean
            def truncated_normal(mean, std):
                a, b = (mean - 2 * std - mean) / std, (mean + 2 * std - mean) / std
                x = stats.truncnorm(a, b, loc=mean, scale=std)
                return x.rvs()

            new_traces = traces
            for i, t in enumerate(traces):
                # for every trace check if a late pulse has to be added
                if np.random.random_sample() < chance:
                    # determine position of main peak
                    beginning = first_rise(t)
                    if beginning is not None:
                        # and add the late pulse
                        b_loc = truncated_normal(*pos) + beginning * 2.5
                        tau_loc = truncated_normal(*tau)
                        sigma_loc = truncated_normal(*sigma)
                        scale_loc = truncated_normal(*scale)

                        a = f(np.linspace(1.25, 80 * 2.5 + 1.25, 80), b_loc, tau_loc,
                              sigma_loc)
                        t += scale_loc * np.max(t) * a

                        new_traces[i, :] = t
            return new_traces
        # apply the late pulse function to every pulse (this is raw python function,
        # which is why it is so slow)
        res_2 = tf.py_func(add_late_pulses,
                           [res, self.chance_1, self.loc_1, self.scale_1, self.tau,
                            self.sigma], tf.float32)

        # add noise
        final_res = res_2 + tf.random_normal(tf.shape(res), 0, 1.8)
        # reshape into the same shape as the input (for now hardcoded for 4 detectors,
        # but you can of course change this)
        return tf.reshape(final_res, [-1, 4, 80])

    def compute_output_shape(self, input_shape):
        return input_shape