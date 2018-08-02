import numpy as np

def _simulateCathode(N_photon):
    """Simulate Cathode

    :param N_photon: number photons
    :return: number of emitted cathode electrons

    """
    N_electron = 0
    for i in range(0, int(N_photon)):
        if np.random.random() < .25:
            N_electron += 1
    return N_electron


def _simulateTrace(N, start, tau=6, stop=300.0, gain=40, sigma=.7):
    """Simulate event trace

    :param N: Number of emitted cathode electrons
    :param start: Photon arrivel time in ns
    :param t_rise: Risetime of the pulse
    :param t_fall: Falltime of the pulse
    :param stop: End of trace
    :param GR: Gain times resistance of the PMT

    :return: trace with binned simulated data according to Leo ISBN 978-3-642-57920-2
    page 190

    """
    if N == 0:
        trace = np.zeros(80)
        return trace
    trace = []
    for i in np.arange(0, start, 2.5):
        trace.append(0)
    for i in np.arange(start, stop, 2.5):
        t = np.float(i - start)
        constant = gain  # time in s instead of ns
        trace.append(-constant * (np.exp(-.5 * (np.log(t / tau) / sigma) ** 2)))
    return np.array(trace)


def _after_pulse(photontimes_hist, chance=.017 * 10, std=50, mean=150):
    new_hist = np.zeros(photontimes_hist.shape)
    for i, n in enumerate(photontimes_hist):
        for photon in range(int(n)):
            if np.random.random() < chance:
                new_position = int(i + (np.random.normal(mean, std)) / 2.5)
                if new_position < 80:
                    new_hist[new_position] = new_hist[new_position] + 1
    return new_hist


def _simulate_PMT(photontimes_hist):
    """Simulate an entire PMT from cathode to response of PMT type

    :param photontimes: an array with the arrival times of photons at the pmt

    :return: np array with the trace in V

    """

    # First check if the photontimes list is empty
    if np.sum(photontimes_hist) == 0:
        return np.array(np.zeros(80))
    # Determine how many particles arrived per 2.5 nanosecond
    n_phot, bin_edges = photontimes_hist, np.linspace(0, 200, 81)

    n_phot_afterpulse = _after_pulse(n_phot)

    t_arr = (0.5 * (bin_edges[1:] + bin_edges[:-1])) - 1.25

    # Simulate the ideal response per nanosecond and combine all single ns responses
    n_elec0 = _simulateCathode(n_phot[0])
    trace = _simulateTrace(n_elec0, t_arr[0], stop=t_arr[-1] + 2.5)
    for nphot, nphot_afterpulse, tarr in zip(n_phot[1:], n_phot_afterpulse[1:],
                                             t_arr[1:]):
        n_elec = _simulateCathode(nphot)
        n_elec_afterpulse = _simulateCathode(nphot_afterpulse)
        trace += _simulateTrace(n_elec, tarr, stop=t_arr[-1] + 2.5)
        trace += _simulateTrace(n_elec_afterpulse, tarr, stop=t_arr[-1] + 2.5) / 7
    trace = np.array(trace)

    return trace