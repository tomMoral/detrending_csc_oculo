import numpy as np
import matplotlib.pyplot as plt

from alphacsc.utils import check_random_state


NYSTAGMUS_TYPES = ["pendular", "pendular", "pendular", "pendular",
                   "jerk_sf_up", "jerk_sf_down", "jerk_fs_up", "jerk_fs_down"]

MEAN_AMPLITUDES = {
    'low_freq': 7,
    'saccad': 20,
    'nystagmus': 3
}
STD_AMPLITUDES = {
    'low_freq': 3,
    'saccad': 10,
    'nystagmus': 2
}


def format_docstring(func):
    func.__doc__ = func.__doc__.format(**globals())
    return func


def jerk(t, curv=0):
    """Generate a jerk signal with a slow phase followed by a fast phase.

    The generated signal is 1Hz and amplitude 1.

    PArameters
    ----------
    t: np.ndarray, shape (n_times, )
        Time to generate the jerk. Should be a numbers between 0 and 1.
    curv: float (default: 0)
        Curvature of the slow phase.
    """
    mask = t < 0.9
    slow_phase = curv*(t/0.9)**2 + (1-curv)*(t/0.9)-0.5
    fast_phase = 1-(t-0.9)/(1-0.9)-0.5
    return mask * slow_phase + ~mask*fast_phase


def sigmoid(t, dt_sigm):
    return 1 / (1+np.exp(-1.2 * dt_sigm * t))


def generate_amplitude(sig, random_state=None):
    rng = check_random_state(random_state)
    return max(1, MEAN_AMPLITUDES[sig] + STD_AMPLITUDES[sig] * rng.randn())


@format_docstring
def generate_signal(n_times=5000, s_freq=1000, nystagmus_type="pendular",
                    nystagmus_freq=4, curv=0, saccad_freq=.5, dt_sigm=0.1,
                    std_noise=.3, nystagmus_amp=MEAN_AMPLITUDES['nystagmus'],
                    saccad_amp=MEAN_AMPLITUDES['saccad'],
                    low_freq_amp=MEAN_AMPLITUDES['low_freq'],
                    display=False, random_state=None):
    """Generate a fac-simile of a nystagmus signal

    In this helper, all amplitude are measured as the standard deviation of the
    signal.

    Parameters
    ----------
    n_times: int (default: 5000)
        Length of the generated signal.
    s_freq: float (default: 1000)
        Sampling frequency for the signals
    nystagmus_type: str (default: 'pendular)
        Type of nystagmic pattern included in the signal. This should be one
        of {NYSTAGMUS_TYPES}.
    nystagmus_freq: float (default: 4)
        Frequency of the nystagmus signal.
    curv: float (default: 0)
        Curvature of the jerk patterns for the nystagmus.
    saccad_freq: float (default: .5)
        Frequency of the saccad signal.
    dt_sigm: float (default: .5)
        Deviation of the sigmoid used to generate the saccad signal. The
        saccad duraction will be 1 / dt_sigm
    std_noise: float (default: 3)
        Amplitude of the noise.
    nystagmus_amp: float (default: 3)
        Amplitude of the nystagmus signal.
    saccad_amp: float (default: 20)
        Amplitude of the saccad signal.
    low_freq_amp: float (default: 5)
        Amplitude of the generated low frequency signal.
    display: boolean (default: False)
        If set to True, displays the resulting signal.
    random_state: int, RandomState or None (default: None)
        random_state for the random number generator

    """

    rng = check_random_state(random_state)

    t = np.arange(n_times) / s_freq

    # Generate a low frequency signal by generating low frequency Fourier
    # coefficients.
    max_freq_idx = max(1, int(n_times / (2 * s_freq)))
    freq_signal = np.zeros(n_times // 2 + 1, dtype=np.complex)
    for i in range(max_freq_idx):
        freq_signal[i] = rng.randn() + rng.randn() * 1j
    signal = np.fft.irfft(freq_signal)
    signal -= signal.mean()

    if np.std(signal) > 1e-5:
        signal *= low_freq_amp / np.std(signal)

    # Generate saccades location, amplitude, and sign
    end_last_sacc = 0
    saccs = []
    while end_last_sacc < n_times / s_freq:
        sign = 2 * (rng.rand() > .5) - 1
        amp = 1.5 + .5 * rng.randn()
        dt = rng.exponential(saccad_freq)
        start = end_last_sacc + dt
        end_last_sacc = start + dt_sigm
        saccs.append((start, end_last_sacc, sign, amp))

    # Generate the saccade signals as sigmoids with
    saccsignal = np.zeros(n_times)
    for i, tt in enumerate(t):
        for start, end, sign, amp in saccs:
            if start < tt:
                saccsignal[i] += sign * amp * sigmoid(i-start-50, dt_sigm)

    if np.std(saccsignal) > 1e-5:
        saccsignal *= saccad_amp / np.std(saccsignal)

    # Create the gaze signal as the sum of the low-freq signal, the saccad
    # signal and a noise term.
    signal = signal + saccsignal
    signal += std_noise * rng.randn(n_times)

    # Create the nystagmus signal depending on its type .
    if nystagmus_type == "pendular":
        nyst = np.cos(2 * np.pi * t*nystagmus_freq)
    elif "jerk" in nystagmus_type:
        sign = -1 if "_down_" in nystagmus_type else 1
        nyst = sign * jerk((t * nystagmus_freq % 1), curv)
        if "_fs_" in nystagmus_type:
            nyst = nyst[::-1]
    else:
        raise NotImplementedError(f"Unknown nystagmus type {nystagmus_type}. "
                                  f"Should be one of {NYSTAGMUS_TYPES}.")
    nyst *= nystagmus_amp / np.std(nyst)

    if display:
        plt.plot(signal + nyst)
        plt.show()

    return signal, nyst


def simulate_oculo(n_trials, n_times, std_noise=.3, display=False,
                   random_state=None):
    """Simulate a dataset of n_trials oculo signal with facsimile nystagmus.

    Parameters
    ----------
    n_trials: int
        Number of signal generated
    n_times: int
        Length of the generated signals
    std_noise: float
        Standard deviation of the additive Gaussian white noise.
    display: boolean (default: False)
        If set to True, displays the resulting signal.
    random_state: int, RandomState or None (default: None)
        random_state for the random number generator

    """

    rng = check_random_state(random_state)

    # Sampling frequency
    s_freq = 1000
    # Mean saccades frequency
    saccad_freq = .5
    # Duration of the saccads in second
    dt_sigm = 0.1

    trends = np.zeros((n_trials, n_times))
    nystagmus = np.zeros((n_trials, n_times))
    for i in range(n_trials):
        print(f"Generate signals: {i/n_trials:7.2%}\r", end='', flush=True)
        # Generate the signal parameter
        # Nystagmus type
        nystagmus_type = rng.choice(NYSTAGMUS_TYPES)
        # Nystagmus Frequency
        nystagmus_freq = max(1, min(4+2*rng.randn(), 6))
        # Nystagmus amplitude
        nystagmus_amp = generate_amplitude('nystagmus', random_state=rng)
        # Amplitude of the saccads
        saccad_amp = generate_amplitude('saccad', random_state=rng)
        # Amplitude of the low freq component
        low_freq_amp = generate_amplitude('low_freq', random_state=rng)
        # Curvature of the jerk
        curv = max(0, min(0.5+0.25*rng.randn(), 1))

        # Generate the signal
        trends[i], nystagmus[i] = generate_signal(
            n_times=n_times, nystagmus_type=nystagmus_type, s_freq=s_freq,
            nystagmus_freq=nystagmus_freq, nystagmus_amp=nystagmus_amp,
            saccad_freq=saccad_freq, saccad_amp=saccad_amp,
            dt_sigm=dt_sigm, curv=curv, low_freq_amp=low_freq_amp,
            std_noise=std_noise, display=display, random_state=rng)
    print(f"Generate signals: done".ljust(40))
    return trends, nystagmus


if __name__ == '__main__':

    import os
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate a dataset of simulated oculo signals"
    )
    parser.add_argument('-n', '--n-trials', type=int, default=100,
                        help="Number of generated signals")
    parser.add_argument('-t', '--n-times', type=int, default=5000,
                        help="Number of generated signals")
    parser.add_argument('--display', action='store_true',
                        help="Display the generated signals")
    parser.add_argument('--data-dir', type=str, default='.',
                        help="Display the generated signals")
    args = parser.parse_args()
    n_trials = args.n_trials
    n_times = args.n_times

    trends, nystagmus = simulate_oculo(n_trials=n_trials, n_times=n_times,
                                       display=args.display)

    np.save(os.path.join(args.data_dir, "trends"), trends)
    np.save(os.path.join(args.data_dir, "nysts"), nystagmus)