import numpy as np
from scipy import fft
from .utils import next_power_of_2

def convolve(data: np.ndarray, stf: np.ndarray, dt: float, tshift: float) -> np.ndarray:

    # For following FFTs
    N = data.shape[-1]

    # KEEP IT AT 2 * N the FFT is circular and we need to pad the data sufficiently
    NP2 = next_power_of_2(2 * N)

    # Fourier Transform the STF
    TRACE = fft.fft(data, axis=-1, n=NP2)
    STF = fft.fft(stf, axis=-1, n=NP2)

    # Compute correctional phase shift
    shift = -tshift
    phshift = np.exp(-1.0j * shift * np.fft.fftfreq(NP2, dt) * 2 * np.pi)

    # Return the convolution
    return np.real(fft.ifft(TRACE * STF[..., :] * phshift[..., :]))[..., :N] * dt   