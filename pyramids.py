from operator import mul
from scipy import signal
import functools
import numpy as np
import warnings


from scipy.signal import convolve



def steer_to_harmonics_mtx(harmonics, angles=None, even_phase=True):
    '''Compute a steering matrix

    This maps a directional basis set onto the angular Fourier harmonics.

    Parameters
    ----------
    harmonics: `array_like`
        array specifying the angular harmonics contained in the steerable basis/filters.
    angles: `array_like` or None
        vector specifying the angular position of each filter (in radians). If None, defaults to
        `pi * np.arange(numh) / numh`, where `numh = harmonics.size + np.count_nonzero(harmonics)`
    even_phase : `bool`
        specifies whether the harmonics are cosine or sine phase aligned about those positions.

    Returns
    -------
    imtx : `np.array`
        This matrix is suitable for passing to the function `steer`.

    '''
    # default parameter
    numh = harmonics.size + np.count_nonzero(harmonics)
    if angles is None:
        angles = np.pi * np.arange(numh) / numh

    # Compute inverse matrix, which maps to Fourier components onto
    # steerable basis
    imtx = np.zeros((angles.size, numh))
    col = 0
    for h in harmonics:
        args = h * angles
        if h == 0:
            imtx[:, col] = np.ones(angles.shape)
            col += 1
        elif even_phase:
            imtx[:, col] = np.cos(args)
            imtx[:, col+1] = np.sin(args)
            col += 2
        else:  # odd phase
            imtx[:, col] = np.sin(args)
            imtx[:, col+1] = -1.0 * np.cos(args)
            col += 2

    r = np.linalg.matrix_rank(imtx)
    if r < np.min(imtx.shape):
        warnings.warn("Matrix is not full rank")

    return np.linalg.pinv(imtx)


def steer(basis, angle, harmonics=None, steermtx=None, return_weights=False, even_phase=True):
    '''Steer BASIS to the specfied ANGLE.

    Parameters
    ----------
    basis : `array_like`
        array whose columns are vectorized rotated copies of a steerable function, or the responses
        of a set of steerable filters.
    angle : `array_like` or `int`
        scalar or column vector the size of the basis. specifies the angle(s) (in radians) to
        steer to
    harmonics : `list` or None
        a list of harmonic numbers indicating the angular harmonic content of the basis. if None
        (default), N even or odd low frequencies, as for derivative filters
    steermtx : `array_like` or None.
        matrix which maps the filters onto Fourier series components (ordered [cos0 cos1 sin1 cos2
        sin2 ... sinN]). See steer_to_harmonics_mtx function for more details. If None (default),
        assumes cosine phase harmonic components, and filter positions at 2pi*n/N.
    return_weights : `bool`
        whether to return the weights or not.
    even_phase : `bool`
        specifies whether the harmonics are cosine or sine phase aligned about those positions.

    Returns
    -------
    res : `np.array`
        the resteered basis
    steervect : `np.array`
        the weights used to resteer the basis. only returned if `return_weights` is True
    '''

    num = basis.shape[1]

    if isinstance(angle, (int, float)):
        angle = np.array([angle])
    else:
        if angle.shape[0] != basis.shape[0] or angle.shape[1] != 1:
            raise Exception("""ANGLE must be a scalar, or a column vector
                                    the size of the basis elements""")

    # If HARMONICS is not specified, assume derivatives.
    if harmonics is None:
        harmonics = np.arange(1 - (num % 2), num, 2)

    if len(harmonics.shape) == 1 or harmonics.shape[0] == 1:
        # reshape to column matrix
        harmonics = harmonics.reshape(harmonics.shape[0], 1)
    elif harmonics.shape[0] != 1 and harmonics.shape[1] != 1:
        raise Exception('input parameter HARMONICS must be 1D!')

    if 2 * harmonics.shape[0] - (harmonics == 0).sum() != num:
        raise Exception('harmonics list is incompatible with basis size!')

    # If STEERMTX not passed, assume evenly distributed cosine-phase filters:
    if steermtx is None:
        steermtx = steer_to_harmonics_mtx(harmonics, np.pi * np.arange(num) / num,
                                          even_phase=even_phase)

    steervect = np.zeros((angle.shape[0], num))
    arg = angle * harmonics[np.nonzero(harmonics)[0]].T
    if all(harmonics):
        steervect[:, range(0, num, 2)] = np.cos(arg)
        steervect[:, range(1, num, 2)] = np.sin(arg)
    else:
        steervect[:, 0] = np.ones((arg.shape[0], 1))
        steervect[:, range(1, num, 2)] = np.cos(arg)
        steervect[:, range(2, num, 2)] = np.sin(arg)

    steervect = np.dot(steervect, steermtx)

    if steervect.shape[0] > 1:
        tmp = np.dot(basis, steervect)
        res = sum(tmp).T
    else:
        res = np.dot(basis, steervect.T)

    if return_weights:
        return res, np.array(steervect).reshape(num)
    else:
        return res

def convert_pyr_coeffs_to_pyr(pyr_coeffs):
    """this function takes a 'new pyramid' and returns the coefficients as a list

    this is to enable backwards compatibility

    Parameters
    ----------
    pyr_coeffs : `dict`
        The `pyr_coeffs` attribute of a `pyramid`.

    Returns
    -------
    coeffs : `list`
        list of `np.array`, which contains the pyramid coefficients in each band, in order from
        bottom of the pyramid to top (going through the orientations in order)
    highpass : `np.array` or None
        either the residual highpass from the pyramid or, if that doesn't exist, None
    lowpass : `np.array` or None
        either the residual lowpass from the pyramid or, if that doesn't exist, None

    """
    highpass = pyr_coeffs.pop('residual_highpass', None)
    lowpass = pyr_coeffs.pop('residual_lowpass', None)
    coeffs = [i[1] for i in sorted(pyr_coeffs.items(), key=lambda x: x[0])]
    return coeffs, highpass, lowpass


def max_pyr_height(imsz, filtsz):
    '''Compute maximum pyramid height for given image and filter sizes.

    Specifically, this computes the number of corrDn operations that can be sequentially performed
    when subsampling by a factor of 2.

    Parameters
    ----------
    imsz : `tuple` or `int`
        the size of the image (should be 2-tuple if image is 2d, `int` if it's 1d)
    filtsz : `tuple` or `int`
        the size of the filter (should be 2-tuple if image is 2d, `int` if it's 1d)

    Returns
    -------
    max_pyr_height : `int`
        The maximum height of the pyramid
    '''
    # check if inputs are one of int, tuple and have consistent type
    assert (isinstance(imsz, int) and isinstance(filtsz, int)) or (
            isinstance(imsz, tuple) and isinstance(filtsz, tuple))
    # 1D image case: reduce to the integer case
    if isinstance(imsz, tuple) and (len(imsz) == 1 or 1 in imsz):
        imsz = functools.reduce(mul, imsz)
        filtsz = functools.reduce(mul, filtsz)
    # integer case
    if isinstance(imsz, int):
        if imsz < filtsz:
            return 0
        else:
            return 1 + max_pyr_height(imsz // 2, filtsz)
    # 2D image case
    if isinstance(imsz, tuple):
        if min(imsz) < max(filtsz):
            return 0
        else:
            return 1 + max_pyr_height((imsz[0] // 2, imsz[1] // 2), filtsz)


def parse_filter(filt, normalize=True):
    """Parse the name or array like, and return a column shaped filter (which is normalized by default)

    Used during pyramid construction.

    Parameters
    ----------
    filt : `str` or `array_like`.
        Name of the filter, as accepted by `named_filter`, or array to use as a filter. See that function for acceptable names.

    Returns
    -------
    filt : `array` or `dict`
        If `filt` was one of the steerable pyramids, then this will be a dictionary
        containing the various steerable pyramid filters. Else, it will be an array containing
        the specified filter.

    See also
    --------
    named_filter : function that converts `filter_name` str into an array or dict of arrays.
    """

    if isinstance(filt, str):
        filt = named_filter(filt)

    elif isinstance(filt, np.ndarray) or isinstance(filt, list) or isinstance(filt, tuple):
        filt = np.array(filt)
        if filt.ndim == 1:
            filt = np.reshape(filt, (filt.shape[0], 1))
        elif filt.ndim == 2 and filt.shape[0] == 1:
            filt = np.reshape(filt, (-1, 1))

    # TODO expand normalization options
    if normalize:
        filt = filt / filt.sum()

    return filt

def binomial_filter(order_plus_one):
    '''returns a vector of binomial coefficients of order (order_plus_one-1)'''
    if order_plus_one < 2:
        raise Exception("Error: order_plus_one argument must be at least 2")

    kernel = np.array([[0.5], [0.5]])
    for i in range(order_plus_one - 2):
        kernel = convolve(np.array([[0.5], [0.5]]), kernel)
    return kernel


def named_filter(name):
    '''Some standard 1D filter kernels.

    These are returned as column vectors (shape [N, 1]) and scaled such that their L2-norm is 1.0 (except for 'binomN')

    * `'binomN'` - binomial coefficient filter of order N-1
    * `'haar'` - Haar wavelet
    * `'qmf8'`, `'qmf12'`, `'qmf16'` - Symmetric Quadrature Mirror Filters [1]_
    * `'daub2'`, `'daub3'`, `'daub4'` - Daubechies wavelet [2]_
    * `'qmf5'`, `'qmf9'`, `'qmf13'`   - Symmetric Quadrature Mirror Filters [3]_, [4]_
    * `'spN_filters'` - steerable pyramid filters of order N (N must be one of {0, 1, 3, 5}) [5]_,
                        [6]_

    References
    ----------
    .. [1] J D Johnston, "A filter family designed for use in quadrature mirror filter banks",
       Proc. ICASSP, pp 291-294, 1980.
    .. [2] I Daubechies, "Orthonormal bases of compactly supported wavelets", Commun. Pure Appl.
       Math, vol. 42, pp 909-996, 1988.
    .. [3] E P Simoncelli,  "Orthogonal sub-band image transforms", PhD Thesis, MIT Dept. of Elec.
       Eng. and Comp. Sci. May 1988. Also available as: MIT Media Laboratory Vision and Modeling
       Technical Report #100.
    .. [4] E P Simoncelli and E H Adelson, "Subband image coding", Subband Transforms, chapter 4,
       ed. John W Woods, Kluwer Academic Publishers,  Norwell, MA, 1990, pp 143--192.
    .. [5] E P Simoncelli and W T Freeman, "The Steerable Pyramid: A Flexible Architecture for
       Multi-Scale Derivative Computation," Second Int'l Conf on Image Processing, Washington, DC,
       Oct 1995.
    .. [6] A Karasaridis and E P Simoncelli, "A Filter Design Technique for Steerable Pyramid
       Image Transforms", ICASSP, Atlanta, GA, May 1996.

    '''

    if name.startswith("binom"):
        kernel = np.sqrt(2) * binomial_filter(int(name[5:]))

    elif name.startswith('sp'):
        kernel = steerable_filters(name)

    elif name is "qmf5":
        kernel = np.array([[-0.076103], [0.3535534], [0.8593118], [0.3535534], [-0.076103]])
    elif name is "qmf9":
        kernel = np.array([[0.02807382], [-0.060944743], [-0.073386624], [0.41472545], [0.7973934],
                           [0.41472545], [-0.073386624], [-0.060944743], [0.02807382]])
    elif name is "qmf13":
        kernel = np.array([[-0.014556438], [0.021651438], [0.039045125], [-0.09800052],
                           [-0.057827797], [0.42995453], [0.7737113], [0.42995453], [-0.057827797],
                           [-0.09800052], [0.039045125], [0.021651438], [-0.014556438]])
    elif name is "qmf8":
        kernel = np.sqrt(2) * np.array([[0.00938715], [-0.07065183], [0.06942827], [0.4899808],
                                        [0.4899808], [0.06942827], [-0.07065183], [0.00938715]])
    elif name is "qmf12":
        kernel = np.array([[-0.003809699], [0.01885659], [-0.002710326], [-0.08469594],
                           [0.08846992], [0.4843894], [0.4843894], [0.08846992],
                           [-0.08469594], [-0.002710326], [0.01885659], [-0.003809699]])
        kernel *= np.sqrt(2)
    elif name is "qmf16":
        kernel = np.array([[0.001050167], [-0.005054526], [-0.002589756], [0.0276414],
                           [-0.009666376], [-0.09039223], [0.09779817], [0.4810284], [0.4810284],
                           [0.09779817], [-0.09039223], [-0.009666376], [0.0276414],
                           [-0.002589756], [-0.005054526], [0.001050167]])
        kernel *= np.sqrt(2)
    elif name is "haar":
        kernel = np.array([[1], [1]]) / np.sqrt(2)
    elif name is "daub2":
        kernel = np.array([[0.482962913145], [0.836516303738], [0.224143868042],
                           [-0.129409522551]])
    elif name is "daub3":
        kernel = np.array([[0.332670552950], [0.806891509311], [0.459877502118], [-0.135011020010],
                           [-0.085441273882], [0.035226291882]])
    elif name is "daub4":
        kernel = np.array([[0.230377813309], [0.714846570553], [0.630880767930],
                           [-0.027983769417], [-0.187034811719], [0.030841381836],
                           [0.032883011667], [-0.010597401785]])
    elif name is "gauss5":  # for backward-compatibility
        kernel = np.sqrt(2) * np.array([[0.0625], [0.25], [0.375], [0.25], [0.0625]])
    elif name is "gauss3":  # for backward-compatibility
        kernel = np.sqrt(2) * np.array([[0.25], [0.5], [0.25]])
    else:
        raise Exception("Error: Unknown filter name: %s" % (name))

    return kernel


def steerable_filters(filter_name):
    '''Steerable pyramid filters.

    Transform described in [1]_, filter kernel design described in [2]_.

    References
    ----------
    .. [1] E P Simoncelli and W T Freeman, "The Steerable Pyramid: A Flexible Architecture for
       Multi-Scale Derivative Computation," Second Int'l Conf on Image Processing, Washington, DC,
       Oct 1995.
    .. [2] A Karasaridis and E P Simoncelli, "A Filter Design Technique for Steerable Pyramid
       Image Transforms", ICASSP, Atlanta, GA, May 1996.
    '''
    if filter_name == 'sp0_filters':
        return _sp0_filters()
    elif filter_name == 'sp1_filters':
        return _sp1_filters()
    elif filter_name == 'sp3_filters':
        return _sp3_filters()
    elif filter_name == 'sp5_filters':
        return _sp5_filters()
    # elif os.path.isfile(filter_name):
    #     raise Exception("Filter files not supported yet")
    else:
        raise Exception("filter parameters value %s not supported" % (filter_name))


def _sp0_filters():
    filters = {}
    filters['harmonics'] = np.array([0])
    filters['lo0filt'] = (
        np.array([[-4.514000e-04, -1.137100e-04, -3.725800e-04, -3.743860e-03,
                   -3.725800e-04, -1.137100e-04, -4.514000e-04],
                  [-1.137100e-04, -6.119520e-03, -1.344160e-02, -7.563200e-03,
                   -1.344160e-02, -6.119520e-03, -1.137100e-04],
                  [-3.725800e-04, -1.344160e-02, 6.441488e-02, 1.524935e-01,
                   6.441488e-02, -1.344160e-02, -3.725800e-04],
                  [-3.743860e-03, -7.563200e-03, 1.524935e-01, 3.153017e-01,
                   1.524935e-01, -7.563200e-03, -3.743860e-03],
                  [-3.725800e-04, -1.344160e-02, 6.441488e-02, 1.524935e-01,
                   6.441488e-02, -1.344160e-02, -3.725800e-04],
                  [-1.137100e-04, -6.119520e-03, -1.344160e-02, -7.563200e-03,
                   -1.344160e-02, -6.119520e-03, -1.137100e-04],
                  [-4.514000e-04, -1.137100e-04, -3.725800e-04, -3.743860e-03,
                   -3.725800e-04, -1.137100e-04, -4.514000e-04]]))
    filters['lofilt'] = (
        np.array([[-2.257000e-04, -8.064400e-04, -5.686000e-05, 8.741400e-04,
                   -1.862800e-04, -1.031640e-03, -1.871920e-03, -1.031640e-03,
                   -1.862800e-04, 8.741400e-04, -5.686000e-05, -8.064400e-04,
                   -2.257000e-04],
                  [-8.064400e-04, 1.417620e-03, -1.903800e-04, -2.449060e-03,
                   -4.596420e-03, -7.006740e-03, -6.948900e-03, -7.006740e-03,
                   -4.596420e-03, -2.449060e-03, -1.903800e-04, 1.417620e-03,
                   -8.064400e-04],
                  [-5.686000e-05, -1.903800e-04, -3.059760e-03, -6.401000e-03,
                   -6.720800e-03, -5.236180e-03, -3.781600e-03, -5.236180e-03,
                   -6.720800e-03, -6.401000e-03, -3.059760e-03, -1.903800e-04,
                   -5.686000e-05],
                  [8.741400e-04, -2.449060e-03, -6.401000e-03, -5.260020e-03,
                   3.938620e-03, 1.722078e-02, 2.449600e-02, 1.722078e-02,
                   3.938620e-03, -5.260020e-03, -6.401000e-03, -2.449060e-03,
                   8.741400e-04],
                  [-1.862800e-04, -4.596420e-03, -6.720800e-03, 3.938620e-03,
                   3.220744e-02, 6.306262e-02, 7.624674e-02, 6.306262e-02,
                   3.220744e-02, 3.938620e-03, -6.720800e-03, -4.596420e-03,
                   -1.862800e-04],
                  [-1.031640e-03, -7.006740e-03, -5.236180e-03, 1.722078e-02,
                   6.306262e-02, 1.116388e-01, 1.348999e-01, 1.116388e-01,
                   6.306262e-02, 1.722078e-02, -5.236180e-03, -7.006740e-03,
                   -1.031640e-03],
                  [-1.871920e-03, -6.948900e-03, -3.781600e-03, 2.449600e-02,
                   7.624674e-02, 1.348999e-01, 1.576508e-01, 1.348999e-01,
                   7.624674e-02, 2.449600e-02, -3.781600e-03, -6.948900e-03,
                   -1.871920e-03],
                  [-1.031640e-03, -7.006740e-03, -5.236180e-03, 1.722078e-02,
                   6.306262e-02, 1.116388e-01, 1.348999e-01, 1.116388e-01,
                   6.306262e-02, 1.722078e-02, -5.236180e-03, -7.006740e-03,
                   -1.031640e-03],
                  [-1.862800e-04, -4.596420e-03, -6.720800e-03, 3.938620e-03,
                   3.220744e-02, 6.306262e-02, 7.624674e-02, 6.306262e-02,
                   3.220744e-02, 3.938620e-03, -6.720800e-03, -4.596420e-03,
                   -1.862800e-04],
                  [8.741400e-04, -2.449060e-03, -6.401000e-03, -5.260020e-03,
                   3.938620e-03, 1.722078e-02, 2.449600e-02, 1.722078e-02,
                   3.938620e-03, -5.260020e-03, -6.401000e-03, -2.449060e-03,
                   8.741400e-04],
                  [-5.686000e-05, -1.903800e-04, -3.059760e-03, -6.401000e-03,
                   -6.720800e-03, -5.236180e-03, -3.781600e-03, -5.236180e-03,
                   -6.720800e-03, -6.401000e-03, -3.059760e-03, -1.903800e-04,
                   -5.686000e-05],
                  [-8.064400e-04, 1.417620e-03, -1.903800e-04, -2.449060e-03,
                   -4.596420e-03, -7.006740e-03, -6.948900e-03, -7.006740e-03,
                   -4.596420e-03, -2.449060e-03, -1.903800e-04, 1.417620e-03,
                   -8.064400e-04],
                  [-2.257000e-04, -8.064400e-04, -5.686000e-05, 8.741400e-04,
                   -1.862800e-04, -1.031640e-03, -1.871920e-03, -1.031640e-03,
                   -1.862800e-04, 8.741400e-04, -5.686000e-05, -8.064400e-04,
                   -2.257000e-04]]))
    filters['mtx'] = np.array([1.000000])
    filters['hi0filt'] = (
        np.array([[5.997200e-04, -6.068000e-05, -3.324900e-04, -3.325600e-04,
                   -2.406600e-04, -3.325600e-04, -3.324900e-04, -6.068000e-05,
                   5.997200e-04],
                  [-6.068000e-05, 1.263100e-04, 4.927100e-04, 1.459700e-04,
                   -3.732100e-04, 1.459700e-04, 4.927100e-04, 1.263100e-04,
                   -6.068000e-05],
                  [-3.324900e-04, 4.927100e-04, -1.616650e-03, -1.437358e-02,
                   -2.420138e-02, -1.437358e-02, -1.616650e-03, 4.927100e-04,
                   -3.324900e-04],
                  [-3.325600e-04, 1.459700e-04, -1.437358e-02, -6.300923e-02,
                   -9.623594e-02, -6.300923e-02, -1.437358e-02, 1.459700e-04,
                   -3.325600e-04],
                  [-2.406600e-04, -3.732100e-04, -2.420138e-02, -9.623594e-02,
                   8.554893e-01, -9.623594e-02, -2.420138e-02, -3.732100e-04,
                   -2.406600e-04],
                  [-3.325600e-04, 1.459700e-04, -1.437358e-02, -6.300923e-02,
                   -9.623594e-02, -6.300923e-02, -1.437358e-02, 1.459700e-04,
                   -3.325600e-04],
                  [-3.324900e-04, 4.927100e-04, -1.616650e-03, -1.437358e-02,
                   -2.420138e-02, -1.437358e-02, -1.616650e-03, 4.927100e-04,
                   -3.324900e-04],
                  [-6.068000e-05, 1.263100e-04, 4.927100e-04, 1.459700e-04,
                   -3.732100e-04, 1.459700e-04, 4.927100e-04, 1.263100e-04,
                   -6.068000e-05],
                  [5.997200e-04, -6.068000e-05, -3.324900e-04, -3.325600e-04,
                   -2.406600e-04, -3.325600e-04, -3.324900e-04, -6.068000e-05,
                   5.997200e-04]]))
    filters['bfilts'] = (
        np.array([-9.066000e-05, -1.738640e-03, -4.942500e-03, -7.889390e-03,
                  -1.009473e-02, -7.889390e-03, -4.942500e-03, -1.738640e-03,
                  -9.066000e-05, -1.738640e-03, -4.625150e-03, -7.272540e-03,
                  -7.623410e-03, -9.091950e-03, -7.623410e-03, -7.272540e-03,
                  -4.625150e-03, -1.738640e-03, -4.942500e-03, -7.272540e-03,
                  -2.129540e-02, -2.435662e-02, -3.487008e-02, -2.435662e-02,
                  -2.129540e-02, -7.272540e-03, -4.942500e-03, -7.889390e-03,
                  -7.623410e-03, -2.435662e-02, -1.730466e-02, -3.158605e-02,
                  -1.730466e-02, -2.435662e-02, -7.623410e-03, -7.889390e-03,
                  -1.009473e-02, -9.091950e-03, -3.487008e-02, -3.158605e-02,
                  9.464195e-01, -3.158605e-02, -3.487008e-02, -9.091950e-03,
                  -1.009473e-02, -7.889390e-03, -7.623410e-03, -2.435662e-02,
                  -1.730466e-02, -3.158605e-02, -1.730466e-02, -2.435662e-02,
                  -7.623410e-03, -7.889390e-03, -4.942500e-03, -7.272540e-03,
                  -2.129540e-02, -2.435662e-02, -3.487008e-02, -2.435662e-02,
                  -2.129540e-02, -7.272540e-03, -4.942500e-03, -1.738640e-03,
                  -4.625150e-03, -7.272540e-03, -7.623410e-03, -9.091950e-03,
                  -7.623410e-03, -7.272540e-03, -4.625150e-03, -1.738640e-03,
                  -9.066000e-05, -1.738640e-03, -4.942500e-03, -7.889390e-03,
                  -1.009473e-02, -7.889390e-03, -4.942500e-03, -1.738640e-03,
                  -9.066000e-05]))
    filters['bfilts'] = filters['bfilts'].reshape(len(filters['bfilts']), 1)
    return filters


def _sp1_filters():
    filters = {}
    filters['harmonics'] = np.array([1])
    filters['mtx'] = np.eye(2)
    filters['lo0filt'] = (
        np.array([[-8.701000e-05, -1.354280e-03, -1.601260e-03, -5.033700e-04,
                   2.524010e-03, -5.033700e-04, -1.601260e-03, -1.354280e-03,
                   -8.701000e-05],
                  [-1.354280e-03, 2.921580e-03, 7.522720e-03, 8.224420e-03,
                   1.107620e-03, 8.224420e-03, 7.522720e-03, 2.921580e-03,
                   -1.354280e-03],
                  [-1.601260e-03, 7.522720e-03, -7.061290e-03, -3.769487e-02,
                   -3.297137e-02, -3.769487e-02, -7.061290e-03, 7.522720e-03,
                   -1.601260e-03],
                  [-5.033700e-04, 8.224420e-03, -3.769487e-02, 4.381320e-02,
                   1.811603e-01, 4.381320e-02, -3.769487e-02, 8.224420e-03,
                   -5.033700e-04],
                  [2.524010e-03, 1.107620e-03, -3.297137e-02, 1.811603e-01,
                   4.376250e-01, 1.811603e-01, -3.297137e-02, 1.107620e-03,
                   2.524010e-03],
                  [-5.033700e-04, 8.224420e-03, -3.769487e-02, 4.381320e-02,
                   1.811603e-01, 4.381320e-02, -3.769487e-02, 8.224420e-03,
                   -5.033700e-04],
                  [-1.601260e-03, 7.522720e-03, -7.061290e-03, -3.769487e-02,
                   -3.297137e-02, -3.769487e-02, -7.061290e-03, 7.522720e-03,
                   -1.601260e-03],
                  [-1.354280e-03, 2.921580e-03, 7.522720e-03, 8.224420e-03,
                   1.107620e-03, 8.224420e-03, 7.522720e-03, 2.921580e-03,
                   -1.354280e-03],
                  [-8.701000e-05, -1.354280e-03, -1.601260e-03, -5.033700e-04,
                   2.524010e-03, -5.033700e-04, -1.601260e-03, -1.354280e-03,
                   -8.701000e-05]]))
    filters['lofilt'] = (
        np.array([[-4.350000e-05, 1.207800e-04, -6.771400e-04, -1.243400e-04,
                   -8.006400e-04, -1.597040e-03, -2.516800e-04, -4.202000e-04,
                   1.262000e-03, -4.202000e-04, -2.516800e-04, -1.597040e-03,
                   -8.006400e-04, -1.243400e-04, -6.771400e-04, 1.207800e-04,
                   -4.350000e-05],
                  [1.207800e-04, 4.460600e-04, -5.814600e-04, 5.621600e-04,
                   -1.368800e-04, 2.325540e-03, 2.889860e-03, 4.287280e-03,
                   5.589400e-03, 4.287280e-03, 2.889860e-03, 2.325540e-03,
                   -1.368800e-04, 5.621600e-04, -5.814600e-04, 4.460600e-04,
                   1.207800e-04],
                  [-6.771400e-04, -5.814600e-04, 1.460780e-03, 2.160540e-03,
                   3.761360e-03, 3.080980e-03, 4.112200e-03, 2.221220e-03,
                   5.538200e-04, 2.221220e-03, 4.112200e-03, 3.080980e-03,
                   3.761360e-03, 2.160540e-03, 1.460780e-03, -5.814600e-04,
                   -6.771400e-04],
                  [-1.243400e-04, 5.621600e-04, 2.160540e-03, 3.175780e-03,
                   3.184680e-03, -1.777480e-03, -7.431700e-03, -9.056920e-03,
                   -9.637220e-03, -9.056920e-03, -7.431700e-03, -1.777480e-03,
                   3.184680e-03, 3.175780e-03, 2.160540e-03, 5.621600e-04,
                   -1.243400e-04],
                  [-8.006400e-04, -1.368800e-04, 3.761360e-03, 3.184680e-03,
                   -3.530640e-03, -1.260420e-02, -1.884744e-02, -1.750818e-02,
                   -1.648568e-02, -1.750818e-02, -1.884744e-02, -1.260420e-02,
                   -3.530640e-03, 3.184680e-03, 3.761360e-03, -1.368800e-04,
                   -8.006400e-04],
                  [-1.597040e-03, 2.325540e-03, 3.080980e-03, -1.777480e-03,
                   -1.260420e-02, -2.022938e-02, -1.109170e-02, 3.955660e-03,
                   1.438512e-02, 3.955660e-03, -1.109170e-02, -2.022938e-02,
                   -1.260420e-02, -1.777480e-03, 3.080980e-03, 2.325540e-03,
                   -1.597040e-03],
                  [-2.516800e-04, 2.889860e-03, 4.112200e-03, -7.431700e-03,
                   -1.884744e-02, -1.109170e-02, 2.190660e-02, 6.806584e-02,
                   9.058014e-02, 6.806584e-02, 2.190660e-02, -1.109170e-02,
                   -1.884744e-02, -7.431700e-03, 4.112200e-03, 2.889860e-03,
                   -2.516800e-04],
                  [-4.202000e-04, 4.287280e-03, 2.221220e-03, -9.056920e-03,
                   -1.750818e-02, 3.955660e-03, 6.806584e-02, 1.445500e-01,
                   1.773651e-01, 1.445500e-01, 6.806584e-02, 3.955660e-03,
                   -1.750818e-02, -9.056920e-03, 2.221220e-03, 4.287280e-03,
                   -4.202000e-04],
                  [1.262000e-03, 5.589400e-03, 5.538200e-04, -9.637220e-03,
                   -1.648568e-02, 1.438512e-02, 9.058014e-02, 1.773651e-01,
                   2.120374e-01, 1.773651e-01, 9.058014e-02, 1.438512e-02,
                   -1.648568e-02, -9.637220e-03, 5.538200e-04, 5.589400e-03,
                   1.262000e-03],
                  [-4.202000e-04, 4.287280e-03, 2.221220e-03, -9.056920e-03,
                   -1.750818e-02, 3.955660e-03, 6.806584e-02, 1.445500e-01,
                   1.773651e-01, 1.445500e-01, 6.806584e-02, 3.955660e-03,
                   -1.750818e-02, -9.056920e-03, 2.221220e-03, 4.287280e-03,
                   -4.202000e-04],
                  [-2.516800e-04, 2.889860e-03, 4.112200e-03, -7.431700e-03,
                   -1.884744e-02, -1.109170e-02, 2.190660e-02, 6.806584e-02,
                   9.058014e-02, 6.806584e-02, 2.190660e-02, -1.109170e-02,
                   -1.884744e-02, -7.431700e-03, 4.112200e-03, 2.889860e-03,
                   -2.516800e-04],
                  [-1.597040e-03, 2.325540e-03, 3.080980e-03, -1.777480e-03,
                   -1.260420e-02, -2.022938e-02, -1.109170e-02, 3.955660e-03,
                   1.438512e-02, 3.955660e-03, -1.109170e-02, -2.022938e-02,
                   -1.260420e-02, -1.777480e-03, 3.080980e-03, 2.325540e-03,
                   -1.597040e-03],
                  [-8.006400e-04, -1.368800e-04, 3.761360e-03, 3.184680e-03,
                   -3.530640e-03, -1.260420e-02, -1.884744e-02, -1.750818e-02,
                   -1.648568e-02, -1.750818e-02, -1.884744e-02, -1.260420e-02,
                   -3.530640e-03, 3.184680e-03, 3.761360e-03, -1.368800e-04,
                   -8.006400e-04],
                  [-1.243400e-04, 5.621600e-04, 2.160540e-03, 3.175780e-03,
                   3.184680e-03, -1.777480e-03, -7.431700e-03, -9.056920e-03,
                   -9.637220e-03, -9.056920e-03, -7.431700e-03, -1.777480e-03,
                   3.184680e-03, 3.175780e-03, 2.160540e-03, 5.621600e-04,
                   -1.243400e-04],
                  [-6.771400e-04, -5.814600e-04, 1.460780e-03, 2.160540e-03,
                   3.761360e-03, 3.080980e-03, 4.112200e-03, 2.221220e-03,
                   5.538200e-04, 2.221220e-03, 4.112200e-03, 3.080980e-03,
                   3.761360e-03, 2.160540e-03, 1.460780e-03, -5.814600e-04,
                   -6.771400e-04],
                  [1.207800e-04, 4.460600e-04, -5.814600e-04, 5.621600e-04,
                   -1.368800e-04, 2.325540e-03, 2.889860e-03, 4.287280e-03,
                   5.589400e-03, 4.287280e-03, 2.889860e-03, 2.325540e-03,
                   -1.368800e-04, 5.621600e-04, -5.814600e-04, 4.460600e-04,
                   1.207800e-04],
                  [-4.350000e-05, 1.207800e-04, -6.771400e-04, -1.243400e-04,
                   -8.006400e-04, -1.597040e-03, -2.516800e-04, -4.202000e-04,
                   1.262000e-03, -4.202000e-04, -2.516800e-04, -1.597040e-03,
                   -8.006400e-04, -1.243400e-04, -6.771400e-04, 1.207800e-04,
                   -4.350000e-05]]))
    filters['hi0filt'] = (
        np.array([[-9.570000e-04, -2.424100e-04, -1.424720e-03, -8.742600e-04,
                   -1.166810e-03, -8.742600e-04, -1.424720e-03, -2.424100e-04,
                   -9.570000e-04],
                  [-2.424100e-04, -4.317530e-03, 8.998600e-04, 9.156420e-03,
                   1.098012e-02, 9.156420e-03, 8.998600e-04, -4.317530e-03,
                   -2.424100e-04],
                  [-1.424720e-03, 8.998600e-04, 1.706347e-02, 1.094866e-02,
                   -5.897780e-03, 1.094866e-02, 1.706347e-02, 8.998600e-04,
                   -1.424720e-03],
                  [-8.742600e-04, 9.156420e-03, 1.094866e-02, -7.841370e-02,
                   -1.562827e-01, -7.841370e-02, 1.094866e-02, 9.156420e-03,
                   -8.742600e-04],
                  [-1.166810e-03, 1.098012e-02, -5.897780e-03, -1.562827e-01,
                   7.282593e-01, -1.562827e-01, -5.897780e-03, 1.098012e-02,
                   -1.166810e-03],
                  [-8.742600e-04, 9.156420e-03, 1.094866e-02, -7.841370e-02,
                   -1.562827e-01, -7.841370e-02, 1.094866e-02, 9.156420e-03,
                   -8.742600e-04],
                  [-1.424720e-03, 8.998600e-04, 1.706347e-02, 1.094866e-02,
                   -5.897780e-03, 1.094866e-02, 1.706347e-02, 8.998600e-04,
                   -1.424720e-03],
                  [-2.424100e-04, -4.317530e-03, 8.998600e-04, 9.156420e-03,
                   1.098012e-02, 9.156420e-03, 8.998600e-04, -4.317530e-03,
                   -2.424100e-04],
                  [-9.570000e-04, -2.424100e-04, -1.424720e-03, -8.742600e-04,
                   -1.166810e-03, -8.742600e-04, -1.424720e-03, -2.424100e-04,
                   -9.570000e-04]]))
    filters['bfilts'] = (
        np.array([[6.125880e-03, -8.052600e-03, -2.103714e-02, -1.536890e-02,
                   -1.851466e-02, -1.536890e-02, -2.103714e-02, -8.052600e-03,
                   6.125880e-03, -1.287416e-02, -9.611520e-03, 1.023569e-02,
                   6.009450e-03, 1.872620e-03, 6.009450e-03, 1.023569e-02,
                   -9.611520e-03, -1.287416e-02, -5.641530e-03, 4.168400e-03,
                   -2.382180e-02, -5.375324e-02, -2.076086e-02, -5.375324e-02,
                   -2.382180e-02, 4.168400e-03, -5.641530e-03, -8.957260e-03,
                   -1.751170e-03, -1.836909e-02, 1.265655e-01, 2.996168e-01,
                   1.265655e-01, -1.836909e-02, -1.751170e-03, -8.957260e-03,
                   0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                   0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00,
                   0.000000e+00, 8.957260e-03, 1.751170e-03, 1.836909e-02,
                   -1.265655e-01, -2.996168e-01, -1.265655e-01, 1.836909e-02,
                   1.751170e-03, 8.957260e-03, 5.641530e-03, -4.168400e-03,
                   2.382180e-02, 5.375324e-02, 2.076086e-02, 5.375324e-02,
                   2.382180e-02, -4.168400e-03, 5.641530e-03, 1.287416e-02,
                   9.611520e-03, -1.023569e-02, -6.009450e-03, -1.872620e-03,
                   -6.009450e-03, -1.023569e-02, 9.611520e-03, 1.287416e-02,
                   -6.125880e-03, 8.052600e-03, 2.103714e-02, 1.536890e-02,
                   1.851466e-02, 1.536890e-02, 2.103714e-02, 8.052600e-03,
                   -6.125880e-03],
                  [-6.125880e-03, 1.287416e-02, 5.641530e-03, 8.957260e-03,
                   0.000000e+00, -8.957260e-03, -5.641530e-03, -1.287416e-02,
                   6.125880e-03, 8.052600e-03, 9.611520e-03, -4.168400e-03,
                   1.751170e-03, 0.000000e+00, -1.751170e-03, 4.168400e-03,
                   -9.611520e-03, -8.052600e-03, 2.103714e-02, -1.023569e-02,
                   2.382180e-02, 1.836909e-02, 0.000000e+00, -1.836909e-02,
                   -2.382180e-02, 1.023569e-02, -2.103714e-02, 1.536890e-02,
                   -6.009450e-03, 5.375324e-02, -1.265655e-01, 0.000000e+00,
                   1.265655e-01, -5.375324e-02, 6.009450e-03, -1.536890e-02,
                   1.851466e-02, -1.872620e-03, 2.076086e-02, -2.996168e-01,
                   0.000000e+00, 2.996168e-01, -2.076086e-02, 1.872620e-03,
                   -1.851466e-02, 1.536890e-02, -6.009450e-03, 5.375324e-02,
                   -1.265655e-01, 0.000000e+00, 1.265655e-01, -5.375324e-02,
                   6.009450e-03, -1.536890e-02, 2.103714e-02, -1.023569e-02,
                   2.382180e-02, 1.836909e-02, 0.000000e+00, -1.836909e-02,
                   -2.382180e-02, 1.023569e-02, -2.103714e-02, 8.052600e-03,
                   9.611520e-03, -4.168400e-03, 1.751170e-03, 0.000000e+00,
                   -1.751170e-03, 4.168400e-03, -9.611520e-03, -8.052600e-03,
                   -6.125880e-03, 1.287416e-02, 5.641530e-03, 8.957260e-03,
                   0.000000e+00, -8.957260e-03, -5.641530e-03, -1.287416e-02,
                   6.125880e-03]]).T)
    filters['bfilts'] = np.negative(filters['bfilts'])
    return filters


def _sp3_filters():
    filters = {}
    filters['harmonics'] = np.array([1, 3])
    filters['mtx'] = (
        np.array([[0.5000, 0.3536, 0, -0.3536],
                  [-0.0000, 0.3536, 0.5000, 0.3536],
                  [0.5000, -0.3536, 0, 0.3536],
                  [-0.0000, 0.3536, -0.5000, 0.3536]]))
    filters['hi0filt'] = (
        np.array([[-4.0483998600E-4, -6.2596000498E-4, -3.7829999201E-5,
                   8.8387000142E-4, 1.5450799838E-3, 1.9235999789E-3,
                   2.0687500946E-3, 2.0898699295E-3, 2.0687500946E-3,
                   1.9235999789E-3, 1.5450799838E-3, 8.8387000142E-4,
                   -3.7829999201E-5, -6.2596000498E-4, -4.0483998600E-4],
                  [-6.2596000498E-4, -3.2734998967E-4, 7.7435001731E-4,
                   1.5874400269E-3, 2.1750701126E-3, 2.5626500137E-3,
                   2.2892199922E-3, 1.9755100366E-3, 2.2892199922E-3,
                   2.5626500137E-3, 2.1750701126E-3, 1.5874400269E-3,
                   7.7435001731E-4, -3.2734998967E-4, -6.2596000498E-4],
                  [-3.7829999201E-5, 7.7435001731E-4, 1.1793200392E-3,
                   1.4050999889E-3, 2.2253401112E-3, 2.1145299543E-3,
                   3.3578000148E-4, -8.3368999185E-4, 3.3578000148E-4,
                   2.1145299543E-3, 2.2253401112E-3, 1.4050999889E-3,
                   1.1793200392E-3, 7.7435001731E-4, -3.7829999201E-5],
                  [8.8387000142E-4, 1.5874400269E-3, 1.4050999889E-3,
                   1.2960999738E-3, -4.9274001503E-4, -3.1295299996E-3,
                   -4.5751798898E-3, -5.1014497876E-3, -4.5751798898E-3,
                   -3.1295299996E-3, -4.9274001503E-4, 1.2960999738E-3,
                   1.4050999889E-3, 1.5874400269E-3, 8.8387000142E-4],
                  [1.5450799838E-3, 2.1750701126E-3, 2.2253401112E-3,
                   -4.9274001503E-4, -6.3222697936E-3, -2.7556000277E-3,
                   5.3632198833E-3, 7.3032598011E-3, 5.3632198833E-3,
                   -2.7556000277E-3, -6.3222697936E-3, -4.9274001503E-4,
                   2.2253401112E-3, 2.1750701126E-3, 1.5450799838E-3],
                  [1.9235999789E-3, 2.5626500137E-3, 2.1145299543E-3,
                   -3.1295299996E-3, -2.7556000277E-3, 1.3962360099E-2,
                   7.8046298586E-3, -9.3812197447E-3, 7.8046298586E-3,
                   1.3962360099E-2, -2.7556000277E-3, -3.1295299996E-3,
                   2.1145299543E-3, 2.5626500137E-3, 1.9235999789E-3],
                  [2.0687500946E-3, 2.2892199922E-3, 3.3578000148E-4,
                   -4.5751798898E-3, 5.3632198833E-3, 7.8046298586E-3,
                   -7.9501636326E-2, -0.1554141641, -7.9501636326E-2,
                   7.8046298586E-3, 5.3632198833E-3, -4.5751798898E-3,
                   3.3578000148E-4, 2.2892199922E-3, 2.0687500946E-3],
                  [2.0898699295E-3, 1.9755100366E-3, -8.3368999185E-4,
                   -5.1014497876E-3, 7.3032598011E-3, -9.3812197447E-3,
                   -0.1554141641, 0.7303866148, -0.1554141641,
                   -9.3812197447E-3, 7.3032598011E-3, -5.1014497876E-3,
                   -8.3368999185E-4, 1.9755100366E-3, 2.0898699295E-3],
                  [2.0687500946E-3, 2.2892199922E-3, 3.3578000148E-4,
                   -4.5751798898E-3, 5.3632198833E-3, 7.8046298586E-3,
                   -7.9501636326E-2, -0.1554141641, -7.9501636326E-2,
                   7.8046298586E-3, 5.3632198833E-3, -4.5751798898E-3,
                   3.3578000148E-4, 2.2892199922E-3, 2.0687500946E-3],
                  [1.9235999789E-3, 2.5626500137E-3, 2.1145299543E-3,
                   -3.1295299996E-3, -2.7556000277E-3, 1.3962360099E-2,
                   7.8046298586E-3, -9.3812197447E-3, 7.8046298586E-3,
                   1.3962360099E-2, -2.7556000277E-3, -3.1295299996E-3,
                   2.1145299543E-3, 2.5626500137E-3, 1.9235999789E-3],
                  [1.5450799838E-3, 2.1750701126E-3, 2.2253401112E-3,
                   -4.9274001503E-4, -6.3222697936E-3, -2.7556000277E-3,
                   5.3632198833E-3, 7.3032598011E-3, 5.3632198833E-3,
                   -2.7556000277E-3, -6.3222697936E-3, -4.9274001503E-4,
                   2.2253401112E-3, 2.1750701126E-3, 1.5450799838E-3],
                  [8.8387000142E-4, 1.5874400269E-3, 1.4050999889E-3,
                   1.2960999738E-3, -4.9274001503E-4, -3.1295299996E-3,
                   -4.5751798898E-3, -5.1014497876E-3, -4.5751798898E-3,
                   -3.1295299996E-3, -4.9274001503E-4, 1.2960999738E-3,
                   1.4050999889E-3, 1.5874400269E-3, 8.8387000142E-4],
                  [-3.7829999201E-5, 7.7435001731E-4, 1.1793200392E-3,
                   1.4050999889E-3, 2.2253401112E-3, 2.1145299543E-3,
                   3.3578000148E-4, -8.3368999185E-4, 3.3578000148E-4,
                   2.1145299543E-3, 2.2253401112E-3, 1.4050999889E-3,
                   1.1793200392E-3, 7.7435001731E-4, -3.7829999201E-5],
                  [-6.2596000498E-4, -3.2734998967E-4, 7.7435001731E-4,
                   1.5874400269E-3, 2.1750701126E-3, 2.5626500137E-3,
                   2.2892199922E-3, 1.9755100366E-3, 2.2892199922E-3,
                   2.5626500137E-3, 2.1750701126E-3, 1.5874400269E-3,
                   7.7435001731E-4, -3.2734998967E-4, -6.2596000498E-4],
                  [-4.0483998600E-4, -6.2596000498E-4, -3.7829999201E-5,
                   8.8387000142E-4, 1.5450799838E-3, 1.9235999789E-3,
                   2.0687500946E-3, 2.0898699295E-3, 2.0687500946E-3,
                   1.9235999789E-3, 1.5450799838E-3, 8.8387000142E-4,
                   -3.7829999201E-5, -6.2596000498E-4, -4.0483998600E-4]]))
    filters['lo0filt'] = (
        np.array([[-8.7009997515E-5, -1.3542800443E-3, -1.6012600390E-3,
                   -5.0337001448E-4, 2.5240099058E-3, -5.0337001448E-4,
                   -1.6012600390E-3, -1.3542800443E-3, -8.7009997515E-5],
                  [-1.3542800443E-3, 2.9215801042E-3, 7.5227199122E-3,
                   8.2244202495E-3, 1.1076199589E-3, 8.2244202495E-3,
                   7.5227199122E-3, 2.9215801042E-3, -1.3542800443E-3],
                  [-1.6012600390E-3, 7.5227199122E-3, -7.0612900890E-3,
                   -3.7694871426E-2, -3.2971370965E-2, -3.7694871426E-2,
                   -7.0612900890E-3, 7.5227199122E-3, -1.6012600390E-3],
                  [-5.0337001448E-4, 8.2244202495E-3, -3.7694871426E-2,
                   4.3813198805E-2, 0.1811603010, 4.3813198805E-2,
                   -3.7694871426E-2, 8.2244202495E-3, -5.0337001448E-4],
                  [2.5240099058E-3, 1.1076199589E-3, -3.2971370965E-2,
                   0.1811603010, 0.4376249909, 0.1811603010,
                   -3.2971370965E-2, 1.1076199589E-3, 2.5240099058E-3],
                  [-5.0337001448E-4, 8.2244202495E-3, -3.7694871426E-2,
                   4.3813198805E-2, 0.1811603010, 4.3813198805E-2,
                   -3.7694871426E-2, 8.2244202495E-3, -5.0337001448E-4],
                  [-1.6012600390E-3, 7.5227199122E-3, -7.0612900890E-3,
                   -3.7694871426E-2, -3.2971370965E-2, -3.7694871426E-2,
                   -7.0612900890E-3, 7.5227199122E-3, -1.6012600390E-3],
                  [-1.3542800443E-3, 2.9215801042E-3, 7.5227199122E-3,
                   8.2244202495E-3, 1.1076199589E-3, 8.2244202495E-3,
                   7.5227199122E-3, 2.9215801042E-3, -1.3542800443E-3],
                  [-8.7009997515E-5, -1.3542800443E-3, -1.6012600390E-3,
                   -5.0337001448E-4, 2.5240099058E-3, -5.0337001448E-4,
                   -1.6012600390E-3, -1.3542800443E-3, -8.7009997515E-5]]))
    filters['lofilt'] = (
        np.array([[-4.3500000174E-5, 1.2078000145E-4, -6.7714002216E-4,
                   -1.2434000382E-4, -8.0063997302E-4, -1.5970399836E-3,
                   -2.5168000138E-4, -4.2019999819E-4, 1.2619999470E-3,
                   -4.2019999819E-4, -2.5168000138E-4, -1.5970399836E-3,
                   -8.0063997302E-4, -1.2434000382E-4, -6.7714002216E-4,
                   1.2078000145E-4, -4.3500000174E-5],
                  [1.2078000145E-4, 4.4606000301E-4, -5.8146001538E-4,
                   5.6215998484E-4, -1.3688000035E-4, 2.3255399428E-3,
                   2.8898599558E-3, 4.2872801423E-3, 5.5893999524E-3,
                   4.2872801423E-3, 2.8898599558E-3, 2.3255399428E-3,
                   -1.3688000035E-4, 5.6215998484E-4, -5.8146001538E-4,
                   4.4606000301E-4, 1.2078000145E-4],
                  [-6.7714002216E-4, -5.8146001538E-4, 1.4607800404E-3,
                   2.1605400834E-3, 3.7613599561E-3, 3.0809799209E-3,
                   4.1121998802E-3, 2.2212199401E-3, 5.5381999118E-4,
                   2.2212199401E-3, 4.1121998802E-3, 3.0809799209E-3,
                   3.7613599561E-3, 2.1605400834E-3, 1.4607800404E-3,
                   -5.8146001538E-4, -6.7714002216E-4],
                  [-1.2434000382E-4, 5.6215998484E-4, 2.1605400834E-3,
                   3.1757799443E-3, 3.1846798956E-3, -1.7774800071E-3,
                   -7.4316998944E-3, -9.0569201857E-3, -9.6372198313E-3,
                   -9.0569201857E-3, -7.4316998944E-3, -1.7774800071E-3,
                   3.1846798956E-3, 3.1757799443E-3, 2.1605400834E-3,
                   5.6215998484E-4, -1.2434000382E-4],
                  [-8.0063997302E-4, -1.3688000035E-4, 3.7613599561E-3,
                   3.1846798956E-3, -3.5306399222E-3, -1.2604200281E-2,
                   -1.8847439438E-2, -1.7508180812E-2, -1.6485679895E-2,
                   -1.7508180812E-2, -1.8847439438E-2, -1.2604200281E-2,
                   -3.5306399222E-3, 3.1846798956E-3, 3.7613599561E-3,
                   -1.3688000035E-4, -8.0063997302E-4],
                  [-1.5970399836E-3, 2.3255399428E-3, 3.0809799209E-3,
                   -1.7774800071E-3, -1.2604200281E-2, -2.0229380578E-2,
                   -1.1091699824E-2, 3.9556599222E-3, 1.4385120012E-2,
                   3.9556599222E-3, -1.1091699824E-2, -2.0229380578E-2,
                   -1.2604200281E-2, -1.7774800071E-3, 3.0809799209E-3,
                   2.3255399428E-3, -1.5970399836E-3],
                  [-2.5168000138E-4, 2.8898599558E-3, 4.1121998802E-3,
                   -7.4316998944E-3, -1.8847439438E-2, -1.1091699824E-2,
                   2.1906599402E-2, 6.8065837026E-2, 9.0580143034E-2,
                   6.8065837026E-2, 2.1906599402E-2, -1.1091699824E-2,
                   -1.8847439438E-2, -7.4316998944E-3, 4.1121998802E-3,
                   2.8898599558E-3, -2.5168000138E-4],
                  [-4.2019999819E-4, 4.2872801423E-3, 2.2212199401E-3,
                   -9.0569201857E-3, -1.7508180812E-2, 3.9556599222E-3,
                   6.8065837026E-2, 0.1445499808, 0.1773651242,
                   0.1445499808, 6.8065837026E-2, 3.9556599222E-3,
                   -1.7508180812E-2, -9.0569201857E-3, 2.2212199401E-3,
                   4.2872801423E-3, -4.2019999819E-4],
                  [1.2619999470E-3, 5.5893999524E-3, 5.5381999118E-4,
                   -9.6372198313E-3, -1.6485679895E-2, 1.4385120012E-2,
                   9.0580143034E-2, 0.1773651242, 0.2120374441,
                   0.1773651242, 9.0580143034E-2, 1.4385120012E-2,
                   -1.6485679895E-2, -9.6372198313E-3, 5.5381999118E-4,
                   5.5893999524E-3, 1.2619999470E-3],
                  [-4.2019999819E-4, 4.2872801423E-3, 2.2212199401E-3,
                   -9.0569201857E-3, -1.7508180812E-2, 3.9556599222E-3,
                   6.8065837026E-2, 0.1445499808, 0.1773651242,
                   0.1445499808, 6.8065837026E-2, 3.9556599222E-3,
                   -1.7508180812E-2, -9.0569201857E-3, 2.2212199401E-3,
                   4.2872801423E-3, -4.2019999819E-4],
                  [-2.5168000138E-4, 2.8898599558E-3, 4.1121998802E-3,
                   -7.4316998944E-3, -1.8847439438E-2, -1.1091699824E-2,
                   2.1906599402E-2, 6.8065837026E-2, 9.0580143034E-2,
                   6.8065837026E-2, 2.1906599402E-2, -1.1091699824E-2,
                   -1.8847439438E-2, -7.4316998944E-3, 4.1121998802E-3,
                   2.8898599558E-3, -2.5168000138E-4],
                  [-1.5970399836E-3, 2.3255399428E-3, 3.0809799209E-3,
                   -1.7774800071E-3, -1.2604200281E-2, -2.0229380578E-2,
                   -1.1091699824E-2, 3.9556599222E-3, 1.4385120012E-2,
                   3.9556599222E-3, -1.1091699824E-2, -2.0229380578E-2,
                   -1.2604200281E-2, -1.7774800071E-3, 3.0809799209E-3,
                   2.3255399428E-3, -1.5970399836E-3],
                  [-8.0063997302E-4, -1.3688000035E-4, 3.7613599561E-3,
                   3.1846798956E-3, -3.5306399222E-3, -1.2604200281E-2,
                   -1.8847439438E-2, -1.7508180812E-2, -1.6485679895E-2,
                   -1.7508180812E-2, -1.8847439438E-2, -1.2604200281E-2,
                   -3.5306399222E-3, 3.1846798956E-3, 3.7613599561E-3,
                   -1.3688000035E-4, -8.0063997302E-4],
                  [-1.2434000382E-4, 5.6215998484E-4, 2.1605400834E-3,
                   3.1757799443E-3, 3.1846798956E-3, -1.7774800071E-3,
                   -7.4316998944E-3, -9.0569201857E-3, -9.6372198313E-3,
                   -9.0569201857E-3, -7.4316998944E-3, -1.7774800071E-3,
                   3.1846798956E-3, 3.1757799443E-3, 2.1605400834E-3,
                   5.6215998484E-4, -1.2434000382E-4],
                  [-6.7714002216E-4, -5.8146001538E-4, 1.4607800404E-3,
                   2.1605400834E-3, 3.7613599561E-3, 3.0809799209E-3,
                   4.1121998802E-3, 2.2212199401E-3, 5.5381999118E-4,
                   2.2212199401E-3, 4.1121998802E-3, 3.0809799209E-3,
                   3.7613599561E-3, 2.1605400834E-3, 1.4607800404E-3,
                   -5.8146001538E-4, -6.7714002216E-4],
                  [1.2078000145E-4, 4.4606000301E-4, -5.8146001538E-4,
                   5.6215998484E-4, -1.3688000035E-4, 2.3255399428E-3,
                   2.8898599558E-3, 4.2872801423E-3, 5.5893999524E-3,
                   4.2872801423E-3, 2.8898599558E-3, 2.3255399428E-3,
                   -1.3688000035E-4, 5.6215998484E-4, -5.8146001538E-4,
                   4.4606000301E-4, 1.2078000145E-4],
                  [-4.3500000174E-5, 1.2078000145E-4, -6.7714002216E-4,
                   -1.2434000382E-4, -8.0063997302E-4, -1.5970399836E-3,
                   -2.5168000138E-4, -4.2019999819E-4, 1.2619999470E-3,
                   -4.2019999819E-4, -2.5168000138E-4, -1.5970399836E-3,
                   -8.0063997302E-4, -1.2434000382E-4, -6.7714002216E-4,
                   1.2078000145E-4, -4.3500000174E-5]]))
    filters['bfilts'] = (
        np.array([[-8.1125000725E-4, 4.4451598078E-3, 1.2316980399E-2,
                   1.3955879956E-2,  1.4179450460E-2, 1.3955879956E-2,
                   1.2316980399E-2, 4.4451598078E-3, -8.1125000725E-4,
                   3.9103501476E-3, 4.4565401040E-3, -5.8724298142E-3,
                   -2.8760801069E-3, 8.5267601535E-3, -2.8760801069E-3,
                   -5.8724298142E-3, 4.4565401040E-3, 3.9103501476E-3,
                   1.3462699717E-3, -3.7740699481E-3, 8.2581602037E-3,
                   3.9442278445E-2, 5.3605638444E-2, 3.9442278445E-2,
                   8.2581602037E-3, -3.7740699481E-3, 1.3462699717E-3,
                   7.4700999539E-4, -3.6522001028E-4, -2.2522680461E-2,
                   -0.1105690673, -0.1768419296, -0.1105690673,
                   -2.2522680461E-2, -3.6522001028E-4, 7.4700999539E-4,
                   0.0000000000, 0.0000000000, 0.0000000000,
                   0.0000000000, 0.0000000000, 0.0000000000,
                   0.0000000000, 0.0000000000, 0.0000000000,
                   -7.4700999539E-4, 3.6522001028E-4, 2.2522680461E-2,
                   0.1105690673, 0.1768419296, 0.1105690673,
                   2.2522680461E-2, 3.6522001028E-4, -7.4700999539E-4,
                   -1.3462699717E-3, 3.7740699481E-3, -8.2581602037E-3,
                   -3.9442278445E-2, -5.3605638444E-2, -3.9442278445E-2,
                   -8.2581602037E-3, 3.7740699481E-3, -1.3462699717E-3,
                   -3.9103501476E-3, -4.4565401040E-3, 5.8724298142E-3,
                   2.8760801069E-3, -8.5267601535E-3, 2.8760801069E-3,
                   5.8724298142E-3, -4.4565401040E-3, -3.9103501476E-3,
                   8.1125000725E-4, -4.4451598078E-3, -1.2316980399E-2,
                   -1.3955879956E-2, -1.4179450460E-2, -1.3955879956E-2,
                   -1.2316980399E-2, -4.4451598078E-3, 8.1125000725E-4],
                  [0.0000000000, -8.2846998703E-4, -5.7109999034E-5,
                   4.0110000555E-5, 4.6670897864E-3, 8.0871898681E-3,
                   1.4807609841E-2, 8.6204400286E-3, -3.1221499667E-3,
                   8.2846998703E-4, 0.0000000000, -9.7479997203E-4,
                   -6.9718998857E-3, -2.0865600090E-3, 2.3298799060E-3,
                   -4.4814897701E-3, 1.4917500317E-2, 8.6204400286E-3,
                   5.7109999034E-5, 9.7479997203E-4, 0.0000000000,
                   -1.2145539746E-2, -2.4427289143E-2, 5.0797060132E-2,
                   3.2785870135E-2, -4.4814897701E-3, 1.4807609841E-2,
                   -4.0110000555E-5, 6.9718998857E-3, 1.2145539746E-2,
                   0.0000000000, -0.1510555595, -8.2495503128E-2,
                   5.0797060132E-2, 2.3298799060E-3, 8.0871898681E-3,
                   -4.6670897864E-3, 2.0865600090E-3, 2.4427289143E-2,
                   0.1510555595, 0.0000000000, -0.1510555595,
                   -2.4427289143E-2, -2.0865600090E-3, 4.6670897864E-3,
                   -8.0871898681E-3, -2.3298799060E-3, -5.0797060132E-2,
                   8.2495503128E-2, 0.1510555595, 0.0000000000,
                   -1.2145539746E-2, -6.9718998857E-3, 4.0110000555E-5,
                   -1.4807609841E-2, 4.4814897701E-3, -3.2785870135E-2,
                   -5.0797060132E-2, 2.4427289143E-2, 1.2145539746E-2,
                   0.0000000000, -9.7479997203E-4, -5.7109999034E-5,
                   -8.6204400286E-3, -1.4917500317E-2, 4.4814897701E-3,
                   -2.3298799060E-3, 2.0865600090E-3, 6.9718998857E-3,
                   9.7479997203E-4, 0.0000000000, -8.2846998703E-4,
                   3.1221499667E-3, -8.6204400286E-3, -1.4807609841E-2,
                   -8.0871898681E-3, -4.6670897864E-3, -4.0110000555E-5,
                   5.7109999034E-5, 8.2846998703E-4, 0.0000000000],
                  [8.1125000725E-4, -3.9103501476E-3, -1.3462699717E-3,
                   -7.4700999539E-4, 0.0000000000, 7.4700999539E-4,
                   1.3462699717E-3, 3.9103501476E-3, -8.1125000725E-4,
                   -4.4451598078E-3, -4.4565401040E-3, 3.7740699481E-3,
                   3.6522001028E-4, 0.0000000000, -3.6522001028E-4,
                   -3.7740699481E-3, 4.4565401040E-3, 4.4451598078E-3,
                   -1.2316980399E-2, 5.8724298142E-3, -8.2581602037E-3,
                   2.2522680461E-2, 0.0000000000, -2.2522680461E-2,
                   8.2581602037E-3, -5.8724298142E-3, 1.2316980399E-2,
                   -1.3955879956E-2, 2.8760801069E-3, -3.9442278445E-2,
                   0.1105690673, 0.0000000000, -0.1105690673,
                   3.9442278445E-2, -2.8760801069E-3, 1.3955879956E-2,
                   -1.4179450460E-2, -8.5267601535E-3, -5.3605638444E-2,
                   0.1768419296, 0.0000000000, -0.1768419296,
                   5.3605638444E-2, 8.5267601535E-3, 1.4179450460E-2,
                   -1.3955879956E-2, 2.8760801069E-3, -3.9442278445E-2,
                   0.1105690673, 0.0000000000, -0.1105690673,
                   3.9442278445E-2, -2.8760801069E-3, 1.3955879956E-2,
                   -1.2316980399E-2, 5.8724298142E-3, -8.2581602037E-3,
                   2.2522680461E-2, 0.0000000000, -2.2522680461E-2,
                   8.2581602037E-3, -5.8724298142E-3, 1.2316980399E-2,
                   -4.4451598078E-3, -4.4565401040E-3, 3.7740699481E-3,
                   3.6522001028E-4, 0.0000000000, -3.6522001028E-4,
                   -3.7740699481E-3, 4.4565401040E-3, 4.4451598078E-3,
                   8.1125000725E-4, -3.9103501476E-3, -1.3462699717E-3,
                   -7.4700999539E-4, 0.0000000000, 7.4700999539E-4,
                   1.3462699717E-3, 3.9103501476E-3, -8.1125000725E-4],
                  [3.1221499667E-3, -8.6204400286E-3, -1.4807609841E-2,
                   -8.0871898681E-3, -4.6670897864E-3, -4.0110000555E-5,
                   5.7109999034E-5, 8.2846998703E-4, 0.0000000000,
                   -8.6204400286E-3, -1.4917500317E-2, 4.4814897701E-3,
                   -2.3298799060E-3, 2.0865600090E-3, 6.9718998857E-3,
                   9.7479997203E-4, -0.0000000000, -8.2846998703E-4,
                   -1.4807609841E-2, 4.4814897701E-3, -3.2785870135E-2,
                   -5.0797060132E-2, 2.4427289143E-2, 1.2145539746E-2,
                   0.0000000000, -9.7479997203E-4, -5.7109999034E-5,
                   -8.0871898681E-3, -2.3298799060E-3, -5.0797060132E-2,
                   8.2495503128E-2, 0.1510555595, -0.0000000000,
                   -1.2145539746E-2, -6.9718998857E-3, 4.0110000555E-5,
                   -4.6670897864E-3, 2.0865600090E-3, 2.4427289143E-2,
                   0.1510555595, 0.0000000000, -0.1510555595,
                   -2.4427289143E-2, -2.0865600090E-3, 4.6670897864E-3,
                   -4.0110000555E-5, 6.9718998857E-3, 1.2145539746E-2,
                   0.0000000000, -0.1510555595, -8.2495503128E-2,
                   5.0797060132E-2, 2.3298799060E-3, 8.0871898681E-3,
                   5.7109999034E-5, 9.7479997203E-4, -0.0000000000,
                   -1.2145539746E-2, -2.4427289143E-2, 5.0797060132E-2,
                   3.2785870135E-2, -4.4814897701E-3, 1.4807609841E-2,
                   8.2846998703E-4, -0.0000000000, -9.7479997203E-4,
                   -6.9718998857E-3, -2.0865600090E-3, 2.3298799060E-3,
                   -4.4814897701E-3, 1.4917500317E-2, 8.6204400286E-3,
                   0.0000000000, -8.2846998703E-4, -5.7109999034E-5,
                   4.0110000555E-5, 4.6670897864E-3, 8.0871898681E-3,
                   1.4807609841E-2, 8.6204400286E-3, -3.1221499667E-3]]).T)
    return filters


def _sp5_filters():
    filters = {}
    filters['harmonics'] = np.array([1, 3, 5])
    filters['mtx'] = (
        np.array([[0.3333, 0.2887, 0.1667, 0.0000, -0.1667, -0.2887],
                  [0.0000, 0.1667, 0.2887, 0.3333, 0.2887, 0.1667],
                  [0.3333, -0.0000, -0.3333, -0.0000, 0.3333, -0.0000],
                  [0.0000, 0.3333, 0.0000, -0.3333, 0.0000, 0.3333],
                  [0.3333, -0.2887, 0.1667, -0.0000, -0.1667, 0.2887],
                  [-0.0000, 0.1667, -0.2887, 0.3333, -0.2887, 0.1667]]))
    filters['hi0filt'] = (
        np.array([[-0.00033429, -0.00113093, -0.00171484,
                   -0.00133542, -0.00080639, -0.00133542,
                   -0.00171484, -0.00113093, -0.00033429],
                  [-0.00113093, -0.00350017, -0.00243812,
                   0.00631653, 0.01261227, 0.00631653,
                   -0.00243812, -0.00350017, -0.00113093],
                  [-0.00171484, -0.00243812, -0.00290081,
                   -0.00673482, -0.00981051, -0.00673482,
                   -0.00290081, -0.00243812, -0.00171484],
                  [-0.00133542, 0.00631653, -0.00673482,
                   -0.07027679, -0.11435863, -0.07027679,
                   -0.00673482, 0.00631653, -0.00133542],
                  [-0.00080639, 0.01261227, -0.00981051,
                   -0.11435863, 0.81380200, -0.11435863,
                   -0.00981051, 0.01261227, -0.00080639],
                  [-0.00133542, 0.00631653, -0.00673482,
                   -0.07027679, -0.11435863, -0.07027679,
                   -0.00673482, 0.00631653, -0.00133542],
                  [-0.00171484, -0.00243812, -0.00290081,
                   -0.00673482, -0.00981051, -0.00673482,
                   -0.00290081, -0.00243812, -0.00171484],
                  [-0.00113093, -0.00350017, -0.00243812,
                   0.00631653, 0.01261227, 0.00631653,
                   -0.00243812, -0.00350017, -0.00113093],
                  [-0.00033429, -0.00113093, -0.00171484,
                   -0.00133542, -0.00080639, -0.00133542,
                   -0.00171484, -0.00113093, -0.00033429]]))
    filters['lo0filt'] = (
        np.array([[0.00341614, -0.01551246, -0.03848215, -0.01551246,
                  0.00341614],
                 [-0.01551246, 0.05586982, 0.15925570, 0.05586982,
                  -0.01551246],
                 [-0.03848215, 0.15925570, 0.40304148, 0.15925570,
                  -0.03848215],
                 [-0.01551246, 0.05586982, 0.15925570, 0.05586982,
                  -0.01551246],
                 [0.00341614, -0.01551246, -0.03848215, -0.01551246,
                  0.00341614]]))
    filters['lofilt'] = (
        2 * np.array([[0.00085404, -0.00244917, -0.00387812, -0.00944432,
                       -0.00962054, -0.00944432, -0.00387812, -0.00244917,
                       0.00085404],
                      [-0.00244917, -0.00523281, -0.00661117, 0.00410600,
                       0.01002988, 0.00410600, -0.00661117, -0.00523281,
                       -0.00244917],
                      [-0.00387812, -0.00661117, 0.01396746, 0.03277038,
                       0.03981393, 0.03277038, 0.01396746, -0.00661117,
                       -0.00387812],
                      [-0.00944432, 0.00410600, 0.03277038, 0.06426333,
                       0.08169618, 0.06426333, 0.03277038, 0.00410600,
                       -0.00944432],
                      [-0.00962054, 0.01002988, 0.03981393, 0.08169618,
                       0.10096540, 0.08169618, 0.03981393, 0.01002988,
                       -0.00962054],
                      [-0.00944432, 0.00410600, 0.03277038, 0.06426333,
                       0.08169618, 0.06426333, 0.03277038, 0.00410600,
                       -0.00944432],
                      [-0.00387812, -0.00661117, 0.01396746, 0.03277038,
                       0.03981393, 0.03277038, 0.01396746, -0.00661117,
                       -0.00387812],
                      [-0.00244917, -0.00523281, -0.00661117, 0.00410600,
                       0.01002988, 0.00410600, -0.00661117, -0.00523281,
                       -0.00244917],
                      [0.00085404, -0.00244917, -0.00387812, -0.00944432,
                       -0.00962054, -0.00944432, -0.00387812, -0.00244917,
                       0.00085404]]))
    filters['bfilts'] = (
        np.array([[0.00277643, 0.00496194, 0.01026699, 0.01455399, 0.01026699,
                   0.00496194, 0.00277643, -0.00986904, -0.00893064,
                   0.01189859, 0.02755155, 0.01189859, -0.00893064,
                   -0.00986904, -0.01021852, -0.03075356, -0.08226445,
                   -0.11732297, -0.08226445, -0.03075356, -0.01021852,
                   0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                   0.00000000, 0.00000000, 0.01021852, 0.03075356, 0.08226445,
                   0.11732297, 0.08226445, 0.03075356, 0.01021852, 0.00986904,
                   0.00893064, -0.01189859, -0.02755155, -0.01189859,
                   0.00893064, 0.00986904, -0.00277643, -0.00496194,
                   -0.01026699, -0.01455399, -0.01026699, -0.00496194,
                   -0.00277643],
                  [-0.00343249, -0.00640815, -0.00073141, 0.01124321,
                   0.00182078, 0.00285723, 0.01166982, -0.00358461,
                   -0.01977507, -0.04084211, -0.00228219, 0.03930573,
                   0.01161195, 0.00128000, 0.01047717, 0.01486305,
                   -0.04819057, -0.12227230, -0.05394139, 0.00853965,
                   -0.00459034, 0.00790407, 0.04435647, 0.09454202,
                   -0.00000000, -0.09454202, -0.04435647, -0.00790407,
                   0.00459034, -0.00853965, 0.05394139, 0.12227230,
                   0.04819057, -0.01486305, -0.01047717, -0.00128000,
                   -0.01161195, -0.03930573, 0.00228219, 0.04084211,
                   0.01977507, 0.00358461, -0.01166982, -0.00285723,
                   -0.00182078, -0.01124321, 0.00073141, 0.00640815,
                   0.00343249],
                  [0.00343249, 0.00358461, -0.01047717, -0.00790407,
                   -0.00459034, 0.00128000, 0.01166982, 0.00640815,
                   0.01977507, -0.01486305, -0.04435647, 0.00853965,
                   0.01161195, 0.00285723, 0.00073141, 0.04084211, 0.04819057,
                   -0.09454202, -0.05394139, 0.03930573, 0.00182078,
                   -0.01124321, 0.00228219, 0.12227230, -0.00000000,
                   -0.12227230, -0.00228219, 0.01124321, -0.00182078,
                   -0.03930573, 0.05394139, 0.09454202, -0.04819057,
                   -0.04084211, -0.00073141, -0.00285723, -0.01161195,
                   -0.00853965, 0.04435647, 0.01486305, -0.01977507,
                   -0.00640815, -0.01166982, -0.00128000, 0.00459034,
                   0.00790407, 0.01047717, -0.00358461, -0.00343249],
                  [-0.00277643, 0.00986904, 0.01021852, -0.00000000,
                   -0.01021852, -0.00986904, 0.00277643, -0.00496194,
                   0.00893064, 0.03075356, -0.00000000, -0.03075356,
                   -0.00893064, 0.00496194, -0.01026699, -0.01189859,
                   0.08226445, -0.00000000, -0.08226445, 0.01189859,
                   0.01026699, -0.01455399, -0.02755155, 0.11732297,
                   -0.00000000, -0.11732297, 0.02755155, 0.01455399,
                   -0.01026699, -0.01189859, 0.08226445, -0.00000000,
                   -0.08226445, 0.01189859, 0.01026699, -0.00496194,
                   0.00893064, 0.03075356, -0.00000000, -0.03075356,
                   -0.00893064, 0.00496194, -0.00277643, 0.00986904,
                   0.01021852, -0.00000000, -0.01021852, -0.00986904,
                   0.00277643],
                  [-0.01166982, -0.00128000, 0.00459034, 0.00790407,
                   0.01047717, -0.00358461, -0.00343249, -0.00285723,
                   -0.01161195, -0.00853965, 0.04435647, 0.01486305,
                   -0.01977507, -0.00640815, -0.00182078, -0.03930573,
                   0.05394139, 0.09454202, -0.04819057, -0.04084211,
                   -0.00073141, -0.01124321, 0.00228219, 0.12227230,
                   -0.00000000, -0.12227230, -0.00228219, 0.01124321,
                   0.00073141, 0.04084211, 0.04819057, -0.09454202,
                   -0.05394139, 0.03930573, 0.00182078, 0.00640815,
                   0.01977507, -0.01486305, -0.04435647, 0.00853965,
                   0.01161195, 0.00285723, 0.00343249, 0.00358461,
                   -0.01047717, -0.00790407, -0.00459034, 0.00128000,
                   0.01166982],
                  [-0.01166982, -0.00285723, -0.00182078, -0.01124321,
                   0.00073141, 0.00640815, 0.00343249, -0.00128000,
                   -0.01161195, -0.03930573, 0.00228219, 0.04084211,
                   0.01977507, 0.00358461, 0.00459034, -0.00853965,
                   0.05394139, 0.12227230, 0.04819057, -0.01486305,
                   -0.01047717, 0.00790407, 0.04435647, 0.09454202,
                   -0.00000000, -0.09454202, -0.04435647, -0.00790407,
                   0.01047717, 0.01486305, -0.04819057, -0.12227230,
                   -0.05394139, 0.00853965, -0.00459034, -0.00358461,
                   -0.01977507, -0.04084211, -0.00228219, 0.03930573,
                   0.01161195, 0.00128000, -0.00343249, -0.00640815,
                   -0.00073141, 0.01124321, 0.00182078, 0.00285723,
                   0.01166982]]).T)
    return filters

class Pyramid:
    """Base class for multiscale pyramids

    You should not instantiate this base class, it is instead inherited by the other classes found
    in this module.

    Parameters
    ----------
    image : `array_like`
        1d or 2d image upon which to construct to the pyramid.
    edge_type : {'circular', 'reflect1', 'reflect2', 'repeat', 'zero', 'extend', 'dont-compute'}
        Specifies how to handle edges. Options are:

        * `'circular'` - circular convolution
        * `'reflect1'` - reflect about the edge pixels
        * `'reflect2'` - reflect, doubling the edge pixels
        * `'repeat'` - repeat the edge pixels
        * `'zero'` - assume values of zero outside image boundary
        * `'extend'` - reflect and invert
        * `'dont-compute'` - zero output when filter overhangs imput boundaries.

    Attributes
    ----------
    image : `array_like`
        The input image used to construct the pyramid.
    image_size : `tuple`
        The size of the input image.
    pyr_type : `str` or `None`
        Human-readable string specifying the type of pyramid. For base class, is None.
    edge_type : `str`
        Specifies how edges were handled.
    pyr_coeffs : `dict`
        Dictionary containing the coefficients of the pyramid. Keys are `(level, band)` tuples and
        values are 1d or 2d numpy arrays (same number of dimensions as the input image)
    pyr_size : `dict`
        Dictionary containing the sizes of the pyramid coefficients. Keys are `(level, band)`
        tuples and values are tuples.
    is_complex : `bool`
        Whether the coefficients are complex- or real-valued. Only `SteerablePyramidFreq` can have
        a value of True, all others must be False.
    """

    def __init__(self, image, edge_type):

        self.image = np.array(image).astype(np.float)
        if self.image.ndim == 1:
            self.image = self.image.reshape(-1, 1)
        assert self.image.ndim == 2, "Error: Input signal must be 1D or 2D."

        self.image_size = self.image.shape
        if not hasattr(self, 'pyr_type'):
            self.pyr_type = None
        self.edge_type = edge_type
        self.pyr_coeffs = {}
        self.pyr_size = {}
        self.is_complex = False


    def _set_num_scales(self, filter_name, height, extra_height=0):
        """Figure out the number of scales (height) of the pyramid

        The user should not call this directly. This is called during construction of a pyramid,
        and is based on the size of the filters (thus, should be called after instantiating the
        filters) and the input image, as well as the `extra_height` parameter (which corresponds to
        the residuals, which the Gaussian pyramid contains and others do not).

        This sets `self.num_scales` directly instead of returning something, so be careful.

        Parameters
        ----------
        filter_name : `str`
            Name of the filter in the `filters` dict that determines the height of the pyramid
        height : `'auto'` or `int`
            During construction, user can specify the number of scales (height) of the pyramid.
            The pyramid will have this number of scales unless that's greater than the maximum
            possible height.
        extra_height : `int`, optional
            The automatically calculated maximum number of scales is based on the size of the input
            image and filter size. The Gaussian pyramid also contains the final residuals and so we
            need to add one more to this number.

        Returns
        -------
        None
        """
        # the Gaussian and Laplacian pyramids can go one higher than the value returned here, so we
        # use the extra_height argument to allow for that
        max_ht = max_pyr_height(self.image.shape, self.filters[filter_name].shape) + extra_height
        if height == 'auto':
            self.num_scales = max_ht
        elif height > max_ht:
            raise Exception("Cannot build pyramid higher than %d levels." % (max_ht))
        else:
            self.num_scales = int(height)

    def _recon_levels_check(self, levels):
        """Check whether levels arg is valid for reconstruction and return valid version

        When reconstructing the input image (i.e., when calling `recon_pyr()`), the user specifies
        which levels to include. This makes sure those levels are valid and gets them in the form
        we expect for the rest of the reconstruction. If the user passes `'all'`, this constructs
        the appropriate list (based on the values of `self.pyr_coeffs`).

        Parameters
        ----------
        levels : `list`, `int`,  or {`'all'`, `'residual_highpass'`, or `'residual_lowpass'`}
            If `list` should contain some subset of integers from `0` to `self.num_scales-1`
            (inclusive) and `'residual_highpass'` and `'residual_lowpass'` (if appropriate for the
            pyramid). If `'all'`, returned value will contain all valid levels. Otherwise, must be
            one of the valid levels.

        Returns
        -------
        levels : `list`
            List containing the valid levels for reconstruction.

        """
        if isinstance(levels, str) and levels == 'all':
            levels = ['residual_highpass'] + list(range(self.num_scales)) + ['residual_lowpass']
        else:
            if not hasattr(levels, '__iter__') or isinstance(levels, str):
                # then it's a single int or string
                levels = [levels]
            levs_nums = np.array([int(i) for i in levels if isinstance(i, int) or i.isdigit()])
            assert (levs_nums >= 0).all(), "Level numbers must be non-negative."
            assert (levs_nums < self.num_scales).all(), "Level numbers must be in the range [0, %d]" % (self.num_scales-1)
            levs_tmp = list(np.sort(levs_nums))  # we want smallest first
            if 'residual_highpass' in levels:
                levs_tmp = ['residual_highpass'] + levs_tmp
            if 'residual_lowpass' in levels:
                levs_tmp = levs_tmp + ['residual_lowpass']
            levels = levs_tmp
        # not all pyramids have residual highpass / lowpass, but it's easier to construct the list
        # including them, then remove them if necessary.
        if 'residual_lowpass' not in self.pyr_coeffs.keys() and 'residual_lowpass' in levels:
            levels.pop(-1)
        if 'residual_highpass' not in self.pyr_coeffs.keys() and 'residual_highpass' in levels:
            levels.pop(0)
        return levels

    def _recon_bands_check(self, bands):
        """Check whether bands arg is valid for reconstruction and return valid version

        When reconstructing the input image (i.e., when calling `recon_pyr()`), the user specifies
        which orientations to include. This makes sure those orientations are valid and gets them
        in the form we expect for the rest of the reconstruction. If the user passes `'all'`, this
        constructs the appropriate list (based on the values of `self.pyr_coeffs`).

        Parameters
        ----------
        bands : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to `self.num_orientations-1`.
            If `'all'`, returned value will contain all valid orientations. Otherwise, must be one
            of the valid orientations.

        Returns
        -------
        bands: `list`
            List containing the valid orientations for reconstruction.
        """
        if isinstance(bands, str) and bands == "all":
            bands = np.arange(self.num_orientations)
        else:
            bands = np.array(bands, ndmin=1)
            assert (bands >= 0).all(), "Error: band numbers must be larger than 0."
            assert (bands < self.num_orientations).all(), "Error: band numbers must be in the range [0, %d]" % (self.num_orientations - 1)
        return bands

    def _recon_keys(self, levels, bands, max_orientations=None):
        """Make a list of all the relevant keys from `pyr_coeffs` to use in pyramid reconstruction

        When reconstructing the input image (i.e., when calling `recon_pyr()`), the user specifies
        some subset of the pyramid coefficients to include in the reconstruction. This function
        takes in those specifications, checks that they're valid, and returns a list of tuples
        that are keys into the `pyr_coeffs` dictionary.

        Parameters
        ----------
        levels : `list`, `int`,  or {`'all'`, `'residual_highpass'`, `'residual_lowpass'`}
            If `list` should contain some subset of integers from `0` to `self.num_scales-1`
            (inclusive) and `'residual_highpass'` and `'residual_lowpass'` (if appropriate for the
            pyramid). If `'all'`, returned value will contain all valid levels. Otherwise, must be
            one of the valid levels.
        bands : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to `self.num_orientations-1`.
            If `'all'`, returned value will contain all valid orientations. Otherwise, must be one
            of the valid orientations.
        max_orientations: `None` or `int`.
            The maximum number of orientations we allow in the reconstruction. when we determine
            which ints are allowed for bands, we ignore all those greater than max_orientations.

        Returns
        -------
        recon_keys : `list`
            List of `tuples`, all of which are keys in `pyr_coeffs`. These are the coefficients to
            include in the reconstruction of the image.

        """
        levels = self._recon_levels_check(levels)
        bands = self._recon_bands_check(bands)
        if max_orientations is not None:
            for i in bands:
                if i >= max_orientations:
                    warnings.warn(("You wanted band %d in the reconstruction but max_orientation"
                                   " is %d, so we're ignoring that band" % (i, max_orientations)))
            bands = [i for i in bands if i < max_orientations]
        recon_keys = []
        for level in levels:
            # residual highpass and lowpass
            if isinstance(level, str):
                recon_keys.append(level)
            # else we have to get each of the (specified) bands at
            # that level
            else:
                recon_keys.extend([(level, band) for band in bands])
        return recon_keys


class SteerablePyramidBase(Pyramid):
    """base class for steerable pyramid

    should not be called directly, we just use it so we can make both SteerablePyramidFreq and
    SteerablePyramidSpace inherit the steer_coeffs function

    """
    def __init__(self, image, edge_type):
        super().__init__(image=image, edge_type=edge_type)

    def steer_coeffs(self, angles, even_phase=True):
        """Steer pyramid coefficients to the specified angles

        This allows you to have filters that have the Gaussian derivative order specified in
        construction, but arbitrary angles or number of orientations.

        Parameters
        ----------
        angles : `list`
            list of angles (in radians) to steer the pyramid coefficients to
        even_phase : `bool`
            specifies whether the harmonics are cosine or sine phase aligned about those positions.

        Returns
        -------
        resteered_coeffs : `dict`
            dictionary of re-steered pyramid coefficients. will have the same number of scales as
            the original pyramid (though it will not contain the residual highpass or lowpass).
            like `self.pyr_coeffs`, keys are 2-tuples of ints indexing the scale and orientation,
            but now we're indexing `angles` instead of `self.num_orientations`.
        resteering_weights : `dict`
            dictionary of weights used to re-steer the pyramid coefficients. will have the same
            keys as `resteered_coeffs`.

        """
        resteered_coeffs = {}
        resteering_weights = {}
        for i in range(self.num_scales):
            basis = np.vstack([self.pyr_coeffs[(i, j)].flatten() for j in
                               range(self.num_orientations)]).T
            for j, a in enumerate(angles):
                res, steervect = steer(basis, a, return_weights=True, even_phase=even_phase)
                resteered_coeffs[(i, j)] = res.reshape(self.pyr_coeffs[(i, 0)].shape)
                resteering_weights[(i, j)] = steervect

        return resteered_coeffs, resteering_weights

class SteerablePyramidSpace(SteerablePyramidBase):
    """Steerable pyramid (using spatial convolutions)

    Notes
    -----
    Transform described in [1]_, filter kernel design described in [2]_.

    Parameters
    ----------
    image : `array_like`
        2d image upon which to construct to the pyramid.
    height : 'auto' or `int`.
        The height of the pyramid. If 'auto', will automatically determine based on the size of
        `image`.
    order : {0, 1, 3, 5}.
        The Gaussian derivative order used for the steerable filters. If you want a different
        value, see SteerablePyramidFreq. Note that to achieve steerability the minimum number
        of orientation is `order` + 1, and is used here. To get more orientations at the same
        order, use the method `steer_coeffs`
    edge_type : {'circular', 'reflect1', 'reflect2', 'repeat', 'zero', 'extend', 'dont-compute'}
        Specifies how to handle edges. Options are:

        * `'circular'` - circular convolution
        * `'reflect1'` - reflect about the edge pixels
        * `'reflect2'` - reflect, doubling the edge pixels
        * `'repeat'` - repeat the edge pixels
        * `'zero'` - assume values of zero outside image boundary
        * `'extend'` - reflect and invert
        * `'dont-compute'` - zero output when filter overhangs imput boundaries.

    Attributes
    ----------
    image : `array_like`
        The input image used to construct the pyramid.
    image_size : `tuple`
        The size of the input image.
    pyr_type : `str` or `None`
        Human-readable string specifying the type of pyramid. For base class, is None.
    edge_type : `str`
        Specifies how edges were handled.
    pyr_coeffs : `dict`
        Dictionary containing the coefficients of the pyramid. Keys are `(level, band)` tuples and
        values are 1d or 2d numpy arrays (same number of dimensions as the input image)
    pyr_size : `dict`
        Dictionary containing the sizes of the pyramid coefficients. Keys are `(level, band)`
        tuples and values are tuples.
    is_complex : `bool`
        Whether the coefficients are complex- or real-valued. Only `SteerablePyramidFreq` can have
        a value of True, all others must be False.

    References
    ----------
    .. [1] E P Simoncelli and W T Freeman, "The Steerable Pyramid: A Flexible Architecture for
       Multi-Scale Derivative Computation," Second Int'l Conf on Image Processing, Washington, DC,
       Oct 1995.
    .. [2] A Karasaridis and E P Simoncelli, "A Filter Design Technique for Steerable Pyramid
       Image Transforms", ICASSP, Atlanta, GA, May 1996.
    """

    def __init__(self, image, height='auto', order=1, edge_type='symm'):
        super().__init__(image=image, edge_type=edge_type)

        self.order = order
        self.num_orientations = self.order + 1
        self.filters = parse_filter("sp{:d}_filters".format(self.num_orientations-1), normalize=False)
        self.pyr_type = 'SteerableSpace'
        self._set_num_scales('lofilt', height)

        hi0 = corrDn(image=self.image, filt=self.filters['hi0filt'], edge_type=self.edge_type)

        self.pyr_coeffs['residual_highpass'] = hi0
        self.pyr_size['residual_highpass'] = hi0.shape

        lo = corrDn(image=self.image, filt=self.filters['lo0filt'], edge_type=self.edge_type)
        for i in range(self.num_scales):
            # assume square filters  -- start of buildSpyrLevs
            bfiltsz = int(np.floor(np.sqrt(self.filters['bfilts'].shape[0])))

            for b in range(self.num_orientations):
                filt = self.filters['bfilts'][:, b].reshape(bfiltsz, bfiltsz).T
                band = corrDn(image=lo, filt=filt, edge_type=self.edge_type)
                self.pyr_coeffs[(i, b)] = np.array(band)
                self.pyr_size[(i, b)] = band.shape

            lo = corrDn(image=lo, filt=self.filters['lofilt'], edge_type=self.edge_type, step=(2, 2))

        self.pyr_coeffs['residual_lowpass'] = lo
        self.pyr_size['residual_lowpass'] = lo.shape

    def recon_pyr(self, order=None, edge_type=None, levels='all', bands='all'):
        """Reconstruct the image, optionally using subset of pyramid coefficients.

        Parameters
        ----------
        order : {None, 0, 1, 3, 5}.
            the Gaussian derivative order you want to use for the steerable pyramid filters used to
            reconstruct the pyramid. If None, uses the same order as that used to construct the
            pyramid.
        edge_type : {None, 'circular', 'reflect1', 'reflect2', 'repeat', 'zero', 'extend',
                     'dont-compute'}
            Specifies how to handle edges. Options are:

            * None (default) - use `self.edge_type`, the edge_type used to construct the pyramid
            * `'circular'` - circular convolution
            * `'reflect1'` - reflect about the edge pixels
            * `'reflect2'` - reflect, doubling the edge pixels
            * `'repeat'` - repeat the edge pixels
            * `'zero'` - assume values of zero outside image boundary
            * `'extend'` - reflect and inverts
            * `'dont-compute'` - zero output when filter overhangs imput boundaries.
        levels : `list`, `int`,  or {`'all'`, `'residual_highpass'`}
            If `list` should contain some subset of integers from `0` to `self.num_scales-1`
            (inclusive) and `'residual_lowpass'`. If `'all'`, returned value will contain all
            valid levels. Otherwise, must be one of the valid levels.
        bands : `list`, `int`, or `'all'`.
            If list, should contain some subset of integers from `0` to `self.num_orientations-1`.
            If `'all'`, returned value will contain all valid orientations. Otherwise, must be one
            of the valid orientations.

        Returns
        -------
        recon : `np.array`
            The reconstructed image.
        """

        if order is None:
            filters = self.filters
            recon_keys = self._recon_keys(levels, bands)
        else:
            filters = parse_filter("sp{:d}_filters".format(order), normalize=False)
            recon_keys = self._recon_keys(levels, bands, order+1)

        # assume square filters  -- start of buildSpyrLevs
        bfiltsz = int(np.floor(np.sqrt(filters['bfilts'].shape[0])))

        if edge_type is None:
            edges = self.edge_type
        else:
            edges = edge_type


        # initialize reconstruction
        if 'residual_lowpass' in recon_keys:
            recon = self.pyr_coeffs['residual_lowpass']
        else:
            recon = np.zeros_like(self.pyr_coeffs['residual_lowpass'])

        for lev in reversed(range(self.num_scales)):
            # we need to upConv once per level, in order to up-sample
            # the image back to the right shape.
            recon = upConv(image=recon, filt=filters['lofilt'], edge_type=edges,
                           step=(2, 2), start=(0, 0), stop=self.pyr_size[(lev, 0)])
            # I think the most effective way to do this is to just
            # check every possible sub-band and then only add in the
            # ones we want (given that we have to loop through the
            # levels above in order to up-sample)
            for band in reversed(range(self.num_orientations)):
                if (lev, band) in recon_keys:
                    filt = filters['bfilts'][:, band].reshape(bfiltsz, bfiltsz, order='F')
                    recon += upConv(image=self.pyr_coeffs[(lev, band)], filt=filt, edge_type=edges,
                                    stop=self.pyr_size[(lev, band)])

        # apply lo0filt
        recon = upConv(image=recon, filt=filters['lo0filt'], edge_type=edges, stop=recon.shape)

        if 'residual_highpass' in recon_keys:
            recon += upConv(image=self.pyr_coeffs['residual_highpass'], filt=filters['hi0filt'],
                            edge_type=edges, start=(0, 0), step=(1, 1), stop=recon.shape)

        return recon

def corrDn(image, filt, edge_type='symm', step=(1, 1), start=(0, 0),stop=None):
    """Correlation of image with filter.
    
        Parameters
        ----------
        image : `np.array`
            The image to be filtered.
        filt : `np.array`
            The filter to be used.
        edge_type: str {fill, wrap, symm}, optional 
        step : tuple of ints
            The step size used to sample the image.
        start : `tuple`
            2-tuple which specifies the start of the window over which we perform the convolution.


        """
    if(stop is None):
        stop = image.shape
    filt_output = signal.correlate2d(image, filt, mode='same',boundary=edge_type)
    output = filt_output[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1]]
    return output
    
def upConv(image, filt, edge_type='symm', step=(1, 1), start=(0, 0),stop=None):
    """Up-convolution of image with filter.
    Up sample by inserting 0s and then apply the filter. 
    
        Parameters
        ----------
        image : `np.array`
            The image to be filtered.
        filt : `np.array`
            The filter to be used.
        edge_type: str {fill, wrap, symm}, optional 
            fill
                pad input arrays with fillvalue.
            wrap
                circular boundary conditions.
            symm
                symmetrical boundary conditions.
        step : tuple of ints
            The step size used to upsample the image.
        start : `tuple`
            2-tuple which specifies the start of the image over which we perform the convolution.


        """
    if(stop is None):
        stop = image.shape
    output = np.zeros((int(stop[0]*step[0]), int(stop[1]*step[1])))
    output[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1]] = image
    filt_output = signal.correlate2d(output, filt, mode='same',boundary=edge_type)
    return filt_output   