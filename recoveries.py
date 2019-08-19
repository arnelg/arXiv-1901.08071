import numpy as np

import qutip as qt

import channels
import codes


"""
Misc functions
"""


def matrixf(rho, f, safe=False):
    # Apply f(rho) to diagonalizable matrix rho
    abstol = 1e-8
    vals, vecs = rho.eigenstates()
    out = qt.Qobj(np.zeros(rho.shape), dims=rho.dims)
    for i, lam in enumerate(vals):
        if lam > abstol or not safe:
            out += f(lam)*vecs[i]*vecs[i].dag()
    return out


def replace_empty_lists_with_ones(listin):
    for i, li in enumerate(listin):
        if li == []:
            listin[i] = [1]
        elif isinstance(li, list):
            li = replace_empty_lists_with_ones(li)
    return listin


class RecoveryException(Exception):
    pass


"""
Measurements
"""


def phasestate(s, r, phi0=0., fockdim=None):
    if fockdim is None:
        fockdim = s
    phim = phi0 + (2.0 * np.pi * r) / s
    n = np.arange(s)
    data = 1.0 / np.sqrt(s) * np.exp(1.0j * n * phim)
    data = np.hstack((data, np.zeros(fockdim-s)))
    return qt.Qobj(data)


class WedgeMeasurement(channels.POVM):
    def __init__(self, N, s, fockdim=None, offset=0.):
        if fockdim is None:
            fockdim = s
        self.s = s
        self.fockdim = fockdim
        povm_list = []
        for m in range(N):
            povm = self._povm(m*2*np.pi/N + offset, (m+1)*2*np.pi/N + offset)
            povm_list.append(povm)
        channels.POVM.__init__(self, povm_list)

    def _povm(self, theta1, theta2):
        def matrixelement(n, m, theta1, theta2):
            return np.where(n == m,
                            theta2-theta1,
                            1j/(n-m)*(np.exp(1j*theta1*(n-m))
                                      - np.exp(1j*theta2*(n-m))))
        povm = 1/(2*np.pi)*np.fromfunction(lambda n, m: matrixelement(n, m,
                                           theta1, theta2), (self.s, self.s))
        return qt.Qobj(povm)


class PhaseMeasurement(channels.POVM):
    def __init__(self, dim, offset=0.):
        outcomes = []
        povm_list = []
        for r in range(dim):
            outcomes.append((2*np.pi*r/dim + offset) % (2*np.pi))
            phi = qt.phase_basis(dim, r, phi0=offset)
            povm_list.append(phi*phi.dag())
        channels.POVM.__init__(self, povm_list, kraus=povm_list,
                               outcomes=outcomes)


class PrettyGoodMeasurement(channels.POVM):
    def __init__(self, code, noise):

        sigma = noise(code.projector)
        x = matrixf(sigma, lambda x: x**(-1/2), safe=True)
        povm_list = [
                     x*noise.channel_matrix(code.plus)*x,
                     x*noise.channel_matrix(code.minus)*x,
                    ]
        povm_list.append(code.identity - sum(povm_list))
        channels.POVM.__init__(self, povm_list)


"""
class PrettyGoodMeasurement(channels.POVM):
    def __init__(self, code, noise, ancilla=None):
        if ancilla is None:
            ancilla = code

        crot = qt.to_super(code.crot(ancilla))
        noise2 = crot*qt.composite(noise.channel_matrix,
                                   qt.to_super(ancilla.identity))*crot.dag()
        sigma = noise2(qt.tensor(code.projector, ancilla.projector))
        x = matrixf(sigma, lambda x: x**(-1/2), safe=True)
        povm_list = [
                     x*noise2(qt.tensor(code.plus, ancilla.plus))*x,
                     x*noise2(qt.tensor(code.plus, ancilla.minus))*x,
                     x*noise2(qt.tensor(code.minus, ancilla.plus))*x,
                     x*noise2(qt.tensor(code.minus, ancilla.minus))*x,
                    ]
        povm_list.append(qt.tensor(code.identity, ancilla.identity)
                         - np.sum(povm_list))
        channels.POVM.__init__(self, povm_list)
"""


class CafarovanLoockMeasurement(channels.POVM):
    def __init__(self, code, noise, lmax=-1, ancilla=None):
        if ancilla is None:
            ancilla = code

        crot = code.crot(ancilla)
        kraus2 = [crot*qt.tensor(k, ancilla.identity)*crot.dag()
                  for k in noise.kraus[:lmax]]
        logicals = [
                     qt.tensor(code.plus, ancilla.plus),
                     qt.tensor(code.plus, ancilla.minus),
                     qt.tensor(code.minus, ancilla.plus),
                     qt.tensor(code.minus, ancilla.minus),
                    ]

        def elem(psi, k):
            x = qt.expect(k.dag()*k, psi)
            if x > 0:
                return k*psi*psi.dag()*k.dag()/x
            else:
                return None

        povm_list = [np.sum((elem(psi, k) for k in kraus2))
                     for psi in logicals]
        povm_list.append(qt.tensor(code.identity, ancilla.identity)
                         - np.sum(povm_list))
        channels.POVM.__init__(self, povm_list)


"""
Code recovery channels
"""


class BarnumKnillRecovery(channels.Channel):

    def __init__(self, code, noise, lmax=-1):
        k_list = []
        k_sum = qt.Qobj()
        P = code.projector
        EP = matrixf(noise.channel_matrix(P), lambda x: x**(-1/2), safe=True)
        # EP = noise.channel_matrix(P)
        for e in noise.kraus[:lmax]:
            op = P*e.dag()*EP
            k_list.append(op)
            k_sum += op.dag()*op
        k_list.append((code.identity-k_sum).sqrtm())
        channels.Channel.__init__(self, k_list)


class CafarovanLoockRecovery(channels.Channel):
    """
    Eq. (51) in https://arxiv.org/abs/1308.4582
    """
    def __init__(self, code, noise, lmax=-1, warn=True):
        if warn:
            if lmax > code.N:
                print("warning: CafarovanLoockRecovery does not work well with"
                      + "lmax > code.N")
        k_list = []
        k_sum = qt.Qobj()
        for e in noise.kraus[:lmax]:
            d = qt.expect(e.dag()*e, code.zero)
            op = qt.Qobj()
            if d > 0:
                op = code.zero*code.zero.dag()*e.dag()/np.sqrt(d)
            d = qt.expect(e.dag()*e, code.one)
            if d > 0:
                op += code.one*code.one.dag()*e.dag()/np.sqrt(d)
            k_list.append(op)
            k_sum += op.dag()*op
        k_list.append((code.identity-k_sum).sqrtm())
        # k_list.append(code.identity-s)
        channels.Channel.__init__(self, k_list)
        self._k_sum = k_sum


class SDPRecovery(channels.Channel):
    """
    Find optimal recovery by solving SDP
    """

    def __init__(self, code, noise):
        import matlab.engine
        self._eng = matlab.engine.start_matlab()
        self._code = code
        self._noise = noise
        channels.Channel.__init__(self)

    def __del__(self):
        self._eng.quit()

    def find_recovery(self):
        import matlab
        S = self._code.encoder(kraus=True)
        physdim = S.dims[0][0]
        codedim = S.dims[1][0]

        # C = qt.kraus_to_choi([(1./codedim)*S.dag()*k.dag()
        #                       for k in self._noise.kraus])
        C = (1./codedim**2)*(self._noise.channel_matrix *
                             self._code.encoder()).dag()
        C = qt.super_to_choi(C)
        C = matlab.double(C.data.toarray().tolist(), is_complex=True)
        X = self._eng.cvxsdp(C, physdim, codedim)
        X = np.array(X)

        dims = [[[physdim], [codedim]], [[physdim], [codedim]]]
        choi = qt.Qobj(X, dims=dims, superrep='choi')
        channels.Channel.__init__(self, choi=choi)

    def find_encoder(self, recovery, tol=1e-8):
        import matlab
        physdim = self._code.encoder().dims[0][0][0]
        codedim = self._code.decoder().dims[0][0][0]

        D = (1./codedim**2)*(recovery*self._noise.channel_matrix).dag()
        D = qt.super_to_choi(D)
        D = matlab.double(D.data.toarray().tolist(), is_complex=True)
        X = self._eng.cvxsdp(D, codedim, physdim)
        X = np.array(X)

        dims = [[[codedim], [physdim]], [[codedim], [physdim]]]
        choi = qt.Qobj(X, dims=dims, superrep='choi')
        # remove all negligible eigenvectors from choi matrix
        kraus = codes.choi_to_kraus(choi, tol=1e-8)
        return kraus

    def __Cmatrix_old(self, kraus, S, dim):
        rho = (1/dim)*qt.identity(dim)
        klist = [rho*S.dag()*k.dag() for k in kraus]
        return qt.kraus_to_choi(klist)
