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


class CROTTeleport(channels.Channel):
    """ Chooses recovery based on overlap with phase state"""

    def __init__(self, code, ancilla=None, noise=None, offset=None,
                 correct=True, safety=True):
        if ancilla is None:
            ancilla = code
        if safety and not code.dim % (2*code.N) == 0:
            raise ValueError('code.dim should be a multiple of 2N.')
        if safety and not ancilla.dim % (2*ancilla.N) == 0:
            raise ValueError('code.dim should be a multiple of 2N.')
        if offset is None:
            offset = 0.
        if noise is None:
            noise = channels.IdentityChannel(code.dim)

        # CROT |psi> |+>
        crot = code.crot(ancilla)
        crot = qt.sprepost(crot, crot.dag())
        # ancilla in |+>
        q_prep = qt.tensor(code.identity, ancilla.plus)
        s_prep = qt.sprepost(q_prep, q_prep.dag())
        crot_prep = crot*s_prep

        # phase measurement on data and correct for Pauli's
        s_post = []
        X_gate = qt.sprepost(ancilla.logical_X, ancilla.logical_X.dag())
        H_gate = qt.sprepost(ancilla.logical_H, ancilla.logical_H.dag())
        channel = qt.Qobj()

        noisy_plus = noise.channel_matrix(code.plus)
        noisy_minus = noise.channel_matrix(code.minus)

        for r in range(code.dim):
            phi = qt.phase_basis(code.dim, r, phi0=offset)
            q = qt.tensor(phi, ancilla.identity)
            s_post = qt.sprepost(q, q.dag())
            out = s_post.dag()*crot_prep
            if not correct:
                channel += out
            else:
                if (qt.expect(phi*phi.dag(), noisy_plus) >
                        qt.expect(phi*phi.dag(), noisy_minus)):
                    # closer to |+>
                    channel += H_gate*out
                else:
                    # closer to |->
                    channel += H_gate*X_gate*out
        channels.Channel.__init__(self, channel_matrix=channel)


class CROTTeleport2(channels.Channel):
    """
    Chooses recovery based on wedges.
    """

    def __init__(self, code, ancilla=None, offset=None,
                 correct=True, safety=True):
        if ancilla is None:
            ancilla = code
        if safety and not code.dim % (2*code.N) == 0:
            raise ValueError('code.dim should be a multiple of 2N.')
        if safety and not ancilla.dim % (2*ancilla.N) == 0:
            raise ValueError('code.dim should be a multiple of 2N.')
        if offset is None:
            offset = int(code.dim/(4*code.N))

        # CROT |psi> |+>
        crot = code.crot(ancilla)
        crot = qt.sprepost(crot, crot.dag())
        # ancilla in |+>
        q_prep = qt.tensor(code.identity, ancilla.plus)
        s_prep = qt.sprepost(q_prep, q_prep.dag())
        crot_prep = crot*s_prep

        # phase measurement on data and correct for Pauli's
        s_post = []
        X_gate = qt.sprepost(ancilla.logical_X, ancilla.logical_X.dag())
        H_gate = qt.sprepost(ancilla.logical_H, ancilla.logical_H.dag())
        channel = qt.Qobj()

        j = int(code.dim/(2*code.N))

        for r in range(code.dim):
            # theta = (2*np.pi*r/code.dim - offset) % (2*np.pi)
            theta = 2*np.pi*r/code.dim
            phi = qt.phase_basis(code.dim, r, phi0=0.)
            q = qt.tensor(phi, ancilla.identity)
            s_post = qt.sprepost(q, q.dag())
            out = s_post.dag()*crot_prep
            if not correct:
                channel += out
            else:
                # thetamod = theta % (2*np.pi/code.N)
                rmod = (r+offset) % (2*j)
                # if 0 <= thetamod and thetamod < np.pi/code.N - epsilon:
                if 0 <= rmod and rmod < j:
                    # closer to |+>
                    channel += H_gate*out
                else:
                    # closer to |->
                    channel += H_gate*X_gate*out
        channels.Channel.__init__(self, channel_matrix=channel)


class CROTTeleport3(channels.Channel):
    """ Takes an arbitrary measurement. Much slower in general. """

    def __init__(self, code, ancilla=None, noise=None, correct=True,
                 measurement=None, recovery=None):
        if ancilla is None:
            ancilla = code
        if measurement is None:
            measurement = WedgeMeasurement(4*code.N, code.dim,
                                           offset=-np.pi/(2*code.N))
        self.measurement = measurement
        if noise is None:
            noise = channels.IdentityChannel(code.dim)

        # CROT |psi> |+>
        crot = code.crot(ancilla)
        crot = qt.sprepost(crot, crot.dag())
        # ancilla in |+>
        prep = qt.tensor(code.identity, ancilla.plus)
        prep = qt.sprepost(prep, prep.dag())
        prep = crot*prep

        if correct is True:
            X_gate = qt.to_super(ancilla.logical_X)
            H_gate = qt.to_super(ancilla.logical_H)
        noisy_plus = noise.channel_matrix(code.plus)
        noisy_minus = noise.channel_matrix(code.minus)
        channel = qt.Qobj()

        # super-operator used to delete mode after measurement
        nothing = qt.basis(1, 0)
        ida = ancilla.identity
        k_list = [qt.tensor(nothing*qt.basis(code.dim, i).dag(), ida)
                  for i in range(code.dim)]
        delete = qt.kraus_to_super(k_list)

        channel_elems = []
        for i, m in enumerate(measurement.povm_elements):
            msqrt = matrixf(m, lambda x: np.sqrt(x), safe=True)
            post = qt.to_super(qt.tensor(msqrt, ancilla.identity))
            # post = qt.tensor_contract(post, (0, 2))
            post = delete*post
            post.dims = [[[ancilla.dim], [ancilla.dim]],
                         [[code.dim, ancilla.dim], [code.dim, ancilla.dim]]]
            out = post*prep
            if not correct:
                channel += out
                channel_elems.append(out)
            elif recovery is not None:
                channel += recovery[i]*out
            else:
                if (qt.expect(m, noisy_plus) > qt.expect(m, noisy_minus)):
                    # closer to |+>
                    channel += H_gate*out
                else:
                    # closer to |->
                    channel += H_gate*X_gate*out
        # out = qt.tensor_contract(out, (0, 2))
        self.channel_elems = channel_elems
        channels.Channel.__init__(self, channel_matrix=channel)


class CROTencoder(channels.Channel):

    def __init__(self, code):
        qubitcode = TrivialCode()
        teleport = CROTTeleport(qubitcode, code).channel_matrix
        channels.Channel.__init__(self, channel_matrix=teleport)


class CROTdecoder(channels.Channel):

    def __init__(self, code):
        qubitcode = TrivialCode()
        teleport = CROTTeleport(code, qubitcode).channel_matrix
        channels.Channel.__init__(self, channel_matrix=teleport)


class DoubleCROT(channels.Channel):

    def __init__(self, code, ancilla=None, teleport=CROTTeleport,
                 kwargs1={}, kwargs2={}):
        if ancilla is None:
            ancilla = code
        crot1 = teleport(code, ancilla, **kwargs1)
        crot2 = teleport(ancilla, code, **kwargs2)
        channel = crot2.channel_matrix*crot1.channel_matrix
        channels.Channel.__init__(self, channel_matrix=channel)


class DoubleCROT2(channels.Channel):
    """ Double CROT with precomputed recoveries. """

    def __init__(self, code, recs, ancilla=None, teleport=CROTTeleport3,
                 kwargs1={}, kwargs2={}):
        if ancilla is None:
            ancilla = code
        crot1 = teleport(code, ancilla, correct=False, **kwargs1)
        crot2 = teleport(ancilla, code, correct=False, **kwargs2)
        recs = np.reshape(recs, (len(crot1.measurement.povm_elements),
                          len(crot2.measurement.povm_elements)))
        channel = sum((recs[i, j]*y*x
                       for i, x in enumerate(crot1.channel_elems)
                       for j, y in enumerate(crot2.channel_elems)))
        channels.Channel.__init__(self, channel_matrix=channel)


class DoubleCROTRecovery():

    def __init__(self, code, measurement=None, ancilla=None):
        """ Precompute stuff that is independent of noise. """
        self.code = code
        if ancilla is None:
            ancilla = code
        self.ancilla = ancilla
        if measurement is None:
            m1 = PhaseMeasurement(code.dim, offset=0.)
            m2 = PhaseMeasurement(ancilla.dim, offset=0.)
            measurement = channels.tensor_povm(m1, m2)

        self.povms = np.array(measurement.povm_elements)
        self.logicals = np.array([
            qt.tensor(self.code.plus, self.ancilla.plus),
            qt.tensor(self.code.plus, self.ancilla.minus),
            qt.tensor(self.code.minus, self.ancilla.plus),
            qt.tensor(self.code.minus, self.ancilla.minus),
            ])

        pauli = np.array([qt.identity(2), qt.sigmaz(), qt.sigmax(),
                         qt.sigmax()*qt.sigmaz()])

        def super(pi, pj, pk):
            return qt.sprepost(pk.dag()*pi, pj.dag()*pk)
        vsuper = np.vectorize(super)
        self.pmat = vsuper(pauli[:, None, None], pauli[None, :, None],
                           pauli[None, None, :])

    def channel(self, noise, return_on_code_space=False):
        """ Construct the logical channel given a noise model. """
        crot = qt.to_super(self.code.crot(self.ancilla))
        noise2 = crot*qt.composite(noise.channel_matrix,
                        qt.to_super(self.ancilla.identity))*crot.dag()

        # calculate c_{ij}(x) matrix
        def exp(psi1, psi2, m):
            return qt.expect(m, noise2(psi2*psi1.dag()))
        vexp = np.vectorize(exp, otypes=[complex])
        c_matrix = vexp(self.logicals[:, None, None],
                        self.logicals[None, :, None],
                        self.povms[None, None, :])

        c_red = np.diagonal(c_matrix, axis1=0, axis2=1)
        imax = np.argmax(np.real(c_red), axis=1)

        # construct logical channel
        c_coarse_grained = np.zeros((4, 4, 4), dtype=complex)
        for k in range(4):
            idx = np.where(imax == k)[0]
            c_coarse_grained[:, :, k] = np.sum(c_matrix[:, :, idx], axis=2)
        channel = 0.25*np.sum(c_coarse_grained*self.pmat)
        fid = 0.25*np.sum((c_coarse_grained[i, i, i] for i in range(4)))
        self._channelfid = fid
        self._avgfid = (2*fid+1)/3
        # return channels.Channel.__init__(channel_matrix=channel)
        if return_on_code_space:
            return self.code.decoder()*channel*self.code.decoder()
        else:
            return channel


class DoubleCROTFidelity():

    def __init__(self, code, ancilla=None, measurement=None):
        self.code = code
        if ancilla is None:
            ancilla = code
        self.ancilla = ancilla
        if measurement is None:
            m1 = PhaseMeasurement(code.dim)
            m2 = PhaseMeasurement(ancilla.dim)
            self.measurement = channels.tensor_povm(m1, m2)
        else:
            self.measurement = measurement

    def fidelity(self, noise):
        crot = qt.to_super(self.code.crot(self.ancilla))
        noise2 = crot*qt.composite(noise.channel_matrix,
                        qt.to_super(self.ancilla.identity))*crot.dag()
        logicals = [
                    noise2(qt.tensor(self.code.plus, self.ancilla.plus)),
                    noise2(qt.tensor(self.code.plus, self.ancilla.minus)),
                    noise2(qt.tensor(self.code.minus, self.ancilla.plus)),
                    noise2(qt.tensor(self.code.minus, self.ancilla.minus)),
                   ]

        fid = 0.25*np.sum((np.max([qt.expect(m, rho) for rho in logicals])
                          for m in self.measurement.povm_elements))
        return (2*fid+1)/3

    def find_optimal_recoveries(self, noise):
        crot = qt.to_super(self.code.crot(self.ancilla))
        noise2 = crot*qt.composite(noise.channel_matrix,
                        qt.to_super(self.ancilla.identity))*crot.dag()
        logicals = [
                    noise2(qt.tensor(self.code.plus, self.ancilla.plus)),
                    noise2(qt.tensor(self.code.plus, self.ancilla.minus)),
                    noise2(qt.tensor(self.code.minus, self.ancilla.plus)),
                    noise2(qt.tensor(self.code.minus, self.ancilla.minus)),
                   ]

        rec_idx = [np.argmax([qt.expect(m, rho) for rho in logicals])
                   for m in self.measurement.povm_elements]
        pauli = [qt.to_super(p) for p in [self.code.identity,
                 self.code.logical_X, self.code.logical_Z,
                 self.code.logical_X*self.code.logical_Z]]
        pauli_recs = [pauli[i] for i in rec_idx]
        return pauli_recs
