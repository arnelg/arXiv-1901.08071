import numpy as np
import scipy as sp

import qutip as qt

import channels
import codes
import recoveries


class LossChannel(channels.Channel):
    """
    Sometimes errors arise if lmax is too big, probably due to overflow when
    taking the factorial??

    gamma = kappa tau
    """

    def __init__(self, gamma, dim, lmax):
        kt = gamma
        a = qt.destroy(dim)
        k_list = []
        x = np.exp(-kt)
        n_fac = channels.n_pow_op(np.sqrt(x), 1, dim)
        for l in range(lmax):
            c = channels.factorial_quotient(1-x, l)
            k_list.append(np.sqrt(c)*n_fac*a**l)
        self._kt = kt
        self._kraus = k_list
        channels.Channel.__init__(self, k_list)

    @property
    def kraus(self):
        return self._kraus

    def propagate(self, ancilla_dim, N, M):
        n = qt.num(ancilla_dim)
        return [(-1j*np.pi*k/(N*M)*n).expm() for k in range(len(self.kraus))]


class TeleCorrector():

    def __init__(self, code, ancilla, data_meas, ancilla_meas,
                 code_out=None):
        self.code = code
        self.ancilla = ancilla
        if code_out is None:
            code_out = codes.TrivialCode()
        self.code_out = code_out
        self.data_meas = data_meas
        self.ancilla_meas = ancilla_meas

    def channel(self, loss, dephasing, verbose=False):
        data_kraus = loss.kraus
        ancilla_phase = loss.propagate(self.ancilla.dim, self.code.N,
                                       self.ancilla.N)
        nkraus = len(data_kraus)
        ndata_meas = len(self.data_meas.povm_elements)
        nancilla_meas = len(self.ancilla_meas.povm_elements)
        cmat1 = np.zeros((ndata_meas, nkraus, 2, 2), dtype=complex)
        cmat2 = np.zeros((nancilla_meas, nkraus, 2, 2), dtype=complex)

        # compute c1(x1|k;a;b), c2(x2|k;a;b) matrices
        for k, (n, p) in enumerate(zip(data_kraus, ancilla_phase)):
            cmat1[:, k, :, :] = np.reshape(
                 [qt.expect(m, n*dephasing(keta*ketb.dag())*n.dag())
                  for m in self.data_meas.povm_elements
                  for keta in [self.code.plus, self.code.minus]
                  for ketb in [self.code.plus, self.code.minus]],
                 cmat1[:, k, :, :].shape)
            cmat2[:, k, :, :] = np.reshape(
                 [qt.expect(m, p*keta*ketb.dag()*p.dag())
                  for m in self.ancilla_meas.povm_elements
                  for keta in [self.ancilla.plus, self.ancilla.minus]
                  for ketb in [self.ancilla.plus, self.ancilla.minus]],
                 cmat2[:, k, :, :].shape)

        # compute p1(x1|k;a), p2(x2|k;a) matrices
        pmat1 = np.diagonal(cmat1, axis1=2, axis2=3)
        pmat2 = np.diagonal(cmat2, axis1=2, axis2=3)

        # compute c(x1, x2| a, a', b') and p(x1, x2| a, b)
        cmat = np.moveaxis(np.tensordot(cmat1, cmat2, axes=([1], [1])), 3, 1)
        pmat = np.moveaxis(np.tensordot(pmat1, pmat2, axes=([1], [1])), 2, 1)

        # determine most likely a, b
        abmat = np.empty((ndata_meas, nancilla_meas), dtype=object)
        for x1 in range(ndata_meas):
            for x2 in range(nancilla_meas):
                idx = (x1, x2, Ellipsis)
                a, b = np.unravel_index(np.argmax(pmat[idx]),
                                        pmat[idx].shape)
                abmat[x1, x2] = (a, b)

        # construct channel
        channel = qt.Qobj()
        Paulis = np.array(
                [[self.code_out.identity,
                  self.code_out.logical_Z_allspace],
                 [self.code_out.logical_X_allspace,
                  self.code_out.logical_X_allspace
                  * self.code_out.logical_Z_allspace]])
        for a1, a2 in np.ndindex(2, 2):
            for b1, b2 in np.ndindex(2, 2):
                for x1 in range(ndata_meas):
                    for x2 in range(nancilla_meas):
                        r = Paulis[abmat[x1, x2]].dag()
                        channel += (cmat[x1, x2, a1, b1, a2, b2]
                            * qt.sprepost(r*Paulis[a1, a2],
                                          Paulis[b1, b2].dag()*r.dag()))
        return 0.25*channel


class HybridCorrector():

    def __init__(self, code, ancilla_state, data_meas, ancilla_meas,
                 code_out=None, M=1/2):
        self.code = code
        if ancilla_state.type == 'ket':
            self.ancilla_state = qt.ket2dm(ancilla_state)
        else:
            self.ancilla_state = ancilla_state
        if code_out is None:
            code_out = codes.TrivialCode()
        self.code_out = code_out
        self.ancilla_dim = self.ancilla_state.dims[0][0]
        self.data_meas = data_meas
        self.ancilla_meas = ancilla_meas
        self.M = M

    def channel(self, loss, dephasing, rep=1, verbose=False):
        """ Computes logical channel """
        data_kraus = loss.kraus
        ancilla_phase = loss.propagate(self.ancilla_dim, self.code.N, self.M)
        out_phase = loss.propagate(self.code_out.dim, self.code.N,
                                   self.code_out.N)
        nkraus = len(data_kraus)
        ndata_meas = len(self.data_meas.povm_elements)
        nancilla_meas = len(self.ancilla_meas.povm_elements)
        pmat = np.zeros((nancilla_meas, nkraus), dtype=complex)
        cmat = np.zeros((ndata_meas, nkraus, 2, 2), dtype=complex)

        # compute p(x|k) and c(y|k;a;b) matrices
        for k, (n, p) in enumerate(zip(data_kraus, ancilla_phase)):
            pmat[:, k] = [qt.expect(m, p*self.ancilla_state*p.dag())
                          for m in self.ancilla_meas.povm_elements]

            cmat[:, k, :, :] = np.reshape(
                [qt.expect(m, n*dephasing(keta*ketb.dag())*n.dag())
                 for m in self.data_meas.povm_elements
                 for keta in [self.code.plus, self.code.minus]
                 for ketb in [self.code.plus, self.code.minus]],
                cmat[:, k, :, :].shape)

        # compute p(x1,...,xn|k) matrix
        pnmat = np.zeros((*[nancilla_meas]*rep, nkraus), dtype=complex)
        for k in range(nkraus):
            out = pmat[:, k]
            for i in range(rep-1):
                out = np.outer(pmat[:, k], out)
            pnmat[..., k] = out.reshape(*[nancilla_meas]*rep)

        # compute q(y|k;a) matrix
        qmat = np.diagonal(cmat, axis1=2, axis2=3)

        # compute p(x;y|k;a) = p(x|k)q(y|k;a) matrix
        pqmat = np.zeros((*[nancilla_meas]*rep, ndata_meas, nkraus, 2),
                         dtype=complex)
        for k in range(nkraus):
            for a in range(2):
                pqmat[..., k, a] = (np.outer(pnmat[..., k], qmat[:, k, a])
                                    ).reshape(*[nancilla_meas]*rep, ndata_meas)

        # determine most likely k and a
        kamat = np.empty((*[nancilla_meas]*rep, ndata_meas), dtype=object)
        # for x in range(nancilla_meas):
        for x in np.ndindex(*[nancilla_meas]*rep):
            for y in range(ndata_meas):
                idx = x + (y, Ellipsis)
                k, a = np.unravel_index(np.argmax(pqmat[idx]),
                                        pqmat[idx].shape)
                kamat[x + (y,)] = (k, a)

        # construct channel
        channel = qt.Qobj()
        clogical = [self.code_out.logical_H_allspace,
                    self.code_out.logical_X_allspace
                    * self.code_out.logical_H_allspace]
        recs = []
        for k, p in enumerate(out_phase):
            for a, ca in enumerate(clogical):
                for b, cb in enumerate(clogical):
                    # for x in range(nancilla_meas):
                    for x in np.ndindex(*[nancilla_meas]*rep):
                        for y in range(ndata_meas):
                            xyidx = x + (y,)
                            r = (
                                 clogical[kamat[xyidx][1]].dag()
                                 * out_phase[kamat[xyidx][0]].dag())
                            recs.append(qt.sprepost(r, r.dag()))
                            if verbose:
                                tmp = np.abs(pnmat[x + (k,)]*cmat[y, k, a, b])
                                if (tmp > 1e-5 and (k != kamat[xyidx][0]
                                                    or a != kamat[xyidx][1])):
                                    print('syndrome:', k, kamat[xyidx][0], tmp)
                                    print(a, b, kamat[xyidx][1])
                            channel += (pnmat[x + (k,)]*cmat[y, k, a, b]
                                        * qt.sprepost(r*p*ca, cb.dag()*p.dag()*r.dag()))
        self.recs = recs
        return 0.5*channel
