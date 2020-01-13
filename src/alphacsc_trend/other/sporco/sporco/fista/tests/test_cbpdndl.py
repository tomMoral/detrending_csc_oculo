from __future__ import division
from builtins import object

import pytest
import numpy as np

from sporco.fista import cbpdndl



class TestSet01(object):

    def setup_method(self, method):
        N = 16
        Nd = 5
        M = 4
        K = 3
        np.random.seed(12345)
        self.D0 = np.random.randn(Nd, Nd, M)
        self.S = np.random.randn(N, N, K)


    def test_01(self):
        lmbda = 1e-1
        opt = cbpdndl.ConvBPDNDictLearn.Options({'MaxMainIter': 10})
        try:
            b = cbpdndl.ConvBPDNDictLearn(self.D0, self.S[...,0], lmbda,
                                          opt=opt, dimK=0)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_02(self):
        lmbda = 1e-1
        opt = cbpdndl.ConvBPDNDictLearn.Options({'MaxMainIter': 10})
        try:
            b = cbpdndl.ConvBPDNDictLearn(self.D0, self.S, lmbda, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)



    def test_03(self):
        lmbda = 1e-1
        opt = cbpdndl.MixConvBPDNDictLearn.Options({'MaxMainIter': 10})
        try:
            b = cbpdndl.MixConvBPDNDictLearn(self.D0, self.S,
                                lmbda, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_04(self):
        lmbda = 1e-1
        opt = cbpdndl.MixConvBPDNDictLearn.Options({'MaxMainIter': 10,
                                        'CCMOD' : { 'BackTrack': {'Enabled': True} } })
        try:
            b = cbpdndl.MixConvBPDNDictLearn(self.D0, self.S,
                                lmbda, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_05(self):
        N = 16
        Nc = 3
        Nd = 5
        M = 4
        K = 3
        D0 = np.random.randn(Nd, Nd, Nc, M)
        S = np.random.randn(N, N, Nc, K)
        lmbda = 1e-1
        opt = cbpdndl.ConvBPDNDictLearn.Options({'MaxMainIter': 10})
        try:
            b = cbpdndl.ConvBPDNDictLearn(D0, S, lmbda, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_06(self):
        lmbda = 1e-1
        opt = cbpdndl.ConvBPDNDictLearn.Options({'AccurateDFid': True,
                                                 'MaxMainIter': 10})
        try:
            b = cbpdndl.ConvBPDNDictLearn(self.D0, self.S, lmbda, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_07(self):
        lmbda = 1e-1
        W = np.ones(self.S.shape[0:2] + (1, self.S.shape[2], 1))
        opt = cbpdndl.MixConvBPDNMaskDcplDictLearn.Options({'MaxMainIter': 10})
        try:
            b = cbpdndl.MixConvBPDNMaskDcplDictLearn(self.D0, self.S, lmbda, W,
                                                  opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_08(self):
        lmbda = 1e-1
        W = np.ones(self.S.shape[0:2] + (1, self.S.shape[2], 1))
        opt = cbpdndl.MixConvBPDNMaskDcplDictLearn.Options(
            {'MaxMainIter': 5, 'CCMOD': {'BackTrack': {'Enabled': True}}})
        try:
            b = cbpdndl.MixConvBPDNMaskDcplDictLearn(self.D0, self.S,
                                        lmbda, W, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_09(self):
        lmbda = 1e-1
        L = 1e2
        W = np.ones(self.S.shape[0:2] + (1, self.S.shape[2], 1))
        opt = cbpdndl.MixConvBPDNMaskDcplDictLearn.Options(
            {'MaxMainIter': 5, 'CCMOD': {'L': L, 'BackTrack': {'Enabled': True}}})
        try:
            b = cbpdndl.MixConvBPDNMaskDcplDictLearn(self.D0, self.S,
                                        lmbda, W, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_10(self):
        lmbda = 1e-1
        W = np.ones(self.S.shape[0:2] + (1, self.S.shape[2], 1))
        opt = cbpdndl.MixConvBPDNMaskDcplDictLearn.Options(
            {'AccurateDFid': True, 'MaxMainIter': 5})
        try:
            b = cbpdndl.MixConvBPDNMaskDcplDictLearn(self.D0, self.S,
                                        lmbda, W, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_11(self):
        N = 16
        Nc = 3
        Nd = 5
        M = 4
        K = 3
        D0 = np.random.randn(Nd, Nd, Nc, M)
        S = np.random.randn(N, N, Nc, K)
        lmbda = 1e-1
        W = np.ones((N, N, 1, K, 1))
        opt = cbpdndl.MixConvBPDNMaskDcplDictLearn.Options({'MaxMainIter': 10})
        try:
            b = cbpdndl.MixConvBPDNMaskDcplDictLearn(D0, S, lmbda, W, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)


    def test_12(self):
        N = 16
        Nc = 3
        Nd = 5
        M = 4
        K = 3
        D0 = np.random.randn(Nd, Nd, 1, M)
        S = np.random.randn(N, N, Nc, K)
        lmbda = 1e-1
        W = np.ones((N, N, Nc, K, 1))
        opt = cbpdndl.MixConvBPDNMaskDcplDictLearn.Options({'MaxMainIter': 10})
        try:
            b = cbpdndl.MixConvBPDNMaskDcplDictLearn(D0, S, lmbda, W, opt=opt)
            b.solve()
        except Exception as e:
            print(e)
            assert(0)

