import os

from regulateme.run import *

base_path = 'regulateme/data_mini'

SS = np.load(os.path.join(base_path, 'SS.npy'))
RR = np.load(os.path.join(base_path, 'RR.npy'))
CC = np.load(os.path.join(base_path, 'CC.npy'))
KK = np.load(os.path.join(base_path, 'KK.npy'))
ll = np.load(os.path.join(base_path, 'll.npy'))
uu = np.load(os.path.join(base_path, 'uu.npy'))
ff = np.load(os.path.join(base_path, 'ff.npy'))
ww = np.load(os.path.join(base_path, 'ww.npy'))
tt0 = np.load(os.path.join(base_path, 'tt0.npy'))
mm = np.load(os.path.join(base_path, 'mm.npy'))

D = float(np.load(os.path.join(base_path, 'D.npy')))
P = float(np.load(os.path.join(base_path, 'P.npy')))

v = np.zeros(len(ll))
p = D*np.ones(len(tt0))
e = D*np.ones(CC.shape[1])
t = D*np.ones(len(tt0))
a = np.zeros(RR.shape[1])





variable = Variable(v, p, t, e, a)
parameter = Parameter(D, P, ll, tt0, uu, ww, CC, KK, RR, SS, mm)
x, info = optimizer(ff, variable, parameter)

print(info)