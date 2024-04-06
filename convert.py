import scipy.io as sio
import numpy as np
import torch
for i in range(10):
	x = torch.load("matrices/X_128_120_"+str(i)+".pt")
	if i>0:
		x_new = torch.concat((x_new, x), axis = 0)
	else:
		x_new = x
		# print((x.size()))

	mat = sio.loadmat("matrices/H_128_120_"+str(i)+".mat")
	h = mat['H']
	# print(h)
	if i>0:
		h_new = np.concatenate((h_new, h), axis = 0)
	else:
		h_new = h

X = torch.from_numpy(np.reshape(x_new.cpu().numpy(), (10000,128)))
H = torch.from_numpy(np.reshape(h_new, (10000,32,32)))
print(X.size())
print(H.size())

torch.save(X,"X_128_120kmph.pt")
torch.save(H,"H_128_120kmph.pt")
