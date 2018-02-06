import numpy as np
import math


_FLOAT_EPS_4 = np.finfo(float).eps * 4.0

def QuatSum(Q, R):
	T = [0, 0, 0, 0]
	T[0] = Q[0] + R[0]
	T[1] = Q[1] + R[1]
	T[2] = Q[2] + R[2]
	T[3] = Q[3] + R[3]
    
	return T
	
def QuatMultiply(Q, R):
    T = [0, 0, 0, 0]
    Qs = Q[0]
    Qv = np.array(Q[1:4])
    Rs = R[0]
    Rv = np.array(R[1:4])
    T[0] = Qs*Rs - np.dot(Qv,Rv)
    T[1:4] = Qs*Rv + Rs*Qv + np.cross(Qv,Rv)
    
    return T

def QuatConjugate(Q):
	T = [0, 0, 0, 0]
	Qs = Q[0]
	Qv = np.array(Q[1:4])
	
	T[0] = Qs
	T[1:4] = -Qv
	return T
	
def QuatNorm(Q):
	T = Q[0]**2 + Q[1]**2 + Q[2]**2 + Q[3]**2 
	return T
	
def NormalizeQuat(Q):
	T = [0,0,0,0]
	T = Q
	QN = QuatNorm(np.array(Q))
	if(QN>0):
		T = np.array(Q)/math.sqrt(QN)
	
	return T
	
def QuatInverse(Q):
    T = QuatConjugate(Q)

    T[0] = float(T[0])/float(QuatNorm(Q))
    T[1] = float(T[1])/float(QuatNorm(Q))
    T[2] = float(T[2])/float(QuatNorm(Q))
    T[3] = float(T[3])/float(QuatNorm(Q))
    
    return T

def QuatLog(Q):
	T = [0, 0, 0, 0]
	Qv = np.array(Q[1:4])
	if (np.linalg.norm(Q) > 0):
		T[0] = np.log(math.sqrt(QuatNorm(Q)))
	if (np.linalg.norm(Qv) > 0):
		T[1:4] = (Qv/np.linalg.norm(Qv))*math.acos(Q[0]/math.sqrt(QuatNorm(Q)))

	return T
	
def QuatExp(Q):
	T = [0, 0, 0, 0]
	H = [0, 0, 0, 0]
	Qs = Q[0]
	Qv = np.array(Q[1:4])
	Scalar = math.exp(Qs)
	QN = math.sqrt(Qv[0]**2 + Qv[1]**2 + Qv[2]**2)
	T[0] = Scalar*math.cos(QN)
	if(QN>0):
		H[1:4] = Scalar*math.sin(QN)*Qv/QN
		if(H[1].shape == ()):
			T[1:4] = H[1:4]
		else:
			T[1] = H[1][0]
			T[2] = H[1][1]
			T[3] = H[1][2]			

	return T

def Quat2Rot(Q):
    w, x, y, z = Q
    QN = QuatNorm(Q)
    if np.all(QN < _FLOAT_EPS_4):
        return np.eye(3)
    s = 2.0 / QN
    X = x * s
    Y = y * s
    Z = z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z
    return np.array([[1.0 - (yY + zZ), xY - wZ, xZ + wY],
                     [xY + wZ, 1.0 - (xX + zZ), yZ - wX],
                     [xZ - wY, yZ + wX, 1.0 - (xX + yY)]])

def Rot2Quat(M):
    Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat

    K = np.array([
        [Qxx - Qyy - Qzz, 0, 0, 0],
        [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0],
        [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
        [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]]
    ) / 3.0

    vals, vecs = np.linalg.eigh(K)
    Q = vecs[[3, 0, 1, 2], np.argmax(vals)]

    if Q[0] < 0:
        Q *= -1
    return Q

def RotVector(V, Q):
    Vec = np.zeros((4,))
    Vec[1:] = V
    return QuatMultiply(Q, QuatMultiply(Vec, QuatConjugate(Q)))[1:]
	
def Euler2Rot(x=0, y=0, z=0):
    Ms = []
    if z:
        cosz = math.cos(z)
        sinz = math.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = math.cos(y)
        siny = math.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = math.cos(x)
        sinx = math.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def Rot2Euler(M, cy_thresh=None):
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: 
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return x,y,z


def Euler2Quat( x=0, y=0, z=0 ):
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    return np.array([
             cx*cy*cz - sx*sy*sz,
             cx*sy*sz + cy*cz*sx,
             cx*cz*sy - sx*cy*sz,
             cx*cy*sz + sx*cz*sy])


def Quat2Euler(Q):
    return Rot2Euler(Quat2Rot(Q))
	
def quaternion_to_rotation(q):
	assert isinstance(q, np.ndarray), 'q must be of type numpy.ndarray'
	assert len(q.shape) == 1, 'function only accepts singular quaternions'
	assert len(q) == 4, 'a quaternion must have 4 elements'

	if np.array_equal(q, np.array([1, 0, 0, 0])):
		return np.zeros((3, 1))

	w0 = 2 * np.array(QuatLog(q))

	return w0[1:]

def rotation_to_quaternion(w):
	vector = [0,0,0,0]
	vector[1:4] = w
	q = np.array(QuatExp(np.array(vector)/2))

	
	return q
	
def RotVel2Quat(W, dt):
	x = np.array([0,0,0,0],dtype=np.float64)
	x[1:] = np.array(W)*dt
	
	Q = QuatExp(x)
	
	return Q

def RotVel2QuatIntagration(W, dt):	
	Q = [0,0,0,0]
	angle = np.linalg.norm(np.array(W))
	axis = [0,0,0]
	if (angle>0):
		axis = np.array(W)/angle
	Q[0] = math.cos(angle*dt/2)
	Q[1:4] = np.array(axis)*math.sin(angle*dt/2)
	return Q
	
def QuatAve(Q, Qguess, alphas):
	'''
	INPUTS:::
		Q					:=	Quaternions vectors to average
		Qguess				:=	Initial guess for average
		alphas				:=	Weight vector for averaging
	OUTPUTS:::
		Qguess				:=	Predicted Quaternion Average of Q
		Covar				:=	Error Covariance
		EviMod				:=	Error vectors
	'''
	numQuats = len(Q)
	D = 3
	EviMod = np.zeros((numQuats, 3))
	EviHold = np.zeros((numQuats, 3))

	Covar = np.zeros((3, 3))
	C = float(1)/(2*D)
	EvNORM = 1
	guessQuat = [0,0,0,0]
	#print "going into algorithm"
	while (EvNORM > 0.0001):
		EviMod = np.zeros((numQuats, 3))
		EviHold = np.zeros((numQuats, 3))
		for i in range(numQuats):
			Qt = QuatMultiply(QuatInverse(Qguess), Q[i])
			Evi = np.array([0,0,0,0])
			Evi = 2*np.array(QuatLog(Qt))
			QN = math.sqrt(Evi[1]**2 + Evi[2]**2 + Evi[3]**2)
			if(QN>0):
				EviMod[i,:] = (-math.pi + (QN + math.pi)%(2*math.pi))*np.array([Evi[1],Evi[2],Evi[3]])/QN
	
		for i in range(numQuats):
			EviHold[i,:] = EviMod[i,:]*alphas[i]
			
		Ev = np.sum(EviHold, axis=0)
		guessQuat[1:4] = Ev
		Qguess = QuatMultiply(Qguess, QuatExp(np.array(guessQuat)/2.0))
		EvNORM = math.sqrt(Ev[0]**2 + Ev[1]**2 + Ev[2]**2)
		#print "Norm of Ev: ", EvNORM
		
	Covar += np.array(2*np.outer(EviMod[0,:], EviMod[0,:]))
	for i in range(1,numQuats):
		Covar += np.array((float(1)/6.0)*np.outer(EviMod[i,:],EviMod[i,:]))
			
		

	return Qguess, EviMod, Covar