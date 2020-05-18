import numpy as np

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[0]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B  
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.m) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.I = np.eye(self.n)
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        F, x, B, P, Q = self.F, self.x, self.B, self.P, self.Q
        self.x = F @ x + np.dot(B, u)
        self.P = F @ P @ F.T + Q 
        return self.x

    def update(self, z):
        H, P, R, I = self.H, self.P, self.R, self.I 
        y = z - H @ self.x
        S = R + H @ P @ H.T
        K = P @ H.T @ np.linalg.inv(S)        
        IK = I - K @ H
        self.x = self.x + K @ y                 
        self.P = (IK @ P) @ (IK.T + K @ R @ K.T) 

def example():
	dt = 1.0/60
	F = np.array([[1, dt, 0.5*dt**2], [0, 1, dt], [0, 0, 1]])
	H = np.array([1, 0, 0]).reshape(1, 3)
	Q = np.array([[0.01, 0.0, 0.0], 
                  [0.0, 0.001, 0.0], 
                  [0.0, 0.0, 0.91]])
	R = np.array([8]).reshape(1, 1)

	x = np.linspace(-10, 10, 100)
	measurements = - (x**2 + 2*x - 2)  + np.random.normal(0, 2, 100)

	kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
	predictions = []

	for z in measurements:
		predictions.append(np.dot(H,  kf.predict())[0])
		kf.update(z)

	import matplotlib.pyplot as plt
	plt.plot(range(len(measurements)), measurements, label = 'Measurements')
	plt.plot(range(len(predictions)), np.array(predictions), label = 'Kalman Filter Prediction')
	plt.legend()
	plt.show()

if __name__ == '__main__':
    example()
