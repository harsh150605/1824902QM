import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y

th = np.linspace(0, np.pi, 100)
ph = np.linspace(-np.pi, np.pi, 100)
l = 5
m = 5
th, ph = np.meshgrid(th, ph)

Ylm = sph_harm_y(l, m, th, ph)
rho = np.abs(Ylm) ** 2
R = np.abs(Ylm)

x = R * np.sin(th) * np.cos(ph)
y = R * np.sin(th) * np.sin(ph)
z = R * np.cos(th)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
surface = ax.plot_surface(x, y, z)
ax.set_title('spherical harmonics')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
