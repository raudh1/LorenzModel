import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class LorenzMap:
    """
    LorenzMap
    With sigma=10, rho=28, and beta=8/3
    """
    def __init__(self, sigma=10, rho=28, beta=8 / 3, delta_t=1e-3):
        self.sigma, self.rho, self.beta = sigma, rho, beta
        self.delta_t = delta_t

    def v_eq(self, t=None, v=None):
        x, y, z = v[0], v[1], v[2]
        dot_x = self.sigma * (-x + y)
        dot_y = x * (self.rho - z) - y
        dot_z = x * y - self.beta * z
        return np.array([dot_x, dot_y, dot_z])

    def step(self, v):
        return v + self.v_eq(v=v) * self.delta_t

    def jacobian(self, v):
        x, y, z = v[0], v[1], v[2]
        res = np.array([[ -self.sigma, self.sigma,       0],
                    [self.rho - z,      -1,      -x],
                    [        y,       x, -self.beta]])
        return res

    def full_traj(self, nb_steps, init_pos):
        t = np.linspace(0, nb_steps * self.delta_t, nb_steps)
        f = solve_ivp(self.v_eq, [0, nb_steps * self.delta_t], init_pos, method='RK45', t_eval=t)
        return np.moveaxis(f.y, -1, 0)

    def plot_traj(self, data, filename, str_params=""):
        data = data
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot(x, y, z, lw=0.5, color='blue')
        ax.set_xlabel("X Axis")
        ax.set_xlim([])
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title(str_params)
        plt.savefig("./"+filename+".png")
        plt.show()

    def plot_traj_part(self, data, data_part, filename):
        data = data
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        xp, yp, zp = data_part[:, 0], data_part[:, 1], data_part[:, 2]
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot(xp, yp, zp, lw=0.5, color='blue')
        ax.set_xlabel("X Axis")
        ax.set_xlim([min(x), max(x)])
        ax.set_ylabel("Y Axis")
        ax.set_ylim([min(y), max(y)])
        ax.set_zlabel("Z Axis")
        ax.set_zlim([min(z), max(z)])
        plt.savefig("./"+filename+".png")
        plt.show()

    def plot_traj_critical(self, data, filename):
        data = data
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot(x, y, z, lw=0.5, color='blue')
        crit1 = [np.sqrt(self.beta * (self.rho - 1)), np.sqrt(self.beta * (self.rho - 1)), self.rho - 1]
        crit2 = [-np.sqrt(self.beta * (self.rho - 1)), -np.sqrt(self.beta * (self.rho - 1)), self.rho - 1]
        ax.scatter([crit1[0], crit2[0]], [crit1[1], crit2[1]], [crit1[2], crit2[2]], s=10, color='red')
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        plt.savefig("./"+filename+".png")
        plt.show()

    def plot_parts_traj(self, data, highlight, name):
        data = data
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.plot(x, y, z, lw=0.5, color='blue')

        ax.plot(x[highlight[0]:highlight[1]], y[highlight[0]:highlight[1]], z[highlight[0]:highlight[1]], lw=0.5, color='red')
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        plt.savefig("./" + name + ".png")
        plt.close(fig)

if __name__ == '__main__':
    ### To save & show a figure with the following parameters
    sigma, rho, beta, delta_t = 10, 28, 8/3, 1e-3
    # lor = LorenzMap(sigma, rho, beta, delta_t)
    # init = array([0, -50, 20])
    init = np.array([0, 1, 1.05])

    # x = init
    # nb_steps = 1000
    # for i in range(nb_steps):
    #     x = lor.step(x)
    # print(x)

    lor = LorenzMap(sigma, rho, beta, delta_t)
    traj1 = lor.full_traj(100000, init)
    lor.plot_parts_traj(traj1, [0, 10000], "traj1")
    lor.plot_parts_traj(traj1, [5000, 15000], "traj2")

    # lor2 = LorenzMap(sigma, rho, beta, delta_t/10)
    # traj2 = lor2.full_traj(100000 * 10, init)[::10]

    # print(traj1[:10])
    # print(traj2[:10])
    # print(sum(abs(np.array(traj1) - np.array(traj2))))

    # for nb_steps in [10, 100, 1000, 10000, 100000]:
    #     str_params = "Sigma = " + str(sigma) + ", rho = " + str(rho) + ", beta = " + str(beta) + ", delta t = " + str(delta_t) + ",\n init pos = " + str(init) + " for " + str(nb_steps) + " steps"
    #     traj = lor.full_traj(nb_steps, init)
    #     lor.plot_traj(traj, str_params, "test_traj_5_"+ str(nb_steps)+"steps")

    ### To print parts of the full trajectory, with beginning highlighted
    # LORENZ_MAP = LorenzMap(delta_t=5e-3)
    # INIT = array([0, 1, 1.05])
    # nb_steps = 135000
    # traj = LORENZ_MAP.full_traj(nb_steps, INIT)
    # size_highlight = 500
    # for i in range(0, nb_steps, size_highlight):
    #     LORENZ_MAP.plot_traj(traj, traj[i:i+size_highlight], "traj_pt_"+str(i)+"_to_"+str(i + size_highlight))
