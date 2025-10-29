from torch.optim.lr_scheduler import StepLR
from torch.autograd import grad
import math, torch, time, os
import gmsh
import pygmsh
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
import random
from scipy.linalg import orthogonal_procrustes

random.seed(111)
np.random.seed(111)
torch.manual_seed(111)
torch.cuda.manual_seed(111)
torch.cuda.manual_seed_all(111)

fig_step = [0, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000,
            8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000,
            22000, 23000, 24000, 25000]

parser = argparse.ArgumentParser(description='hyper parameters')
parser.add_argument('--e', type=int, default=8000, help='Epochs')
parser.add_argument('--d', type=int, default=4, help='depth')
parser.add_argument('--n', type=int, default=64, help='width')
parser.add_argument('--beta', type=int, default=100, help='/beta')
parser.add_argument('--inter', type=int, default=6, help='the interval is [-inter, inter]')
parser.add_argument('--nx', type=int, default=64, help='Sampling')
parser.add_argument('--xi', type=float, default=1e-7, help='Threshold')
parser.add_argument('--omega', type=float, default=0.9, help='omega')
o_min = 0.9
o_max = 0.9
num1 = 1

args = parser.parse_args()

def GetGradients(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True, allow_unused=True)[0]


def V(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return 0.5 * (x ** 2 + y ** 2)

def plot_3D(model, params, step, num):
    x = np.linspace(-6, 6, 400)
    y = np.linspace(-6, 6, 400)
    z1 = (o_min + o_max) / 2
    # z1 = 0.7
    z = np.linspace(z1, z1, 1)
    [X, Y, Z] = np.meshgrid(x, y, z)
    X1, Y1 = np.meshgrid(x, y)
    X_test = np.concatenate([X.flatten()[:, None], Y.flatten()[:, None], Z.flatten()[:, None]], axis=1)
    x_test = torch.from_numpy(X_test).float()
    norm = torch.sum(model(x_test) * model(x_test)) * (2 * params["interval"]) ** 2 / 160000
    point1 = ((model(x_test)[:, 0] ** 2 + model(x_test)[:, 1] ** 2) / norm).detach().numpy().reshape(400, 400)

    fig1 = plt.figure(figsize=(6, 5))
    plt.gca().set_aspect('equal', adjustable='box')
    c0 = plt.pcolor(X1, Y1, point1, cmap='gist_rainbow')
    plt.xticks([-6, 0, 6], fontsize='15')
    plt.yticks([-6, 0, 6], fontsize='15')
    c0 = fig1.colorbar(c0)
    c0.ax.tick_params(labelsize='20')
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.xlabel('x', fontsize='15')
    plt.ylabel('y', fontsize='15')
    plt.text(5.5, 5.5, '', fontsize=25, ha='right', va='top')
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制三维曲面图
    surf = ax.plot_surface(X, Y, point1, cmap='coolwarm')

    # 添加颜色条
    fig.colorbar(surf, shrink=0.5, aspect=5)

    # 设置轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    '''
    plt.savefig('D:/用户/研究生文章/BSE/实验/实验结果/' + f'exam_{num}_{step}')
    # 显示图形
    # plt.show()
    plt.close()


class Net(torch.nn.Module):  # 训练一个旋转神经网络，用来逼近旋转项
    def __init__(self, params, device):
        super(Net, self).__init__()
        self.params = params
        self.device = device
        self.linearIn = nn.Linear(self.params["d"] + 1, self.params["width"])
        # self.linearIn = nn.Linear(1, self.params["width"])
        nn.init.xavier_normal_(self.linearIn.weight)
        nn.init.constant_(self.linearIn.bias, 0)

        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.m = nn.Linear(self.params["width"], self.params["width"])
            nn.init.xavier_normal_(self.m.weight)
            nn.init.constant_(self.m.bias, 0)
            self.linear.append(self.m)

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])
        nn.init.xavier_normal_(self.linearOut.weight)
        nn.init.constant_(self.linearOut.bias, 0)

    def forward(self, X):
        X = X.to(self.device)
        x = torch.sin(self.linearIn(X))
        for layer in self.linear:
            x_temp = torch.sin(layer(x))
            x = x_temp
        x = self.linearOut(x)
        x = torch.tanh(x)
        return x


def G(model, params, device):
    x = np.linspace(-1 * params["interval"], params["interval"], params["nx"])
    y = np.linspace(-1 * params["interval"], params["interval"], params["ny"])
    z = np.linspace(o_min, o_max, num1)
    [X, Z, Y] = np.meshgrid(x, z, y)

    x_train = np.concatenate([X.flatten()[:, None], Y.flatten()[:, None], Z.flatten()[:, None]], axis=1)
    X_train = torch.from_numpy(x_train).float().to(device)
    X_train = X_train.requires_grad_(True)
    U_pred = model(X_train).to(device)
    U_x_real = GetGradients(U_pred[:, 0], X_train)[:, 0].squeeze_()
    U_x_image = GetGradients(U_pred[:, 1], X_train)[:, 0].squeeze_()
    # U_xx_real = GetGradients(U_x_real, X_train)[:, 0].squeeze_()
    # U_xx_image = GetGradients(U_x_image, X_train)[:, 0].squeeze_()

    U_y_real = GetGradients(U_pred[:, 0], X_train)[:, 1].squeeze_()
    U_y_image = GetGradients(U_pred[:, 1], X_train)[:, 1].squeeze_()

    res1 = abs(U_x_real ** 2 + U_x_image ** 2 + U_y_real ** 2 + U_y_image ** 2).cpu().detach().numpy()
    res = (res1 / np.max(res1)).reshape(len(x), -1)
    [X, Y] = np.meshgrid(x, y)
    return x, y, res

def triangle_area(p1, p2, p3):
    """计算三角形面积"""
    return 0.5 * abs(np.cross(p2 - p1, p3 - p1))


def generate_adaptive_mesh(G, domain=[-6, 6, -6, 6], num_points=1500, base_mesh_size=0.5):
    with pygmsh.geo.Geometry() as geom:
        # 创建矩形边界
        points = [
            geom.add_point([domain[0], domain[2], 0], mesh_size=base_mesh_size),
            geom.add_point([domain[1], domain[2], 0], mesh_size=base_mesh_size),
            geom.add_point([domain[1], domain[3], 0], mesh_size=base_mesh_size),
            geom.add_point([domain[0], domain[3], 0], mesh_size=base_mesh_size),
        ]
        lines = [
            geom.add_line(points[0], points[1]),
            geom.add_line(points[1], points[2]),
            geom.add_line(points[2], points[3]),
            geom.add_line(points[3], points[0]),
        ]
        curve_loop = geom.add_curve_loop(lines)
        surface = geom.add_plane_surface(curve_loop)

        # 生成初始散点并根据 G(x) 调整 mesh_size
        np.random.seed(42)
        scatter_points = np.random.uniform([domain[0], domain[2]], [domain[1], domain[3]], (num_points, 2))
        for x in scatter_points:
            mesh_size = 1.0 / G(x[0], x[1])  # G(x) 越大，网格越稀疏
            geom.add_point([x[0], x[1], 0], mesh_size=max(0.1, min(mesh_size, 1.0)))

        # 生成网格
        mesh = geom.generate_mesh(dim=2)

    return mesh.points[:, :2], mesh.cells_dict["triangle"]

def compute_integrals(model, points_np, cells, params, trainstep, optimizer, scheduler):
    Z = np.ones(points_np.shape[0]) * (o_max + o_min) / 2
    point = np.concatenate([points_np[:, :], Z.flatten()[:, None]], axis=1)


    X_train = torch.tensor(point, dtype=torch.float32, requires_grad=True)

    for step in range(trainstep):
        U_pred = model(X_train)  # [N, 2] (real, imag)
        mass1 = U_pred[:, 0] ** 2 + U_pred[:, 1] ** 2
        norm = torch.tensor(0)
        model.zero_grad()

        for tri in cells:
            area = triangle_area(points_np[tri[0]], points_np[tri[1]], points_np[tri[2]])
            area = torch.tensor(area, dtype=torch.float32, requires_grad=True)
            mass = torch.mean(mass1[tri[0]] + mass1[tri[1]] + mass1[tri[2]]) / 3
            norm = area * mass + norm


        x1 = X_train[:, 0]
        y1 = X_train[:, 1]

        U_x_real = GetGradients(U_pred[:, 0], X_train)[:, 0].squeeze_()
        U_x_image = GetGradients(U_pred[:, 1], X_train)[:, 0].squeeze_()

        U_y_real = GetGradients(U_pred[:, 0], X_train)[:, 1].squeeze_()
        U_y_image = GetGradients(U_pred[:, 1], X_train)[:, 1].squeeze_()

        res3 = 0.9 * (-U_pred[:, 0] * (U_y_image * x1 - y1 * U_x_image) + U_pred[:, 1] * (
                x1 * U_y_real - y1 * U_x_real)) / norm

        res1 = 0.5 * abs(U_x_real ** 2 + U_x_image ** 2 + U_y_real ** 2 + U_y_image ** 2) / norm

        res2 = V(X_train).squeeze_() * (U_pred[:, 0] ** 2 + U_pred[:, 1] ** 2) / (norm)
        # res3 = params["omega"] * (-U_pred[:, 0] * (U_y_image * x1 - y1 * U_x_image) + U_pred[:, 1] * (
        #         x1 * U_y_real - y1 * U_x_real)) / norm ** 2
        res4 = params["beta"] * (U_pred[:, 0] ** 2 + U_pred[:, 1] ** 2) ** 2 / (2 * norm ** 2)

        Res = res1 + res2 - res3 + res4

        energy = 0.0
        for tri in cells:
            area = triangle_area(points_np[tri[0]], points_np[tri[1]], points_np[tri[2]])
            avg_energy = torch.mean(Res[tri[0]] + Res[tri[1]] + Res[tri[2]]) / 3
            energy = avg_energy * area + energy

        loss = energy
        if step % 100 == 0 and step != 0:

            if step in fig_step:
                plot_3D(model, params, step, 0)
                print(loss)
        loss = loss
        loss.backward()
        optimizer.step()
        scheduler.step()



def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["d"] = 2  # 2D
    params["interval"] = args.inter
    params["nx"] = args.nx
    params["ny"] = args.nx
    params["n_t"] = 400
    params["width"] = args.n  # Width of layers
    params["depth"] = args.d  # Hidden Layer: depth+10
    params["dd"] = 2  # Output
    params["lr"] = 0.0001  # Learning rate  0.005
    params["beta"] = args.beta
    params["xi"] = args.xi
    params["trainstep"] = args.e
    params["pre_trainstep"] = 2000
    params["Writestep"] = 100
    params["pre_step"] = 100
    params["minimal"] = 10 ** (-14)

    params["step_size"] = 100  # lr decay
    params["gamma"] = 0.95  # lr decay rate  0.98
    params["omega"] = args.omega
    startTime = time.time()

    model1 = Net(params, device).to(device)
    # model1 = torch.load('大区域模型/2d_1.pkl', weights_only=False)
    model1 = torch.load('大区域模型/0.9.pkl', weights_only=False)

    print("Generating network costs %s seconds." % (time.time() - startTime))
    print(params)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=params["lr"])
    # optimizer2 = torch.optim.Adam(model2.parameters(), lr=params["lr"])
    scheduler1 = StepLR(optimizer1, step_size=params["step_size"], gamma=params["gamma"])
    # scheduler2 = StepLR(optimizer2, step_size=params["step_size"], gamma=params["gamma"])
    a, b, d = G(model1, params, device)
    interp_func = RectBivariateSpline(a, b, d, kx=3, ky=3)
    points_np, cells = generate_adaptive_mesh(interp_func)
    print(cells.shape, points_np.shape)
    compute_integrals(model1, points_np, cells, params, 501, optimizer1, scheduler1)

    startTime = time.time()
    # loss1 = train1(model1, device, params, optimizer1, scheduler1, startTime, 1, 1001)

    torch.save(model1, '模型/2d_1.pkl')
    # plot_3D(model1, params, 0, 0)



if __name__ == "__main__":
    main()
