from torch.optim.lr_scheduler import StepLR
from torch.autograd import grad
import math, torch, time, os
import scipy.io as io
import torch.nn as nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
import random
from scipy.linalg import orthogonal_procrustes

random.seed(111)
np.random.seed(111)
torch.manual_seed(111)
torch.cuda.manual_seed(111)
torch.cuda.manual_seed_all(111)

fig_step = [0, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]

parser = argparse.ArgumentParser(description='hyper parameters')
parser.add_argument('--d', type=int, default=4, help='depth')
parser.add_argument('--n', type=int, default=64, help='width')
parser.add_argument('--beta', type=int, default=100, help='/beta')
parser.add_argument('--inter', type=int, default=6, help='the interval is [-inter, inter]')
parser.add_argument('--nx', type=int, default=64, help='Sampling')
parser.add_argument('--omega', type=float, default=0.9, help='omega')
o_min = 0.89
o_max = 0.9
num1 = 3


args = parser.parse_args()



def GetGradients(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True, allow_unused=True)[0]


def V(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return 0.5 * (x ** 2 + y ** 2)



def plot_3D(model, params, step, num):
    plt.rcParams.update({
        "text.usetex": True,      # 启用 LaTeX 渲染
        "font.family": "serif",   # 使用衬线字体（LaTeX 默认）
        "font.serif": ['Computer Modern'], # 使用类似 LaTeX 的字体
        'font.size': 20,          # 默认全局字体大小
        'axes.titlesize': 20,     # 标题默认大小
        'axes.labelsize': 20,     # 坐标轴标签默认大小
        'xtick.labelsize': 20,    # X轴刻度字号
        'ytick.labelsize': 20,    # Y轴刻度字号
        'legend.fontsize': 20,    # 图例字号
    })

    x = np.linspace(-6, 6, 400)
    y = np.linspace(-6, 6, 400)
    z1 = (o_min + o_max) / 2
    z = np.linspace(z1, z1, 1)
    [X, Y, Z] = np.meshgrid(x, y, z)
    X1, Y1 = np.meshgrid(x, y)
    X_test = np.concatenate([X.flatten()[:, None], Y.flatten()[:, None], Z.flatten()[:, None]], axis=1)
    x_test = torch.from_numpy(X_test).float()
    
    # 确保模型在评估模式
    model.eval()
    with torch.no_grad():
        norm = torch.sum(model(x_test) * model(x_test)) * (2 * params["interval"]) ** 2 / (400*400)
        point1 = ((model(x_test)[:, 0] ** 2 + model(x_test)[:, 1] ** 2) / norm).to("cpu").numpy().reshape(400, 400)

    # --- 代码修改部分 ---
    # 1. 创建 Figure 和 Axes 对象
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # 2. 使用 ax 对象绘图和设置
    ax.set_aspect('equal', adjustable='box')
    
    c0 = ax.pcolor(X1, Y1, point1, cmap='gist_rainbow')
    
    # 定义刻度位置和标签
    tick_positions = [-6, 0, 6]
    tick_labels = [r'\textbf{-6}', r'\textbf{0}', r'\textbf{6}']
    
    # 3. 使用 ax.set_xticks 和 ax.set_xticklabels 设置刻度
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    # 添加 colorbar
    cbar = fig.colorbar(c0, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    
    # 4. 使用 LaTeX 命令加粗坐标轴标签
    ax.set_xlabel('$\\textbf{x}$', fontsize=35, labelpad=-5)
    ax.set_ylabel('$\\textbf{y}$', fontsize=35, labelpad=-5)
    

    plt.savefig('D:/用户/研究生文章/BSE/实验/实验结果/' + f'exam_{num}_{step}.png') # 建议添加.png后缀
    plt.close()




class Net(torch.nn.Module):  # 训练一个旋转神经网络，用来逼近旋转项
    def __init__(self, params, device):
        super(Net, self).__init__()
        self.params = params
        self.device = device
        self.linearIn = nn.Linear(self.params["d"] + 2, self.params["width"])
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


distance_ = []


def train1(model, device, params, optimizer, scheduler, pre_time, num, trainstep):
    x = np.linspace(-1 * params["interval"], params["interval"], params["nx"])
    y = np.linspace(-1 * params["interval"], params["interval"], params["ny"])
    z = np.linspace(o_min, o_max, num1)
    [X, Z, Y] = np.meshgrid(x, z, y)

    x_train = np.concatenate([X.flatten()[:, None], Y.flatten()[:, None], Z.flatten()[:, None]], axis=1)
    X_train = torch.from_numpy(x_train).float().to(device)
    X_train = X_train.requires_grad_(True)
    x1 = X_train[:, 0]
    y1 = X_train[:, 1]
    z1 = X_train[:, 2]
   

    start_time = time.time()
    total_start_time = start_time
    Loss = []
    Step = []
    Energy = []
    Time = []


    for step in range(trainstep):
        U_pred = model(X_train).to(device)

        U_x_real = GetGradients(U_pred[:, 0], X_train)[:, 0].squeeze_()
        U_x_image = GetGradients(U_pred[:, 1], X_train)[:, 0].squeeze_()
        # U_xx_real = GetGradients(U_x_real, X_train)[:, 0].squeeze_()
        # U_xx_image = GetGradients(U_x_image, X_train)[:, 0].squeeze_()

        U_y_real = GetGradients(U_pred[:, 0], X_train)[:, 1].squeeze_()
        U_y_image = GetGradients(U_pred[:, 1], X_train)[:, 1].squeeze_()
        # U_yy_real = GetGradients(U_y_real, X_train)[:, 1].squeeze_()
        # U_yy_image = GetGradients(U_y_image, X_train)[:, 1].squeeze_()
        model.zero_grad()
        Res = 0
        for i in range(num1):
            batch = params["nx"] * params["ny"]
            U_pred1 = U_pred[i * batch: (i + 1) * batch]
            U_x_real1 = U_x_real[i * batch: (i + 1) * batch]
            U_x_image1 = U_x_image[i * batch: (i + 1) * batch]
            U_y_real1 = U_y_real[i * batch: (i + 1) * batch]
            U_y_image1 = U_y_image[i * batch: (i + 1) * batch]
            X_train1 = X_train[i * batch: (i + 1) * batch, :]
            x11 = x1[i * batch: (i + 1) * batch]
            y11 = y1[i * batch: (i + 1) * batch]
            z11 = z1[i * batch: (i + 1) * batch]
           
            norm = torch.sqrt(
                torch.sum(U_pred1 ** 2) * (2 * params["interval"]) ** 2 / (params["nx"] * params["ny"]))
            if step <= 3001:
                res3 = (z11 ** 2 + 1) / 2 * (
                        -U_pred1[:, 0] * (U_y_image1 * x11 - y11 * U_x_image1) + U_pred1[:, 1] * (
                        x11 * U_y_real1 - y11 * U_x_real1)) / norm ** 2
            else:
                res3 = z11 * (-U_pred1[:, 0] * (U_y_image1 * x11 - y11 * U_x_image1) + U_pred1[:, 1] * (
                        x11 * U_y_real1 - y11 * U_x_real1)) / norm ** 2

            res1 = 0.5 * abs(U_x_real1 ** 2 + U_x_image1 ** 2 + U_y_real1 ** 2 + U_y_image1 ** 2) / norm ** 2

            # res2 = V(X_train1).squeeze_() * (U_pred1[:, 0] ** 2 + U_pred1[:, 1] ** 2) / (norm ** 2)

            res2 = V(X_train1).squeeze_() * (U_pred1[:, 0] ** 2 + U_pred1[:, 1] ** 2) / (norm ** 2)

            # res3 = params["omega"] * (-U_pred[:, 0] * (U_y_image * x1 - y1 * U_x_image) + U_pred[:, 1] * (
            #         x1 * U_y_real - y1 * U_x_real)) / norm ** 2
            res4 = params["beta"] * (U_pred1[:, 0] ** 2 + U_pred1[:, 1] ** 2) ** 2 / (2 * norm ** 4)

            Res = res1 + res2 - res3 + res4 + Res
            loss_res = torch.sum((2 * params["interval"]) ** 2 * res3 / (params["nx"] * params["ny"]))
            l2 = torch.sum((2 * params["interval"]) ** 2 * res1 / (params["nx"] * params["ny"]))
            l3 = torch.sum((2 * params["interval"]) ** 2 * res2 / (params["nx"] * params["ny"]))
            l4 = torch.sum((2 * params["interval"]) ** 2 * res4 / (params["nx"] * params["ny"]))

        loss = torch.sum((2 * params["interval"]) ** 2 * Res / (params["nx"] * params["ny"] * num1))

        if step % 100 == 0 and step != 0:

            if step in fig_step:
                plot_3D(model, params, step, num)
           
          
            if step == trainstep - 100:
                total_time = time.time() - total_start_time + pre_time
                print('%% U no longer adapts, training stop')
                print('--------stop_step: %d' % step)
                print('--------final energy: %.3e' % loss_res)
                print("Training costs %s seconds." % (total_time))

                Step.append(step)
                Energy.append(loss_res.cpu().detach().numpy())
                Time.append(total_time)
                # plot_3D(model, params, step, num)
                break

        params["nx"] = params["n_t"]
        params["ny"] = params["n_t"]
     
        params["nx"] = args.nx
        params["ny"] = args.nx

        if step % params["Writestep"] == 0:
            elapsed = time.time() - start_time
            print('Epoch: %d, Time: %.2f, Loss: %.3e, l1: %.3e, l2: %.3e, l3: %.3e, l4: %.3e' %
                  (step, elapsed, loss, loss_res, l2, l3, l4))
            
            start_time = time.time()
            Loss.append(loss.cpu().detach().numpy())
        loss.backward()
        optimizer.step()
        scheduler.step()
   
    return min(Loss[-10:])


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
    params["lr"] = 0.005  # Learning rate  0.005
    params["beta"] = args.beta
    params["Writestep"] = 100
    params["step_size"] = 100  # lr decay
    params["gamma"] = 0.95  # lr decay rate  0.95
    params["omega"] = args.omega
    startTime = time.time()

    model1 = Net(params, device).to(device)
    model1 = torch.load('模型/2d_1涡旋.pkl', weights_only=False)

    print("Generating network costs %s seconds." % (time.time() - startTime))
    print(params)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=params["lr"])

    scheduler1 = StepLR(optimizer1, step_size=params["step_size"], gamma=params["gamma"])


    startTime = time.time()


    train1(model1, device, params, optimizer1, scheduler1, startTime, 1, 5001)

    # torch.save(model1, '0.9_more.pkl')
    plot_3D(model1, params, 0, 0)
    


if __name__ == "__main__":
    main()
