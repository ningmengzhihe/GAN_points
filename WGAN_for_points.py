import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visdom
import torch
# 优化器、自动求导
from torch import nn, optim, autograd

# Load Points4
data = pd.read_csv('./data/df_highdim_points.csv', index_col='name')

h_dim = 800
out_dim = data.shape[1]  # 数据有1001个样本，用前1000个生成数据，
batchsz = 100  # 设计batchsz，使得n_sample/batchsz是个整数
viz = visdom.Visdom()  # 同时在terminal中切换到project目录下，执行python -m visdom.server
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def data_split(i: int):
    '''
    给1000个样本分批，从前到后的顺序分批，
    :param i: 在函数外面限制i的取值只能是[0, n_sample/batchsz = 1000/100 = 10), 左闭右开
    :return dataset: np.array
    '''
    point = data.iloc[i*batchsz: (i+1)*batchsz].values
    dataset = point.astype(np.float32)

    return dataset


def gradient_penalty(D, xr, xf):
    """
    :param D:
    :param xr:
    :param xf:
    :return:
    """
    # [b, 1]
    t = torch.rand(batchsz, 1).to(device)
    # [b, 1] => [b, 2] 对于同样的sample，是相同的
    t = t.expand_as(xr)
    # interpolation
    mid = t * xr + (1-t) * xf
    mid.requires_grad_()

    pred = D(mid)
    grads = autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred),
                          create_graph=True, retain_graph=True,
                          only_inputs=True)[0]  # create_graph: 二阶求导， retain_graph保留下来信息

    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()  # 二范数和1越接近越好

    return gp


def generate_image(D, G, xr, epoch):
    '''
    generate a batchsz samples and draw samples
    '''
    with torch.no_grad():
        z = torch.randn(batchsz, 2).to(device)  # [b, 2]
        samples = G(z).cpu().numpy()  # [b, 2]

    plt.figure(figsize=(10, 5))
    for item in samples:
        plt.plot(item)

    viz.matplot(plt, win='sample', opts=dict(title='Generative Samples: %d' % epoch))


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # z: [b, ] => [b, 2]
            # 一共有4层
            nn.Linear(2, h_dim),  # 这里的2是可以修改的，是随机噪声的维度，根据任务修改
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, out_dim)
        )

    def forward(self, z):
        output = self.net(z)
        return output


class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            # 一共有4层
            nn.Linear(out_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            # sigmoid：映射到属于真实分布的概率
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.net(x)
        return output.view(-1)  # 这个是什么？


def main():
    # 设置种子
    torch.manual_seed(23)
    np.random.seed(23)

    G = Generator().to(device)
    D = Discriminator().to(device)
    # print(G)
    # print(D)
    optim_G = optim.Adam(G.parameters(), lr=5e-5, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=5e-5, betas=(0.5, 0.9))

    viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss', legend=['D', 'G']))

    # 开始训练
    i = 0  # 用于给真实数据分批
    for epoch in range(20000):  # 50为了减小运算快速看到效果

        # 1. train Discriminator firstly
        for _ in range(10):
            # 1.1 train on real data
            if i >= int(1000 / batchsz):
                # 限制i的范围是[0, n_sample/batchsz = 1000/100 = 10), 左闭右开
                i = 0
            xr = data_split(i)  # xr = next(data_iter)
            i += 1
            xr = torch.from_numpy(xr).to(device)
            # [b, 2]  => [b, 1]
            predr = D(xr)
            # max predr, min lossr
            lossr = -predr.mean()

            # 1.2 train on fake data
            # [b, 2] 这里的2是输入的2
            z = torch.randn(batchsz, 2).to(device)  # 生成指定大小的正态分布
            xf = G(z).detach()  # tf.stop_gradient()，不会再往前传梯度
            predf = D(xf)
            lossf = predf.mean()

            # 1.3 gradient penalty
            # xf不需要对G求导，所以需要detach
            gp = gradient_penalty(D, xr, xf.detach())

            # aggregate all
            loss_D = lossr + lossf + 0.2 * gp  # lambda参数

            # optimize
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        # 2. train Generator
        z = torch.randn(batchsz, 2).to(device)
        xf = G(z)
        predf = D(xf)  # 这里不能加detach
        # max predf.mean()
        loss_G = -predf.mean()

        # optimize
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 10 == 0:  # 每隔10轮，打印loss和生成信号的曲线
            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')
            print(epoch, loss_D.item(), loss_G.item())
            generate_image(D, G, xr, epoch)


if __name__ == '__main__':
    main()

