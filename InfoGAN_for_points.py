import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import visdom
import torch
from torch import nn, optim, autograd  # 优化器、自动求导
import itertools

# Load Points4
data = pd.read_csv('./data/df_highdim_points.csv', index_col='name')
data_name = pd.read_csv('./data/df_name_points.csv', index_col='name')

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

    # point_name = data_name.iloc[i*batchsz: (i+1)*batchsz].values
    # dataset_name = point_name.astype(np.float32)

    return dataset  # , dataset_name


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

    pred, _ = D(mid)
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
        z_noise = torch.randn(batchsz, 2)  # 生成指定大小的正态分布的噪声
        z_ma = torch.randint(6, 19, (batchsz, 1))  # ma的范围是[6,18]， np.randint生成的是左闭右开的区间
        z_a = torch.randint(-4, 17, (batchsz, 1))  # a的范围是[-4,16]
        z_b = torch.randint(0, 6, (batchsz, 1))  # b的范围是[0,6]
        z = torch.cat((z_noise, z_ma, z_a, z_b), dim=1).to(device) # [b, 2+3]
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
            nn.Linear(5, h_dim),  # 这里的2是可以修改的，是随机噪声的维度(2)+输入3维(ma/a/b)，根据任务修改
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
            # 主网络，也是D和Q共用网络的部分一共有3层
            nn.Linear(out_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True)
        )
        self.D_net = nn.Sequential(
            # 判别器，判别生成数据是否真实的
            nn.Linear(h_dim, 1),
            # sigmoid：映射到属于真实分布的概率
            nn.Sigmoid()
        )
        self.Q_net = nn.Sequential(
            # Info的部分
            nn.Linear(h_dim, 3)  # 这里的3是对应[ma, alpha, beta]三个维度，不能修改
        )

    def forward(self, x):
        output = self.net(x)
        return self.D_net(output).view(-1), self.Q_net(output)  # torch中view函数的作用类似于reshape


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
    optim_Info = torch.optim.Adam(itertools.chain(G.parameters(), D.parameters()), lr=5e-5, betas=(0.5, 0.9))

    viz.line([[0, 0, 0]], [0], win='loss', opts=dict(title='loss', legend=['D', 'G', 'Info']))

    # 开始训练
    i = 0  # 用于给真实数据分批
    for epoch in range(50):  # 可以修改range(50000) 为了减小运算快速看到效果

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
            predr, _ = D(xr)
            # max predr, min lossr
            lossr = -predr.mean()

            # 1.2 train on fake data
            # [b, 2] 这里的2是输入的2
            z_noise = torch.randn(batchsz, 2)  # 生成指定大小的正态分布的噪声
            z_ma = torch.randint(6, 19, (batchsz, 1))  # ma的范围是[6,18]， np.randint生成的是左闭右开的区间
            z_a = torch.randint(-4, 17, (batchsz, 1))  # a的范围是[-4,16]
            z_b = torch.randint(0, 6, (batchsz, 1))  # b的范围是[0,6]
            z = torch.cat((z_noise, z_ma, z_a, z_b), dim=1).to(device)

            xf = G(z).detach()  # tf.stop_gradient()，不会再往前传梯度
            predf, _ = D(xf)
            lossf = predf.mean()

            # 1.3 gradient penalty
            # xf不需要对G求导，所以需要detach
            gp = gradient_penalty(D, xr, xf.detach())

            # aggregate all
            loss_D = lossr + lossf + 0.2 * gp  # lambda参数

            # optimize
            optim_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optim_D.step()

        # 2. train Generator
        z_noise = torch.randn(batchsz, 2)  # 生成指定大小的正态分布的噪声
        z_ma = torch.randint(6, 19, (batchsz, 1))  # ma的范围是[6,18]， np.randint生成的是左闭右开的区间
        z_a = torch.randint(-4, 17, (batchsz, 1))  # a的范围是[-4,16]
        z_b = torch.randint(0, 6, (batchsz, 1))  # b的范围是[0,6]
        z = torch.cat((z_noise, z_ma, z_a, z_b), dim=1).to(device)

        xf = G(z)
        predf, pred_continuous = D(xf)  # 这里不能加detach
        # max predf.mean()
        loss_G = -predf.mean()
        optim_G.zero_grad()

        # 3. train Info
        c_continuous = torch.cat((z_ma, z_a, z_b), dim=1).to(torch.float32).to(device)  # z_ma/z_a/z_b是int类型，转换成float32类型
        # ma/a/b的值太大了，除以（最大值-最小值）放缩
        lb = torch.tensor([6.0, -4.0, 0.0])
        ub = torch.tensor([18.0, 16.0, 6.0])
        ub_lb = ub - lb
        c_continuous = c_continuous / ub_lb.repeat(batchsz, 1)
        pred_continuous = pred_continuous / ub_lb.repeat(batchsz, 1)
        loss_Info = torch.nn.MSELoss(reduction='mean')(pred_continuous, c_continuous)
        optim_Info.zero_grad()

        # 2.和3.的optimize放在一起，否则超过1.4版本的pytorch会报错
        loss_G.backward(retain_graph=True)
        loss_Info.backward()

        optim_G.step()
        optim_Info.step()


        if epoch % 1 == 0:  # 每隔1轮，打印loss和生成信号的曲线
            viz.line([[loss_D.item(), loss_G.item(), loss_Info.item()]], [epoch], win='loss', update='append')
            print(epoch, loss_D.item(), loss_G.item(), loss_Info.item())
            generate_image(D, G, xr, epoch)


if __name__ == '__main__':
    main()

