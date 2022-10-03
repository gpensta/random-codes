from matplotlib.widgets import EllipseSelector
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    print(ellipse.get_angle())
    return ax.add_patch(ellipse)

def f(x, u):
    x1, x2 , x3 = x
    return np.array([np.cos(x3), np.sin(x3), u])

def compute_box(x):
    x_min = np.min(x[0, :])
    x_max = np.max(x[0, :])
    y_min = np.min(x[1, :])
    y_max = np.max(x[1, :])
    return x_min, y_min, x_max - x_min, y_max - y_min

def main():
    X0 = np.random.rand(3, 100)
    X0[2, :] = np.zeros(100)
    dt = 0.5
    fig, ax = plt.subplots()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    for i in range(20):
        u = 2 * np.random.rand(X0.shape[1]) - 1.
        # u = dt *2 
        for j in range(X0.shape[1]):
            X0[:, j] = X0[:, j] + dt * f(X0[:, j], u[j])
        x_min, y_min, dx, dy = compute_box(X0)
        # plt.scatter(X0[0, :], X0[1, :], s=0.1)
        plt.scatter(X0[0, 1], X0[1, 1], s=1., c='b')
        confidence_ellipse(X0[0, :], X0[1, :], ax, facecolor='none', edgecolor='blue')
        # ax.add_patch(Rectangle((x_min, y_min), dx, dy, facecolor='none', edgecolor='red'))
        plt.savefig("plot.png")
        plt.pause(0.05)

if __name__ == '__main__':
    main()