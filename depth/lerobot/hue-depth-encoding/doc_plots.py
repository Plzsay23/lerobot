import numpy as np
import matplotlib.pyplot as plt

from huecodec import codec as hc


SIZE_DEFAULT = 8
SIZE_LARGE = 10
plt.rc("font", family="Roboto")  # controls default font
plt.rc("font", weight="normal")  # controls default font
plt.rc("font", size=SIZE_DEFAULT)  # controls default text sizes
plt.rc("axes", titlesize=SIZE_LARGE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE_DEFAULT)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE_DEFAULT)  # fontsize of the tick labels
plt.rc("figure", titlesize=SIZE_LARGE)  # fontsize of the tick labels


def plot_linear_vs_disparity():

    near = 0.1
    far = 1.1

    depth = np.linspace(near, far, 200)

    lin = np.clip((depth - near) / (far - near), 0.0, 1.0)
    disp = np.clip((1 / depth - 1 / near) / (1 / far - 1 / near), 0.0, 1.0)

    with hc.enc_opts(hc.EncoderOpts(max_hue=300)):

        e_lin = hc.quantize(hc.encode(lin))
        e_disp = hc.quantize(hc.encode(disp))

        d_lin = hc.decode(hc.dequantize(e_lin))  # [0..1]
        d_disp = hc.decode(hc.dequantize(e_disp))

        d_lin = d_lin * (far - near) + near
        d_disp = d_disp * (1 / far - 1 / near) + 1 / near
        d_disp = 1 / d_disp

    # print(d_disp)

    fig = plt.figure(figsize=(10, 6), layout="constrained")
    fig.suptitle(
        f"Encoder/Decoder comparison for linear/disparity variants.\nnear {near} / far {far}"
    )
    gs = fig.add_gridspec(3, 2, height_ratios=(3, 1, 1))

    ax = fig.add_subplot(gs[0, 0])
    # Transformed values
    ax.plot(depth, disp, label="disparity")
    ax.scatter(depth[::5], disp[::5], s=4)
    ax.plot(depth, lin, label="linear")
    ax.scatter(depth[::5], lin[::5], s=4)
    ax.set_xlim(depth.min(), depth.max())
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("depth")
    ax.set_ylabel("normalized depth")
    ax.set_title("Depth normalization")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[0, 1])
    # Transformed values

    ax.plot(depth, abs(d_disp - depth), label="disparity")
    ax.plot(depth, abs(d_lin - depth), label="linear")
    ax.set_xlim(depth.min(), depth.max())
    ax.set_ylim(1e-7, 0.1)
    ax.set_xlabel("depth")
    ax.set_ylabel("absolute depth error")
    ax.set_title("Encoding/Decoding error")
    ax.set_yscale("log")
    ax.grid()
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[1, :])
    ax.set_title("Hue encoding: linear")
    ax.imshow(e_lin.reshape(1, -1, 3), extent=(depth.min(), depth.max(), 0, 1))
    ax.set_aspect("auto")

    ax = fig.add_subplot(gs[2, :])
    ax.set_title("Hue encoding: disparity")
    ax.imshow(e_disp.reshape(1, -1, 3), extent=(depth.min(), depth.max(), 0, 1))
    ax.set_aspect("auto")
    ax.set_xlabel("depth")

    fig.savefig("etc/compare_encoding.svg")
    fig.savefig("etc/compare_encoding.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    plot_linear_vs_disparity()
