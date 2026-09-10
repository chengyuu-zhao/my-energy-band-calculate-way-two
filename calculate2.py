import numpy as np
import matplotlib.pyplot as plt

def get_fermi_from_outcar(outcar_name="OUTCAR"):
    try:
        with open(outcar_name, "r") as f:
            for line in f:
                if "E-fermi" in line:
                    ef = float(line.split()[2])
                    print(f"✅ 从OUTCAR读取费米能级 Ef = {ef:.4f} eV")
                    return ef
    except FileNotFoundError:
        print("⚠️ 未找到OUTCAR，不做费米能级平移")
    return None


def read_eigenval_soc(filename="EIGENVAL"):
    """读取开启SOC的VASP EIGENVAL（单能量列，无分开自旋上下）"""
    f = open(filename, 'r')
    # 跳过前面所有注释头，直到读到 NELECT NKPTS NBANDS 这一行
    while True:
        line = f.readline()
        if not line:
            raise Exception("EIGENVAL文件提前结束")
        parts = line.strip().split()
        if len(parts) == 3:
            try:
                nelect = int(parts[0])
                nkpts = int(parts[1])
                nbands = int(parts[2])
                break
            except ValueError:
                continue
    print(f"✅ EIGENVAL读取成功（SOC）：NELECT={nelect}, NKPTS={nkpts}, NBANDS={nbands}")

    kx = np.zeros(nkpts)
    ky = np.zeros(nkpts)
    kz = np.zeros(nkpts)
    eigen = np.zeros((nkpts, nbands))  # SOC只有一套能级

    for ik in range(nkpts):
        # 读取k点坐标+权重
        while True:
            line = f.readline()
            if not line:
                raise Exception("读取k点时文件提前结束")
            parts = line.strip().split()
            if len(parts)>=4:
                try:
                    kx[ik] = float(parts[0])
                    ky[ik] = float(parts[1])
                    kz[ik] = float(parts[2])
                    break
                except ValueError:
                    continue
        # 读取nbands条能级 SOC格式：iband, energy, occ
        for ib in range(nbands):
            while True:
                line = f.readline()
                if not line:
                    raise Exception(f"kpoint {ik}, band {ib}: 文件提前结束")
                parts = line.strip().split()
                if len(parts)>=2:
                    try:
                        eigen[ik, ib] = float(parts[1])
                        break
                    except ValueError:
                        continue
    f.close()

    # 计算k路径累积距离
    k_dist = np.zeros(nkpts)
    for i in range(1, nkpts):
        dk = np.sqrt((kx[i]-kx[i-1])**2 + (ky[i]-ky[i-1])**2 + (kz[i]-kz[i-1])**2)
        k_dist[i] = k_dist[i-1] + dk
    return k_dist, eigen, nkpts, nbands


def plot_band_soc(k_dist, eigen, efermi=None, ylim=None):
    plt.figure(figsize=(9,6))
    nbands = eigen.shape[1]
    for ib in range(nbands):
        if efermi is not None:
            plt.plot(k_dist, eigen[:,ib] - efermi, c="#2266bb", lw=1.0)
        else:
            plt.plot(k_dist, eigen[:,ib], c="#2266bb", lw=1.0)

    if efermi is not None:
        plt.axhline(y=0, c="black", ls="--", lw=1.5, label="$E_F$")
        plt.ylabel("$E-E_F$ (eV)")
    else:
        plt.ylabel("$E$ (eV)")
    plt.xlabel("K-path")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    if ylim:
        plt.ylim(ylim)
    plt.tight_layout()
    plt.savefig("band_soc.png", dpi=300)
    print("✅ SOC能带图保存为 band_soc.png")
    plt.show()


if __name__ == "__main__":
    ef = get_fermi_from_outcar("OUTCAR")
    k_dist, eigen, nkpts, nbands = read_eigenval_soc("EIGENVAL")
    print(f"k点数量={nkpts}, 每个k点能带数={nbands}")
    y_range = [-1, 1]
    plot_band_soc(k_dist, eigen, efermi=ef, ylim=y_range)
