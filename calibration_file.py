import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def fit_and_plot_linear(
    x,
    y,
    SAVE,
    *,
    title=None,
    xlabel="x",
    ylabel="y",
    xerr=None,
    yerr=None,
    show=True,
):
    """Fit a straight line y = m*x + b to the data and plot data + fit."""
    x = np.asarray(x)
    y = np.asarray(y)
    if xerr is not None:
        xerr = np.asarray(xerr)
    if yerr is not None:
        yerr = np.asarray(yerr)

    coeffs = np.polyfit(x, y, deg=1)
    slope, intercept = coeffs
    print("Slope:", slope , "Intercept:", intercept)
    x_fit = np.linspace(x.min(), x.max(), 200)
    y_fit = slope * x_fit + intercept
    plt.figure()
    if xerr is not None or yerr is not None:
        plt.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color="tab:blue",
            ecolor="tab:blue",
            elinewidth=1,
            capsize=3,
            label="Data",
        )
    else:
        plt.scatter(x, y, label="Data", color="tab:blue")
    plt.plot(x_fit, y_fit, label=f"Linear Fit: y = {slope:.3g}x + {intercept:.3g}", color="tab:orange")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()

    save_path = Path(SAVE)
    if not save_path.is_absolute():
        save_path = PLOT_DIR / save_path
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    return slope, intercept


def plot_horizontal_intensity(image_path, *, show=True, save_path=None):
    """
    Plot the mean intensity along each horizontal pixel position of a PNG image.

    Parameters
    ----------
    image_path : str or Path
        Path to the PNG image file.
    show : bool, optional
        Whether to display the plot interactively. Defaults to True.
    save_path : str or Path, optional
        If provided, save the generated plot to this path.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Arrays with horizontal pixel positions and corresponding mean intensities.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = plt.imread(image_path)
    image = image.astype(np.float64, copy=False)

    if image.ndim == 3:
        # Drop alpha channel if present before converting to grayscale.
        if image.shape[2] == 4:
            image = image[..., :3]
        # Use standard luminance weights to convert RGB to grayscale intensity.
        image = (
            0.299 * image[..., 0]
            + 0.587 * image[..., 1]
            + 0.114 * image[..., 2]
        )
    elif image.ndim != 2:
        raise ValueError(
            f"Unsupported image shape {image.shape}; expecting 2D or 3D array."
        )

    intensity_profile = image.mean(axis=0)
    horizontal_positions = np.arange(intensity_profile.size)

    fig, ax = plt.subplots()
    ax.plot(horizontal_positions, intensity_profile, color="tab:purple")
    ax.set_xlabel("Horizontal Pixel Position")
    ax.set_ylabel("Mean Intensity")
    ax.set_title(f"Horizontal Intensity Profile: {image_path.name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return horizontal_positions, intensity_profile

# Center the strong lines
#wavelengths = [587.5, 667.8, 706.5, 778 ,504.7, 492.1, 447.1]
#k_values =[15.82, 16.84, 17.32, 18.24, 14.76, 14.62, 14.04]
# Orange line
#pixel_positions = [720, 1080, 1440, 360, 0]
#k_value_pixels = [15.82, 15.94, 16.10, 15.66, 15.56]

#g=2*np.cos(np.pi/12)*5.8*10**(-5)*(15.82-8.36)
#print("g: ",g)


#fit_and_plot_linear(k_values, wavelengths,"wavevsk.png", xerr=0.01, title="Wavelength and k", xlabel="k", ylabel="Wavelength (nm)")
#fit_and_plot_linear(k_value_pixels, pixel_positions, "pixelvsk.png",xerr=0.01, title="Pixel Position and k", ylabel="Pixel Position",xlabel="k")


#Rydberg 
#wavelengths_ryd = [485.6, 656.1]
#n_ryd = [4, 3]
#fit_and_plot_linear([(1/4-1/n**2) for n in n_ryd], [1/(wavelength*1e-9) for wavelength in wavelengths_ryd], "rydberg.png", title="Rydberg Plot", xlabel="1/n^2", ylabel="1/Wavelength (1/m)")



##Calibration for P25
#Helium calibration
wavelengths = [587.5, 667.8, 706.5, 778 ,504.7, 501.5, 492.1, 471.3, 447.1, 388.8]
k_values =[3.34, 2.28, 1.76, 0.86, 4.42, 4.46, 4.56, 4.80, 5.10, 5.82]
fit_and_plot_linear(k_values, wavelengths,"wavevsk.png", xerr=0.01, title="Wavelength and k", xlabel="k", ylabel="Wavelength (nm)")
#print(994/77.2)
#print(-77.2*(5.26-12.88))

# Orange line
pixel_positions = [720, 1080, 1440, 360, 0]
k_value_pixels = [3.34, 3.46, 3.58, 3.20, 3.06]
#k_value_pixels = [3.34,3.20 , 3.06, 3.46, 3.58]
fit_and_plot_linear(k_value_pixels, pixel_positions, "pixelvsk.png",xerr=0.01, title="Pixel Position and k", ylabel="Pixel Position",xlabel="k")

kvalues_und = [7.82, 7.58, 7.22,6.50,  5.80, 5.40, 5.38, 4.38]

def calc_lambda(k):
    return -77.2*(k-12.88)

lambda_und = []

for i in range (len(kvalues_und)):
    lambda_und.append(calc_lambda(kvalues_und[i]))

print(lambda_und)
