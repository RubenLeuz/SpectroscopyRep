import tmm.tmm_core as tm
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from pathlib import Path

# test

BASE_DIR = Path(__file__).resolve().parent
PLOT_DIR = BASE_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Load optical constants from AuTi file
def load_optical_constants(filename):
    data = np.loadtxt(filename, comments='#')
    wl = data[:, 0] * 1e9  # Convert wavelength from m to nm
    n_Au = data[:, 1]
    k_Au = data[:, 2]
    n_Ti = data[:, 3]
    k_Ti = data[:, 4]
    return wl, n_Au, k_Au, n_Ti, k_Ti

# Load the data
optical_constants_path = BASE_DIR / "lab" / "Optical_constant_Au_Ti.txt"
wl_data, n_Au, k_Au, n_Ti, k_Ti = load_optical_constants(optical_constants_path)

# Create interpolation functions
n_Au_interp = interp1d(wl_data, n_Au, kind='linear', bounds_error=False, fill_value='extrapolate')
k_Au_interp = interp1d(wl_data, k_Au, kind='linear', bounds_error=False, fill_value='extrapolate')
n_Ti_interp = interp1d(wl_data, n_Ti, kind='linear', bounds_error=False, fill_value='extrapolate')
k_Ti_interp = interp1d(wl_data, k_Ti, kind='linear', bounds_error=False, fill_value='extrapolate')

# Function to get complex refractive index
def get_refractive_index(material, wavelength_nm):
    if material == 'Au':
        n = n_Au_interp(wavelength_nm)
        k = k_Au_interp(wavelength_nm)
    elif material == 'Ti':
        n = n_Ti_interp(wavelength_nm)
        k = k_Ti_interp(wavelength_nm)
    else:
        raise ValueError("Material must be 'Au' or 'Ti'")
    return n + 1j * k

# Example usage:
# refr_Au = get_refractive_index('Au', 500)  # For gold at 500 nm

# Calculate transmission T for the etalon
# Mirrors: 2 nm Ti + 15 nm Au on glass, etalon gap 15 μm
# Structure: air | Ti | Au | air (gap) | Au | Ti | air
# Assuming glass is not included as it's thick, and gap is air

# Wavelength in nm
wl = 500

# Refractive indices (complex)
n_air = 1 + 0j
n_Ti = get_refractive_index('Ti', wl)
n_Au = get_refractive_index('Au', wl)

refr = [n_air, n_Ti, n_Au, n_air, n_Au, n_Ti, n_air]

# Thicknesses in nm (inf for semi-infinite)
layers = [np.inf, 2, 15, 15000, 15, 2, np.inf]

# Angle of incidence in degrees
theta = 0

# Calculate at single wavelength
data = tm.coh_tmm("s", refr, layers, theta, wl)
T = data["T"]

print(f"Transmission T at {wl} nm: {T}")

# Now calculate T over a range of wavelengths and plot
print("\nCalculating T over wavelength range...")

# Wavelength range in nm
wl_range = np.linspace(400, 700, 150)
T_range = []

for wl_temp in wl_range:
    n_Ti_temp = get_refractive_index('Ti', wl_temp)
    n_Au_temp = get_refractive_index('Au', wl_temp)
    
    refr_temp = [n_air, n_Ti_temp, n_Au_temp, n_air, n_Au_temp, n_Ti_temp, n_air]
    
    data_temp = tm.coh_tmm("s", refr_temp, layers, theta, wl_temp)
    T_range.append(data_temp["T"])

T_range = np.array(T_range)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(wl_range, T_range, 'b-', linewidth=2)
plt.xlabel('Wavelength (nm)', fontsize=12)
plt.ylabel('Transmission Coefficient T', fontsize=12)
plt.title('Etalon Transmission vs Wavelength\n(2 nm Ti + 15 nm Au mirrors, 15 μm air gap)', fontsize=13)
plt.grid(True, alpha=0.3)
plt.xlim(wl_range[0], wl_range[-1])
plt.ylim(0, max(T_range) * 1.1)
plt.tight_layout()
plot_path = PLOT_DIR / "etalon_transmission.png"
plt.savefig(plot_path, dpi=150)
plt.show()

print(f"Plot saved as '{plot_path}'")
