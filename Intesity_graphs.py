from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image

# Calibration data (same values as in calibration_file.py).
WAVELENGTHS_NM = [587.5, 667.8, 706.5, 778.0, 504.7, 501.5, 492.1, 471.3, 447.1, 388.8]
K_VALUES = [3.34, 2.28, 1.76, 0.86, 4.42, 4.46, 4.56, 4.80, 5.10, 5.82]

PIXEL_POSITIONS = [720, 1080, 1440, 360, 0]
K_VALUES_PIXELS = [3.34, 3.46, 3.58, 3.20, 3.06]

CENTER_PIXEL = 720
CENTER_WAVELENGTH_NM = 600.0
ETALON_GAPS_UM = {
    "Etalon 15 um": 15.0,
    "Etalon 25 um": 25.0,
    "Etalon 50 um": 50.0,
    "Etalon 100 um": 100.0,
}
ANGLE_ETALON_GAP_UM = 15.0
DIY_ETALON_GAP_UM = 25.0


def linear_fit(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Least-squares fit: y = m*x + b."""
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    if len(x) < 2:
        raise ValueError("At least two points are required.")

    n = float(len(x))
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den = sum((xi - mean_x) ** 2 for xi in x)
    if den == 0:
        raise ValueError("Cannot fit line: all x values are equal.")
    m = num / den
    b = mean_y - m * mean_x
    return m, b


def calibration_nm_per_pixel() -> float:
    """
    Build wavelength(pixel) slope from two calibrations:
    wavelength(k) and pixel(k), then nm/pixel = (dw/dk)/(dp/dk).
    """
    dw_dk, _ = linear_fit(K_VALUES, WAVELENGTHS_NM)
    dp_dk, _ = linear_fit(K_VALUES_PIXELS, PIXEL_POSITIONS)
    return dw_dk / dp_dk


def pixel_to_wavelength(pixel: int, nm_per_pixel: float) -> float:
    """Anchor the calibration at 720 px -> 600 nm."""
    return CENTER_WAVELENGTH_NM + nm_per_pixel * (pixel - CENTER_PIXEL)


def theoretical_fsr_nm(gap_um: float, wavelength_nm: float = CENTER_WAVELENGTH_NM) -> float:
    """FSR from the lab script (Eq. 4): Delta lambda = lambda^2 / (2d + lambda)."""
    d_nm = gap_um * 1000.0
    return (wavelength_nm * wavelength_nm) / (2.0 * d_nm + wavelength_nm)


def theoretical_fsr_nm_with_angle(
    gap_um: float, angle_deg: float, wavelength_nm: float = CENTER_WAVELENGTH_NM
) -> float:
    """
    Angle-dependent FSR from resonance condition with finite incidence angle:
    Delta lambda = lambda^2 / (2 d cos(theta) + lambda)
    """
    d_nm = gap_um * 1000.0
    cos_theta = math.cos(math.radians(angle_deg))
    return (wavelength_nm * wavelength_nm) / (2.0 * d_nm * cos_theta + wavelength_nm)


def moving_average(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return [float(v) for v in values]
    half = window // 2
    n = len(values)
    out = [0.0] * n
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = sum(values[a:b]) / (b - a)
    return out


def _median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def estimate_experimental_fsr_from_dips(
    wavelengths_nm: Sequence[float], y_values: Sequence[float], fsr_guess_nm: float
) -> float:
    """
    Estimate FSR from dip-to-dip spacing.

    Method:
    1) detect strong resonances (peaks) in a smoothed trace
    2) find dip (minimum) between adjacent resonances
    3) average dip-to-dip spacing with outlier rejection
    """
    n = len(y_values)
    if n < 10:
        return float("nan")

    y_smooth = moving_average(y_values, window=15)
    trace_range = max(y_smooth) - min(y_smooth)
    if trace_range <= 0:
        return float("nan")

    lo = int(0.06 * n)
    hi = int(0.94 * n)
    threshold = min(y_smooth) + 0.45 * trace_range
    min_sep_nm = max(0.45 * fsr_guess_nm, 0.8)

    peak_candidates: List[int] = []
    for i in range(max(lo, 1), min(hi, n - 1)):
        if y_smooth[i] >= y_smooth[i - 1] and y_smooth[i] > y_smooth[i + 1]:
            if y_smooth[i] >= threshold:
                peak_candidates.append(i)

    peak_candidates.sort(key=lambda idx: y_smooth[idx], reverse=True)
    peaks: List[int] = []
    for idx in peak_candidates:
        if all(abs(wavelengths_nm[idx] - wavelengths_nm[j]) >= min_sep_nm for j in peaks):
            peaks.append(idx)
    peaks.sort(key=lambda idx: wavelengths_nm[idx])

    if len(peaks) < 3:
        return float("nan")

    dip_indices: List[int] = []
    for left, right in zip(peaks[:-1], peaks[1:]):
        segment = y_smooth[left : right + 1]
        if not segment:
            continue
        local_min_offset = min(range(len(segment)), key=segment.__getitem__)
        dip_indices.append(left + local_min_offset)

    if len(dip_indices) < 2:
        return float("nan")

    dip_wavelengths = [wavelengths_nm[i] for i in dip_indices]
    spacings = [
        dip_wavelengths[i + 1] - dip_wavelengths[i] for i in range(len(dip_wavelengths) - 1)
    ]
    if not spacings:
        return float("nan")

    spacing_median = _median(spacings)
    if not math.isfinite(spacing_median):
        return float("nan")

    filtered = [s for s in spacings if 0.65 * spacing_median <= s <= 1.35 * spacing_median]
    if not filtered:
        filtered = spacings

    fsr_raw = sum(filtered) / len(filtered)
    if not math.isfinite(fsr_raw):
        return float("nan")

    # Harmonic correction:
    # If dips from alternating families are detected, spacing can be a sub-harmonic
    # (e.g. ~FSR/2). Pick the harmonic closest to the theoretical guess.
    harmonic_candidates = [fsr_raw * h for h in (1, 2, 3)]
    return min(harmonic_candidates, key=lambda value: abs(value - fsr_guess_nm))


def find_etalon_capture_files(lab_dir: Path) -> Dict[str, Path]:
    """Read BMP captures for the 4 etalon sizes (15/25/50/100 um)."""
    files = {
        "Etalon 15 um": lab_dir / "etalon_15um" / "etalon_15um_capture.bmp",
        "Etalon 25 um": lab_dir / "etalon_25um" / "etalon_25um_capture.bmp",
        "Etalon 50 um": lab_dir / "etalon_50um" / "etalon_50um_capture.bmp",
        "Etalon 100 um": lab_dir / "etalon_100um" / "etalon_100um_capture.bmp",
    }
    for label, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing BMP for {label}: {path}")
    return files


def _format_angle_label(angle_deg: float) -> str:
    angle_int = int(round(angle_deg))
    if angle_int > 0:
        return f"Angle +{angle_int} deg"
    if angle_int < 0:
        return f"Angle {angle_int} deg"
    return "Angle 0 deg"


def find_angle_capture_files(angle_dir: Path) -> List[Tuple[str, float, Path]]:
    """
    Read angle-dependent captures for the 15 um etalon.

    Returns a list of (label, angle_deg, path) sorted by angle.
    """
    pattern = re.compile(r"etalon_15um_angle_(minus_|plus_)?(\d+)deg\.bmp$")
    captures: List[Tuple[str, float, Path]] = []

    for path in sorted(angle_dir.glob("etalon_15um_angle_*deg.bmp")):
        match = pattern.match(path.name)
        if not match:
            continue
        sign_token, deg_text = match.groups()
        angle_deg = float(deg_text)
        if sign_token == "minus_":
            angle_deg = -angle_deg
        label = _format_angle_label(angle_deg)
        captures.append((label, angle_deg, path))

    if not captures:
        raise FileNotFoundError(f"No angle BMP files found in {angle_dir}")

    captures.sort(key=lambda item: item[1])
    return captures


def find_diy_capture_file(lab_dir: Path) -> Tuple[str, Path]:
    """
    Read DIY etalon BMP capture.

    The current filename indicates a 25 um DIY etalon.
    """
    preferred = lab_dir / "diy_etalon" / "diy_25um_angle_unknown_capture.bmp"
    if preferred.exists():
        return "DIY Etalon 25 um", preferred

    # Fallback for older layouts.
    candidates = sorted(lab_dir.glob("**/diy*25*capture*.bmp"))
    if candidates:
        return "DIY Etalon 25 um", candidates[-1]

    raise FileNotFoundError("No DIY etalon BMP capture found.")


def sum_rgb_per_column(image_path: Path) -> List[int]:
    """
    For each pixel column, add up all RGB values in that column.

    Output length equals image width (here 1440 pixels).
    """
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        raw = rgb.tobytes()

    sums = [0] * width
    row_stride = width * 3
    for row in range(height):
        row_start = row * row_stride
        for col in range(width):
            i = row_start + 3 * col
            sums[col] += raw[i] + raw[i + 1] + raw[i + 2]
    return sums


def build_traces(image_files: Dict[str, Path], nm_per_pixel: float) -> Dict[str, Tuple[List[float], List[float]]]:
    traces: Dict[str, Tuple[List[float], List[float]]] = {}
    for label, image_path in image_files.items():
        rgb_sums = [float(v) for v in sum_rgb_per_column(image_path)]
        wavelengths_nm = [pixel_to_wavelength(pix, nm_per_pixel) for pix in range(len(rgb_sums))]
        if wavelengths_nm[0] > wavelengths_nm[-1]:
            wavelengths_nm = list(reversed(wavelengths_nm))
            rgb_sums = list(reversed(rgb_sums))
        traces[label] = (wavelengths_nm, rgb_sums)
    return traces


def normalize_traces(
    traces: Dict[str, Tuple[List[float], List[float]]]
) -> Dict[str, Tuple[List[float], List[float]]]:
    normalized: Dict[str, Tuple[List[float], List[float]]] = {}
    for label, (wavelengths_nm, rgb_sums) in traces.items():
        max_value = max(rgb_sums)
        if max_value <= 0:
            raise ValueError(f"Cannot normalize {label}: max RGB sum is {max_value}.")
        normalized[label] = (wavelengths_nm, [value / max_value for value in rgb_sums])
    return normalized


def plot_traces_vs_wavelength(
    traces: Dict[str, Tuple[List[float], List[float]]],
    output_path: Path,
    ylabel: str,
    title: str,
    fsr_theory_by_label: Dict[str, float],
    fsr_experimental_by_label: Dict[str, float],
    label_order: Sequence[str] | None = None,
    fsr_box_title: str = "FSR @ 600 nm (Theo / Exp)",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from exc

    plt.figure(figsize=(10, 6))
    for label, (wavelengths_nm, y_values) in traces.items():
        plt.plot(wavelengths_nm, y_values, linewidth=1.2, label=label)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()

    if label_order is None:
        label_order = list(traces.keys())

    fsr_lines: List[str] = [fsr_box_title]
    for label in label_order:
        fsr_theory = fsr_theory_by_label.get(label, float("nan"))
        fsr_exp = fsr_experimental_by_label.get(label, float("nan"))
        fsr_theory_text = f"{fsr_theory:.2f}" if math.isfinite(fsr_theory) else "n/a"
        fsr_exp_text = f"{fsr_exp:.2f}" if math.isfinite(fsr_exp) else "n/a"
        fsr_lines.append(f"{label}: {fsr_theory_text} / {fsr_exp_text} nm")
    fsr_text = "\n".join(fsr_lines)
    plt.gca().text(
        0.02,
        0.98,
        fsr_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    lab_dir = base_dir / "lab"
    plot_dir = base_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    nm_per_pixel = calibration_nm_per_pixel()
    etalon_files = find_etalon_capture_files(lab_dir)
    fsr_theory_by_label = {
        label: theoretical_fsr_nm(gap_um) for label, gap_um in ETALON_GAPS_UM.items()
    }
    etalon_label_order = list(ETALON_GAPS_UM.keys())
    traces = build_traces(etalon_files, nm_per_pixel)

    fsr_experimental_by_label: Dict[str, float] = {}
    for label, (wavelengths_nm, rgb_sums) in traces.items():
        fsr_experimental_by_label[label] = estimate_experimental_fsr_from_dips(
            wavelengths_nm, rgb_sums, fsr_theory_by_label[label]
        )

    normalized_traces = normalize_traces(traces)

    raw_output_path = plot_dir / "etalon_sizes_rgb_sum_vs_wavelength.png"
    normalized_output_path = plot_dir / "etalon_sizes_rgb_sum_vs_wavelength_normalized.png"
    plot_traces_vs_wavelength(
        traces,
        raw_output_path,
        ylabel="RGB Sum",
        title="Etalon 15/25/50/100 um: RGB Sum vs Wavelength",
        fsr_theory_by_label=fsr_theory_by_label,
        fsr_experimental_by_label=fsr_experimental_by_label,
        label_order=etalon_label_order,
    )
    plot_traces_vs_wavelength(
        normalized_traces,
        normalized_output_path,
        ylabel="Normalized RGB Sum",
        title="Etalon 15/25/50/100 um: Normalized RGB Sum vs Wavelength",
        fsr_theory_by_label=fsr_theory_by_label,
        fsr_experimental_by_label=fsr_experimental_by_label,
        label_order=etalon_label_order,
    )

    angle_captures = find_angle_capture_files(lab_dir / "angle")
    angle_files: Dict[str, Path] = {label: path for label, _, path in angle_captures}
    angle_by_label: Dict[str, float] = {label: angle_deg for label, angle_deg, _ in angle_captures}
    angle_label_order = [label for label, _, _ in angle_captures]

    angle_traces = build_traces(angle_files, nm_per_pixel)
    angle_fsr_theory_by_label = {
        label: theoretical_fsr_nm_with_angle(ANGLE_ETALON_GAP_UM, angle_by_label[label])
        for label in angle_label_order
    }
    angle_fsr_experimental_by_label: Dict[str, float] = {}
    for label, (wavelengths_nm, rgb_sums) in angle_traces.items():
        angle_fsr_experimental_by_label[label] = estimate_experimental_fsr_from_dips(
            wavelengths_nm, rgb_sums, angle_fsr_theory_by_label[label]
        )
    angle_normalized_traces = normalize_traces(angle_traces)

    angle_raw_output_path = plot_dir / "etalon_15um_angles_rgb_sum_vs_wavelength.png"
    angle_normalized_output_path = (
        plot_dir / "etalon_15um_angles_rgb_sum_vs_wavelength_normalized.png"
    )
    plot_traces_vs_wavelength(
        angle_traces,
        angle_raw_output_path,
        ylabel="RGB Sum",
        title="Etalon 15 um: RGB Sum vs Wavelength for Different Angles",
        fsr_theory_by_label=angle_fsr_theory_by_label,
        fsr_experimental_by_label=angle_fsr_experimental_by_label,
        label_order=angle_label_order,
        fsr_box_title="FSR @ 600 nm (Theo with cos(theta) / Exp)",
    )
    plot_traces_vs_wavelength(
        angle_normalized_traces,
        angle_normalized_output_path,
        ylabel="Normalized RGB Sum",
        title="Etalon 15 um: Normalized RGB Sum vs Wavelength for Different Angles",
        fsr_theory_by_label=angle_fsr_theory_by_label,
        fsr_experimental_by_label=angle_fsr_experimental_by_label,
        label_order=angle_label_order,
        fsr_box_title="FSR @ 600 nm (Theo with cos(theta) / Exp)",
    )

    # Angle plot without negative angle
    angle_non_negative_labels = [label for label in angle_label_order if angle_by_label[label] >= 0]
    angle_non_negative_traces = {label: angle_traces[label] for label in angle_non_negative_labels}
    angle_non_negative_fsr_theory = {
        label: angle_fsr_theory_by_label[label] for label in angle_non_negative_labels
    }
    angle_non_negative_fsr_experimental = {
        label: angle_fsr_experimental_by_label[label] for label in angle_non_negative_labels
    }
    angle_no_negative_output_path = (
        plot_dir / "etalon_15um_angles_non_negative_rgb_sum_vs_wavelength.png"
    )
    plot_traces_vs_wavelength(
        angle_non_negative_traces,
        angle_no_negative_output_path,
        ylabel="RGB Sum",
        title="Etalon 15 um: RGB Sum vs Wavelength (Angles >= 0 deg)",
        fsr_theory_by_label=angle_non_negative_fsr_theory,
        fsr_experimental_by_label=angle_non_negative_fsr_experimental,
        label_order=angle_non_negative_labels,
        fsr_box_title="FSR @ 600 nm (Theo with cos(theta) / Exp)",
    )

    # Angle plot with only +5 and -5 deg
    angle_pm5_labels = [label for label in angle_label_order if abs(angle_by_label[label]) == 5]
    angle_pm5_traces = {label: angle_traces[label] for label in angle_pm5_labels}
    angle_pm5_fsr_theory = {label: angle_fsr_theory_by_label[label] for label in angle_pm5_labels}
    angle_pm5_fsr_experimental = {
        label: angle_fsr_experimental_by_label[label] for label in angle_pm5_labels
    }
    angle_pm5_output_path = plot_dir / "etalon_15um_angles_plusminus5_rgb_sum_vs_wavelength.png"
    plot_traces_vs_wavelength(
        angle_pm5_traces,
        angle_pm5_output_path,
        ylabel="RGB Sum",
        title="Etalon 15 um: RGB Sum vs Wavelength (Angles -5 and +5 deg)",
        fsr_theory_by_label=angle_pm5_fsr_theory,
        fsr_experimental_by_label=angle_pm5_fsr_experimental,
        label_order=angle_pm5_labels,
        fsr_box_title="FSR @ 600 nm (Theo with cos(theta) / Exp)",
    )

    # DIY etalon plot
    diy_label, diy_path = find_diy_capture_file(lab_dir)
    diy_files = {diy_label: diy_path}
    diy_traces = build_traces(diy_files, nm_per_pixel)
    diy_fsr_theory_by_label = {diy_label: theoretical_fsr_nm(DIY_ETALON_GAP_UM)}
    diy_fsr_experimental_by_label = {
        diy_label: estimate_experimental_fsr_from_dips(
            diy_traces[diy_label][0], diy_traces[diy_label][1], diy_fsr_theory_by_label[diy_label]
        )
    }
    diy_output_path = plot_dir / "diy_25um_rgb_sum_vs_wavelength.png"
    plot_traces_vs_wavelength(
        diy_traces,
        diy_output_path,
        ylabel="RGB Sum",
        title="DIY Etalon 25 um: RGB Sum vs Wavelength",
        fsr_theory_by_label=diy_fsr_theory_by_label,
        fsr_experimental_by_label=diy_fsr_experimental_by_label,
        label_order=[diy_label],
        fsr_box_title="FSR @ 600 nm (Theo / Exp)",
    )

    # DIY vs reference 25 um plot
    compare_labels = ["Etalon 25 um", diy_label]
    diy_vs_25_traces = {
        "Etalon 25 um": traces["Etalon 25 um"],
        diy_label: diy_traces[diy_label],
    }
    diy_vs_25_fsr_theory = {
        "Etalon 25 um": fsr_theory_by_label["Etalon 25 um"],
        diy_label: diy_fsr_theory_by_label[diy_label],
    }
    diy_vs_25_fsr_experimental = {
        "Etalon 25 um": fsr_experimental_by_label["Etalon 25 um"],
        diy_label: diy_fsr_experimental_by_label[diy_label],
    }
    diy_vs_25_output_path = plot_dir / "diy_vs_etalon_25um_rgb_sum_vs_wavelength.png"
    plot_traces_vs_wavelength(
        diy_vs_25_traces,
        diy_vs_25_output_path,
        ylabel="RGB Sum",
        title="DIY 25 um vs Etalon 25 um: RGB Sum vs Wavelength",
        fsr_theory_by_label=diy_vs_25_fsr_theory,
        fsr_experimental_by_label=diy_vs_25_fsr_experimental,
        label_order=compare_labels,
        fsr_box_title="FSR @ 600 nm (Theo / Exp)",
    )

    print(f"Calibration slope: {nm_per_pixel:.8f} nm/pixel")
    print(f"Center anchor: pixel {CENTER_PIXEL} -> {CENTER_WAVELENGTH_NM} nm")
    for label, path in etalon_files.items():
        print(f"{label}: {path.name}")
    print("FSR theoretical/experimental (nm):")
    for label in etalon_label_order:
        fsr_theory = fsr_theory_by_label[label]
        fsr_exp = fsr_experimental_by_label[label]
        fsr_exp_text = f"{fsr_exp:.3f}" if math.isfinite(fsr_exp) else "n/a"
        print(f"  {label}: {fsr_theory:.3f} / {fsr_exp_text}")
    print("Angle scan (15 um) FSR theoretical/experimental (nm):")
    for label in angle_label_order:
        fsr_theory = angle_fsr_theory_by_label[label]
        fsr_exp = angle_fsr_experimental_by_label[label]
        fsr_exp_text = f"{fsr_exp:.3f}" if math.isfinite(fsr_exp) else "n/a"
        print(f"  {label}: {fsr_theory:.3f} / {fsr_exp_text}")
    print("DIY etalon FSR theoretical/experimental (nm):")
    diy_fsr_exp = diy_fsr_experimental_by_label[diy_label]
    diy_fsr_exp_text = f"{diy_fsr_exp:.3f}" if math.isfinite(diy_fsr_exp) else "n/a"
    print(f"  {diy_label}: {diy_fsr_theory_by_label[diy_label]:.3f} / {diy_fsr_exp_text}")
    print(f"Saved plot: {raw_output_path}")
    print(f"Saved plot: {normalized_output_path}")
    print(f"Saved plot: {angle_raw_output_path}")
    print(f"Saved plot: {angle_normalized_output_path}")
    print(f"Saved plot: {angle_no_negative_output_path}")
    print(f"Saved plot: {angle_pm5_output_path}")
    print(f"Saved plot: {diy_output_path}")
    print(f"Saved plot: {diy_vs_25_output_path}")


if __name__ == "__main__":
    main()
