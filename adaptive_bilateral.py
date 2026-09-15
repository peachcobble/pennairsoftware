import cv2
import numpy as np

# written by claude after parsing through the
# adaptive bilateral filtering theoretical paper 

def adaptive_bilateral_filter(f, theta, sigma, rho):
    """
    Brute-force adaptive bilateral filter.

    f, theta, sigma: 2D float arrays, same shape (grayscale channel)
        f     - input image
        theta - per-pixel range kernel center
        sigma - per-pixel range kernel width
    rho: spatial kernel width (scalar)
    """
    f = f.astype(np.float64)
    theta = theta.astype(np.float64)
    sigma = np.maximum(sigma.astype(np.float64), 1e-6)

    win = int(np.ceil(3 * rho))
    H, W = f.shape
    padded = cv2.copyMakeBorder(f, win, win, win, win, cv2.BORDER_REFLECT)

    num = np.zeros((H, W), dtype=np.float64)
    den = np.zeros((H, W), dtype=np.float64)

    for dy in range(-win, win + 1):
        for dx in range(-win, win + 1):
            w = np.exp(-(dx * dx + dy * dy) / (2.0 * rho * rho))
            shifted = padded[win + dy: win + dy + H, win + dx: win + dx + W]
            diff = shifted - theta
            phi = np.exp(-(diff * diff) / (2.0 * sigma * sigma))
            weight = w * phi
            num += weight * shifted
            den += weight

    return num / den


def affine_map(x, out_min, out_max, invert=False):
    """Map x to [out_min, out_max] linearly. invert=True maps low x -> high output."""
    x_min, x_max = x.min(), x.max()
    if x_max - x_min < 1e-12:
        return np.full_like(x, (out_min + out_max) / 2.0)
    norm = (x - x_min) / (x_max - x_min)
    if invert:
        norm = 1.0 - norm
    return out_min + norm * (out_max - out_min)


def sharpen_params(f, avg_ksize=15, log_ksize=0, log_sigma=2.0,
                    sigma_min=5.0, sigma_max=40.0):
    """
    theta(i) = f(i) + (f(i) - f_avg(i))
    sigma(i) via affine map of |LoG(f)|, low response -> high sigma
    """
    f_avg = cv2.blur(f, (avg_ksize, avg_ksize))
    theta = f + (f - f_avg)

    smoothed = cv2.GaussianBlur(f, (log_ksize, log_ksize), log_sigma)
    log = cv2.Laplacian(smoothed, cv2.CV_64F)
    log_abs = np.abs(log)

    sigma = affine_map(log_abs, sigma_min, sigma_max, invert=True)
    return theta, sigma


def sharpen_channel(f, rho=5, avg_ksize=15, sigma_min=5.0, sigma_max=40.0):
    theta, sigma = sharpen_params(f, avg_ksize=avg_ksize,
                                   sigma_min=sigma_min, sigma_max=sigma_max)
    return adaptive_bilateral_filter(f, theta, sigma, rho)


if __name__ == "__main__":
    img = cv2.imread("input.jpg", cv2.IMREAD_COLOR).astype(np.float64)

    channels = cv2.split(img)
    out_channels = [sharpen_channel(c, rho=5) for c in channels]
    out = cv2.merge(out_channels)

    out = np.clip(out, 0, 255).astype(np.uint8)
    cv2.imwrite("output.jpg", out)
