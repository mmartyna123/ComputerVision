# %%
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import datetime
import glob
import cv2
from matplotlib.colors import Normalize


def set_axis(x, no_labels = 7)->tuple[np.array, np.array]:
    """Sets the x-axis positions and labels for a plot.

    Args:
        x (np.array): The x-axis data.
        no_labels (int, optional): The number of labels to display. Defaults to 7.

    Returns:
        tuple[np.array, np.array]: A tuple containing:
            - The positions of the labels on the x-axis.
            - The labels themselves.
    """
    nx = x.shape[0]
    step_x = int(nx / (no_labels - 1))
    x_positions = np.arange(0,nx,step_x)
    x_labels = x[::step_x]
    return x_positions, x_labels

# %%
DX = 5.106500953873407
DT = 0.0016



# %%
path_out = "./Maria/"
files = glob.glob(path_out + "*")
files.sort()
dx = 5.106500953873407
dt = 0.0016


data = []
first_filename = files[0]
for file in files:
    data.append(np.load(file))
data = np.concatenate(data)
time_start = datetime.datetime.strptime(
    "2024-05-07 " + first_filename.split("\\")[-1].split(".")[0], "%Y-%m-%d %H%M%S"
)
index = pd.date_range(start=time_start, periods=len(data), freq=f"{dt}s")


columns = np.arange(len(data[0])) * dx


df = pd.DataFrame(data=data, index=index, columns=columns)

# %%
img = np.array(df)
img -= img.mean()
img = np.abs(img)
low, high = np.percentile(img, [3, 99])
img = np.clip(img, low, high)
img -= img.min()
img *= 255.0 / img.max()
img = img.astype(np.uint8)
img

# %%
import PIL

def imshow(img, lines=[]):
    fig = plt.figure(figsize=(12,16))
    ax = plt.axes()

    norm = Normalize(vmin=0, vmax=255, clip=True)

    im = ax.imshow(img,interpolation='none',aspect='auto',norm=norm)

    x_values = np.linspace(0, img.shape[1], num=100)
    for intercept, slope in lines:
        ls_y_values = intercept + slope * x_values
        ax.plot(x_values, np.clip(ls_y_values, 0, img.shape[0]), color="red", linewidth=2)

    plt.ylabel('time')
    plt.xlabel('space [m]')

    cax = fig.add_axes([ax.get_position().x1+0.06,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    x_positions, x_labels = set_axis(df.columns)
    ax.set_xticks(x_positions, np.round(x_labels))
    y_positions, y_labels = set_axis(df.index.time)
    ax.set_yticks(y_positions, y_labels)
    plt.show()

def fft(img, size=None):
    f = np.fft.fft2(img, size)
    fshift = np.fft.fftshift(f)
    spectrum = 20 * np.log(np.abs(fshift) + 1)
    return fshift, spectrum

def ifft(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    return np.real(img_back)

def get_mask(shape, div):
    mask = np.zeros(shape, np.float32)
    center_row, center_col = shape[0] // 2, shape[1] // 2

    rows_scale = shape[0] / max(shape)
    cols_scale = shape[1] / max(shape)
    axes = (int(center_col / div / cols_scale), int(center_row / div / rows_scale))

    mask = cv2.ellipse(mask, (center_col, center_row), axes, 0, 0, 360, 1, -1)
    return mask

def create_frequency_mask(shape, freq_min, freq_max):
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((X - center_col)**2 + (Y - center_row)**2)

    mask = np.logical_and(freq_min <= dist_from_center, dist_from_center <= freq_max)
    return mask.astype(np.float32)

fshift, spectrum = fft(img)


# Using create_frequency_mask
low_pass_mask = create_frequency_mask(img.shape, 0, 30)
high_pass_mask = 1 - create_frequency_mask(img.shape, 0, 50)

# This one may be also interesting if played with parameters
both_pass_mask = create_frequency_mask(img.shape, 10, 50)

low_pass_fshift = fshift * low_pass_mask
high_pass_fshift = fshift * high_pass_mask
both_pass_fshift = fshift * both_pass_mask


low_pass_img = ifft(low_pass_fshift)
high_pass_img = ifft(high_pass_fshift)
both_pass_img = ifft(both_pass_fshift)

imshow(np.concatenate([low_pass_img, high_pass_img, both_pass_img], 1))


# Using get_mask
low_pass_mask = get_mask(img.shape, 2000)
high_pass_mask = 1 - get_mask(img.shape, 2000)

low_pass_fshift = low_pass_mask * fshift
high_pass_fshift = high_pass_mask * fshift
both_pass_fshift = high_pass_mask * fshift * low_pass_mask


low_pass_fshift_img = ifft(low_pass_fshift)
high_pass_fshift_img = ifft(high_pass_fshift)
both_pass_fshift_img = ifft(both_pass_fshift)

imshow(np.concatenate([low_pass_fshift_img, high_pass_fshift_img, both_pass_fshift_img], 1))

# %%
mod_fshift = fshift.copy()
mod_fshift[0:35000] = 0

mask_1 = create_frequency_mask(img.shape, 1, 30)
mask_2 = create_frequency_mask(img.shape, 1, 30)

mask_1_fshift = mask_1 * mod_fshift
mask_2_fshift = mask_2 * mod_fshift
both_pass_fshift = (mask_1_fshift + mask_2_fshift)


mask_1_fshift_img = ifft(mask_1_fshift)
mask_2_fshift_img = ifft(mask_2_fshift)
both_pass_fshift_img = ifft(both_pass_fshift)

imshow(np.concatenate([mask_1_fshift_img, mask_2_fshift_img, both_pass_fshift_img], 1))

# %%
def draw_hough_lines(img, lines, rho_res=1, theta_res=np.pi/180):
    img_with_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()

    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img_with_lines

def map_hough_lines_to_imshow_lines(hough_lines):
    lines_mapped = []

    for rho, theta in hough_lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            if b != 0:  # Avoid division by zero
                slope = -a / b  # Slope
                intercept = rho / b  # Intercept
                lines_mapped.append((intercept, slope))

    return lines_mapped

img_filtered = both_pass_fshift_img.clip(0, 255).astype(np.uint8)
# img_filtered = cv2.GaussianBlur(img_filtered, (5, 5), 0)

# Does not displayed correctly on imshow
img_copy = img_filtered[::200, :]

imshow(img_filtered)

# Apply edge detection (Canny)
edges = cv2.Canny(img_filtered, 50, 110)

kernel = np.ones((5, 5), np.uint8)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

imshow(edges)

edges = edges[::200, :]

# Detect Hough lines

lines = cv2.HoughLines(edges, rho=1, theta=np.pi/180, threshold=100)

if lines is not None:
    img_with_long_lines = draw_hough_lines(img_filtered, lines)
    imshow(img_with_long_lines)
else:
    print("No lines detected")

# %%
def imshow(img, lines=[]):
    fig = plt.figure(figsize=(12,16))
    ax = plt.axes()

    norm = Normalize(vmin=0, vmax=255, clip=True)

    im = ax.imshow(img,interpolation='none',aspect='auto',norm=norm)

    x_values = np.linspace(0, img.shape[1], num=100)
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            ax.plot([l[0], l[2]], [l[1], l[3]], color="red")

    plt.ylabel('time')
    plt.xlabel('space [m]')

    cax = fig.add_axes([ax.get_position().x1+0.06,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    x_positions, x_labels = set_axis(df.columns)
    ax.set_xticks(x_positions, np.round(x_labels))
    y_positions, y_labels = set_axis(df.index.time)
    ax.set_yticks(y_positions, y_labels)
    plt.show()

max_theta = 0.01
lines = cv2.HoughLinesP(img_filtered, rho=1, theta=np.pi/180, threshold=90, lines=None, minLineLength=100, maxLineGap=100)# min_theta=-max_theta, max_theta=max_theta)

imshow(img_filtered, lines)

# %%
def imshow(img, lines=[]):
    fig = plt.figure(figsize=(12,16))
    ax = plt.axes()

    norm = Normalize(vmin=0, vmax=255, clip=True)

    im = ax.imshow(img,interpolation='none',aspect='auto',norm=norm)

    x_values = np.linspace(0, img.shape[1], num=100)
    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            ax.plot([l[0], l[2]], [l[1], l[3]], color="red")

    plt.ylabel('time')
    plt.xlabel('space [m]')

    cax = fig.add_axes([ax.get_position().x1+0.06,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    x_positions, x_labels = set_axis(df.columns)
    ax.set_xticks(x_positions, np.round(x_labels))
    # y_positions, y_labels = set_axis(df.index.time)
    # ax.set_yticks(y_positions, y_labels)
    plt.show()

img_downsampled = cv2.resize(img_filtered, (150, 52), interpolation=cv2.INTER_AREA)
imshow(img_downsampled)

# %%
lines = cv2.HoughLinesP(img_downsampled, rho=1, theta=np.pi/180, threshold=80, lines=None, minLineLength=8, maxLineGap=16)# min_theta=-max_theta, max_theta=max_theta)
imshow(img_downsampled, lines)

# %%
# import sympy as sp
#
# x, a1, b1, a2, b2, min, max = sp.symbols("x, a1, b1, a2, b2, min, max")
#
# sp.simplify(sp.integrate(a1*x + b1 - (a2*x + b2), (x, min, max)))

# %%
def calc_area(a1, b1, a2, b2, min, max):
    assert min < max
    return max**2 * (a1 - a2)/2 + max*(b1 - b2) + min**2 * (-a1 + a2)/2 - min*(b1 - b2)

def calculate_slope_and_intercept(x1, y1, x2, y2):
        # Calculate the slope (a)
        a = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')  # Handle vertical line
        # Calculate the y-intercept (b)
        b = y1 - a * x1
        return a, b

def line_intersection(p1, p2, p3, p4):
    """
    Find the intersection point of two line segments (p1, p2) and (p3, p4).

    Parameters:
    p1, p2: Tuples representing the endpoints of the first segment (x1, y1), (x2, y2)
    p3, p4: Tuples representing the endpoints of the second segment (x3, y3), (x4, y4)

    Returns:
    A tuple (x, y) representing the intersection point or None if there is no intersection.
    """

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Calculate the determinants
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Check if lines are parallel
    if denom == 0:
        return None  # Lines are parallel or collinear

    # Calculate intersection point
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Check if the intersection point is within both segments
    if 0 <= t <= 1 and 0 <= u <= 1:
        intersect_x = x1 + t * (x2 - x1)
        intersect_y = y1 + t * (y2 - y1)
        return (intersect_x, intersect_y)

    return None  # No intersection within the segments

def remove_duplicates(lines, min_avg_distance):
    to_remove = set()
    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i][0]
        max_i = max(x1, x2)
        min_i = min(x1, x2)
        a, b = calculate_slope_and_intercept(x1, y1, x2, y2)
        areas = []
        for j in range(i + 1, len(lines)):
            x3, y3, x4, y4 = lines[j][0]
            c, d = calculate_slope_and_intercept(x3, y3, x4, y4)
            max_j = max(x3, x4)
            min_j = min(x3, x4)
            max_x = min(max_i, max_j)
            min_x = max(min_i, min_j)
            if min_x >= max_x: # lines don't overlap
                continue
            intersection = line_intersection([x1, y1], [x2, y2], [x3, y3], [x4, y4])
            if intersection is not None:
                p, pval = intersection
                min_y_i = a * min_x + b
                min_y_j = c * min_x + d
                if min_y_i > min_y_j:
                    area = calc_area(a, b, c, d, min_x, p) / (p - min_x)
                    area += calc_area(c, d, a, b, p, max_x) / (max_x - p)
                else:
                    area = calc_area(c, d, a, b, min_x, p) / (p - min_x)
                    area += calc_area(a, b, c, d, p, max_x) / (max_x - p)
                areas.append((j, area))
            else:
                areas.append((j, abs(calc_area(a, b, c, d, min_x, max_x)) / (max_x - min_x)))
        for j, area in areas:
            if area < min_avg_distance: # maybe merge instead of removing
                x3, y3, x4, y4 = lines[j][0]
                if abs(x1 - x2) > abs(x3 - x4):
                    to_remove.add(j)
                else:
                    to_remove.add(i)

    return np.delete(lines, list(to_remove), axis=0)

lines = remove_duplicates(lines, 3)
imshow(img_downsampled, lines)

# %%
def imshow(img, lines=[], lines_scale=np.ones(shape=(2,))):
    fig = plt.figure(figsize=(12,16))
    ax = plt.axes()

    norm = Normalize(vmin=0, vmax=255, clip=True)

    im = ax.imshow(img,interpolation='none',aspect='auto',norm=norm)

    if lines is not None:
        for i in range(0, len(lines)):
            l = lines[i][0]
            a, b = calculate_slope_and_intercept(*l)
            speed = float("inf") if a == 0 else abs(1/a*lines_scale[1]/lines_scale[0]*DX/DT) * 3.6
            if speed > 300: # nobody would drive this fast
                continue
            ax.text(l[0] * lines_scale[1], l[1] * lines_scale[0], f"{round(speed, 2)} km/h", color="red", horizontalalignment="center", verticalalignment="top")
            ax.plot([l[0] * lines_scale[1], l[2] * lines_scale[1]], [l[1] * lines_scale[0], l[3] * lines_scale[0]], color="red")

    plt.ylabel('time')
    plt.xlabel('space [m]')

    cax = fig.add_axes([ax.get_position().x1+0.06,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    x_positions, x_labels = set_axis(df.columns)
    ax.set_xticks(x_positions, np.round(x_labels))
    y_positions, y_labels = set_axis(df.index.time)
    ax.set_yticks(y_positions, y_labels)
    plt.show()

imshow(img, lines, np.array(img.shape) / np.array(img_downsampled.shape))

# %%
def imshow(img, lines=[]):
    fig = plt.figure(figsize=(12,16))
    ax = plt.axes()

    norm = Normalize(vmin=0, vmax=255, clip=True)

    im = ax.imshow(img,interpolation='none',aspect='auto',norm=norm)

    x_values = np.linspace(0, img.shape[1], num=100)
    for intercept, slope in lines:
        ls_y_values = intercept + slope * x_values
        ax.plot(x_values, np.clip(ls_y_values, 0, img.shape[0]), color="red", linewidth=2)

    plt.ylabel('time')
    plt.xlabel('space [m]')

    cax = fig.add_axes([ax.get_position().x1+0.06,ax.get_position().y0,0.02,ax.get_position().height])
    plt.colorbar(im, cax=cax)
    x_positions, x_labels = set_axis(df.columns)
    ax.set_xticks(x_positions, np.round(x_labels))
    y_positions, y_labels = set_axis(df.index.time)
    ax.set_yticks(y_positions, y_labels)
    plt.show()

imshow(img)

# %%
slice_idx = int(0.27 * img.shape[0])
slice = img[slice_idx]

slice

# %%
import cv2

def cvconv(f, g):
    # padding
    pad_v = (g.shape[0] - 1) // 2
    pad_h = (g.shape[1] - 1) // 2
    fb = cv2.copyMakeBorder(f, pad_v, pad_v, pad_h, pad_h, cv2.BORDER_CONSTANT, 0)

    g = np.flip(g)

    # convolution
    fg_cv = cv2.filter2D(fb.astype(g.dtype), -1, g)

    # remove padding from result (opencv does not do this automatically)
    return fg_cv[pad_v : fb.shape[0] - pad_v, pad_h : fb.shape[1] - pad_h]

der = np.array([[-1, 1]], np.float32)

x = np.arange(0, img.shape[1])

y_hat = cvconv(slice[np.newaxis], der) / (x[1] - x[0])
y_hat = y_hat[0]

where = abs(y_hat) < 0.0003
extremes = x[where]
extremes

# %%
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema

a = extremes.reshape(-1, 1)
kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
s = np.linspace(a.min(), a.max())
e = kde.score_samples(s.reshape(-1,1))

mi = argrelextrema(e, np.less)[0]

segments = [a[(a >= left) * (a <= right)] for left, right in zip((s[0],) + tuple(s[mi]), tuple(s[mi]) + (s[-1],))]
segments

# %%
suspected_peaks = [segment.mean() for segment in segments]
suspected_peaks

# %%
def get_peaks(slice):
    der = np.array([[-1, 1]], np.float32)

    x = np.arange(0, img.shape[1])

    y_hat = cvconv(slice[np.newaxis], der) / (x[1] - x[0])
    y_hat = y_hat[0]

    where = abs(y_hat) < 0.0003
    extremes = x[where]

    if len(extremes) == 0:
        return np.array([])

    a = extremes.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
    s = np.linspace(a.min(), a.max())
    e = kde.score_samples(s.reshape(-1,1))

    mi = argrelextrema(e, np.less)[0]

    segments = [a[(a >= left) * (a <= right)] for left, right in zip((s[0],) + tuple(s[mi]), tuple(s[mi]) + (s[-1],))]

    return np.array([segment.mean() for segment in segments])

for i in range(10):
    print(i, [int(x) for x in get_peaks(img[slice_idx+i])])

# %%
from tqdm import trange

# this is just for visualization
peaks_img = np.zeros_like(img)
for i in trange(img.shape[0]):
    for x in get_peaks(img[i]):
        peaks_img[i, int(x)] = 255

imshow(peaks_img)

# %%
def get_peaks(slice, threshold): #min, max):
    der = np.array([[-1, 1]], np.float32)

    x = np.arange(0, img.shape[1])

    y_hat = cvconv(slice[np.newaxis], der) / (x[1] - x[0])
    y_hat = y_hat[0]

    where = abs(y_hat) < 0.0003
    extremes = x[where]

    extreme_values = slice[where]
    extremes = extremes[extreme_values > threshold]
    # extremes = extremes[np.all((extreme_values >= min) & (extreme_values <= max))]

    # if extremes.shape[0] == 0 or extremes.shape[1] == 0:
    if len(extremes) == 0:
        return np.array([])

    a = extremes.reshape(-1, 1)
    kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
    s = np.linspace(a.min(), a.max())
    e = kde.score_samples(s.reshape(-1,1))

    mi = argrelextrema(e, np.less)[0]

    segments = [a[(a >= left) * (a <= right)] for left, right in zip((s[0],) + tuple(s[mi]), tuple(s[mi]) + (s[-1],))]

    return np.array([segment.mean() for segment in segments])

peaks_img = np.zeros_like(img)
for i in trange(img.shape[0]):
    for x in get_peaks(img[i], 50):#30, 255):
        peaks_img[i, int(x)] = 255

imshow(peaks_img)

# %%
img_blur = cv2.blur(img, (5, 5))
peaks_img = np.zeros_like(img)
for i in trange(img.shape[0]):
    for x in get_peaks(img_blur[i], 50):#30, 255):
        peaks_img[i, int(x)] = 255

imshow(peaks_img)

# %%
kernel = np.ones((5, 5), np.uint8)

# Proper closing: Close -> Open -> Close
img_proper_closing = cv2.morphologyEx(
    cv2.morphologyEx(
        cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel),
        cv2.MORPH_OPEN, kernel),
    cv2.MORPH_CLOSE, kernel)

peaks_img = np.zeros_like(img)
for i in trange(img.shape[0]):
    for x in get_peaks(img_proper_closing[i], 70):#30, 255):
        peaks_img[i, int(x)] = 255

imshow(peaks_img)

# %%
current_peak = get_peaks(img[slice_idx])
for i in range(1, 10):
    next_peak = get_peaks(img[slice_idx+i])

    print(current_peak, next_peak)

    # # closest matches
    # max_matches = min(len(last_peak), len(current_peak))
    # # print(max_matches)
    #
    # print(last_peak, current_peak)
    # distances = np.abs(last_peak[:, np.newaxis] - current_peak)
    # # print(distances)
    #
    # distances = distances.min(axis=0)
    # # print(distances)
    #
    # if len(distances) <= max_matches:
    #     indices = np.arange(max_matches)
    # else:
    #     indices = np.argpartition(distances, max_matches)[:max_matches]
    # # print(indices)
    #
    # print(current_peak[indices])

    # print("====")

    current_peak = next_peak

# %%
A = np.array([1, 2, 3])         # Length 3
B = np.array([4, 5, 6])   # Length 5

# Calculate distances using broadcasting
distances = np.abs(A[:, np.newaxis] - B)
print("Distance Matrix (NumPy):")
print(distances)

np.argpartition(distances.min(axis=0), 2)[:2]

# %%
grad_x = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
imshow(grad_x)

# %%
grad_y = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)

grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
imshow(grad)

# %%
_, grad_thold = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow(grad_thold)

# %%
grad_blur = cv2.GaussianBlur(grad, (5,5), 0)
_, grad_blur_thold = cv2.threshold(grad_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow(grad_blur_thold)

# %%
grad_blur_thold_edges = cv2.Canny(grad_blur_thold, 100, 200)
imshow(grad_blur_thold_edges)

# %%
from scipy.ndimage import binary_hit_or_miss

def thinning(img, struct):
    hit, miss = struct.copy(), struct.copy()
    hit[struct == -1] = 0
    miss[struct == 1] = 0
    miss[struct == -1] = 1
    return np.logical_and(img, np.logical_not(binary_hit_or_miss(img, hit, miss)))

def skeletonize(img):
    s1 = np.array([[-1, -1, -1], [0, 1, 0], [1, 1, 1]])
    s2 = np.array([[0, -1, -1], [1, 1, -1], [0, 1, 0]])

    result = img.copy()
    while True:
      temp = result.copy()
      for i in range(4):
        temp = np.rot90(thinning(thinning(temp, s1), s2))
      if np.all(temp == result):
        break
      result = temp.copy()

    return result

grad_blur_thold_skeleton = skeletonize(grad_blur_thold)
grad_blur_thold_skeleton = (grad_blur_thold_skeleton * 255).astype(np.uint8)
imshow(grad_blur_thold_skeleton)

# %%
t = grad_blur_thold_skeleton.copy()
t[img <= 70] = 0
imshow(t)

# %%