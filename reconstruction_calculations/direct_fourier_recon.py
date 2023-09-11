import math
import numpy as np
from scipy import fftpack
from scipy.interpolate import griddata
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon

from gasmas_calculations.oxygen_concentration_calibrated import gasmas_o2_calc


class DFR2D:

    def __init__(self, projections, angles, iterations, calibration_constant, pathlengths=None):
        self.projections = projections
        self.angles = angles
        self.iterations = iterations
        self.calibration_constant = calibration_constant
        self.pathlengths = pathlengths if pathlengths is not None else None

        # self._convert_projections_to_oxy()
        self._create_sinogram_radon()
        # self._create_sinogram_manual()
        # self._fourier_transform_1d()
        # self._mlem()
        self._fsr(dfr=False)
        # self._sinc_transform()
        # self._fourier_transforms()
        # self._slice_theorem()
        # self._reconstruction_from_fourier()

    def _convert_projections_to_oxy(self):
        oxygen_array = inverse_oxy_conc_calc(self.projections, self.pathlengths, self.calibration_constant)
        fig, ax = plt.subplots(2)
        ax[0].imshow(self.projections, aspect='auto')
        ax[0].set_title('projections')
        ax[1].imshow(oxygen_array, aspect='auto')
        ax[1].set_title('oxygen array')
        plt.show()
        self.projections = oxygen_array

    def _create_sinogram_radon(self):
        projections = np.asarray(self.projections)
        projections_padded = pad_projections(projections)
        angles = self.angles
        angles_degrees = np.linspace(0, math.degrees(max(angles)), len(angles), endpoint=True)

        sinogram = radon(projections_padded, angles_degrees, circle=False)
        plt.imshow(sinogram)
        plt.show()

        self.projections = projections_padded
        self.sinogram = sinogram
        self.sinogram_resolution = np.shape(sinogram)[1]
        self.sinogram_exposures = len(self.sinogram)

    def _create_sinogram_manual(self):
        sinogram = np.array(
            [
                np.sum(
                    rotate(
                        self.projections,
                        np.rad2deg(i),
                        reshape=False
                    ),
                    axis=0,
                )
                for i in self.angles
            ]
        )
        self.sinogram = sinogram
        self.sinogram_resolution = np.shape(sinogram)[1]
        self.sinogram_exposures = len(self.sinogram)

    def _mlem(self):
        sinogram = self.sinogram
        angles = self.angles
        iterations = self.iterations
        projections = np.asarray(self.projections)

        angles = np.linspace(0, math.degrees(max(angles)), len(angles), endpoint=True)

        mlem_array = np.ones(np.shape(projections))
        sino_ones_array = np.ones(np.shape(sinogram))
        sens_image = iradon(sino_ones_array, angles, circle=False, filter_name=None)

        for i in range(iterations):
            forward_proj = radon(mlem_array, angles, circle=False)
            ratio = sinogram / (forward_proj + 0.00001)
            correction = iradon(ratio, angles, circle=False, filter_name=None) / sens_image
            print(f'correction avg : {np.average(correction)}')
            mlem_array = mlem_array * correction

            fig, axes = plt.subplots(2, 3)
            axes[0, 1].imshow(sinogram.T, aspect='auto')
            axes[0, 1].set_title('sinogram')

            axes[0, 2].imshow(ratio.T, aspect='auto')
            axes[0, 2].set_title('ratio')

            axes[1, 0].imshow(mlem_array, aspect='auto')
            axes[1, 0].set_title('mlem recon')

            axes[1, 1].imshow(forward_proj.T, aspect='auto')
            axes[1, 1].set_title('fp')

            axes[1, 2].imshow(correction)
            axes[1, 2].set_title('correction')
            plt.suptitle(f'MLEM Reconstruction over {int(i + 1)} iterations')
            plt.tight_layout()
            plt.show()

            plt.imshow(mlem_array)
            plt.show()

    def _fsr(self, dfr=False):
        sinogram = self.sinogram
        num_projections = len(sinogram)
        sinogram_ft = []

        for i, row in enumerate(sinogram):
            sinogram_ft.append(
                np.fft.fftshift(
                    np.fft.fft(
                        np.fft.ifftshift(row)
                    )
                )
            )
        sinogram_ft = np.asarray(np.transpose(sinogram_ft))

        if dfr:
            self._dfr(sinogram_ft)

        ramp_filter = create_ramp_filter(num_projections)

        sinogram_ft_filtered = []
        for i, row in enumerate(sinogram_ft):
            filtered_row = []
            for j, proj_val in enumerate(row):
                filtered_row.append(proj_val * ramp_filter[j])
            sinogram_ft_filtered.append(filtered_row)

        sinogram_ft_filtered = np.asarray(sinogram_ft_filtered)

        reconstructed_image = []
        for row in sinogram_ft_filtered:
            reconstructed_image.append(
                np.fft.ifftshift(
                    np.fft.ifft(
                        np.fft.fftshift(row)
                    )
                )
            )
        reconstructed_image = np.asarray(reconstructed_image)
        reconstructed_image_cropped = crop_recon(reconstructed_image)

        # fig, ax = plt.subplots()
        # im = ax.imshow(np.real(reconstructed_image_cropped), aspect='auto')
        # plt.colorbar(im)
        # plt.show()
        self.recon = reconstructed_image_cropped

    def _dfr(self, fourier_slices):
        angles = self.angles
        num_proj = len(fourier_slices)
        e = (num_proj - 1) // 2
        radii = np.arange(-e, e + 1)

        image_prime = np.zeros((e, e))

        cart_xy = []
        z = []
        for i, angle in enumerate(angles):
            print(i)
            radius = radii[i]
            cart_xy.append(pol2cart(radius, angle))
            z.append(fourier_slices[:, i])

        x, y = np.asarray(cart_xy)[:, 0], np.asarray(cart_xy)[:, 1]
        x, y = [int(i) + e + 1 for i in x], [int(i) + e + 1 for i in y]
        print(x)
        breakpoint()
        for i in range(len(fourier_slices)):
            # image_prime[x[i], y[i]] = z[i]
            print(x[i])
            print(y[i])
            # print(z[i])

        # print(image_prime)
        breakpoint()

    def _sinc_transform(self):
        sinogram = self.sinogram

        for i, row in enumerate(sinogram):
            sinogram[i] = np.sinc(row)

        self.sinogram = sinogram

    def _fourier_transform_1d(self):
        sinogram = self.sinogram

        ft_1d = []
        # for row in sinogram:
        #     ft = np.fft.fft2(np.fft.ifftshift(row))
        #     ft_1d.append(np.fft.fftshift(ft))

        ft = np.fft.fftshift(
            np.fft.fft2(np.fft.ifftshift(sinogram))
        )

        ift = np.fft.fftshift(
            np.fft.ifft2(np.fft.ifftshift(ft))
        )

        fig, ax = plt.subplots(2, 1)
        fig.suptitle('Fourier Shift Transform')
        ax[0].imshow(np.real(ft))
        ax[1].imshow(np.real(ift))
        # plt.imshow(np.real(ift))
        plt.tight_layout()
        plt.show()

    def _fourier_transforms(self):
        n = self.sinogram_exposures
        s = self.sinogram_resolution
        sinogram = self.sinogram

        sinogram_fft_rows = fftpack.fftshift(fftpack.fft(
            fftpack.ifftshift(sinogram, axes=1)), axes=1
        )

        plt.figure()
        plt.imshow(np.imag(sinogram_fft_rows))
        plt.show()

        ith_angles = np.array([self._ith_angle(i) for i in range(n)])
        radii = np.arange(s) - s / 2
        r, a = np.meshgrid(radii, ith_angles)
        r = r.flatten()
        a = a.flatten()
        srcx = (s / 2) + r * np.cos(a)
        srcy = (s / 2) + r * np.sin(a)

        dstx, dsty = np.meshgrid(np.arange(s), np.arange(s))
        dstx = dstx.flatten()
        dsty = dsty.flatten()

        self.srcx, self.srcy = srcx, srcy
        self.dstx, self.dsty = dstx, dsty
        self.sinogram_fft_rows = sinogram_fft_rows

        # plt.figure()
        # plt.title("Sinogram samples in 2D FFT (abs)")
        # plt.scatter(
        #     srcx,
        #     srcy,
        #     c=np.absolute(sinogram_fft_rows.flatten()),
        #     marker=".",
        #     edgecolor="none",
        # )
        # plt.show()

    def _slice_theorem(self):
        s = self.sinogram_resolution
        fft2 = griddata(
            (self.srcy, self.srcx),
            self.sinogram_fft_rows.flatten(),
            (self.dsty, self.dstx),
            method="cubic",
            fill_value=0.0,
        ).reshape((s, s))

        self.fft2 = fft2

        plt.figure()
        plt.suptitle("FFT2 space")
        plt.subplot(121)
        plt.title("Recon (real)")
        plt.imshow(np.real(fft2))
        plt.subplot(122)
        plt.title("Recon (imag)")
        plt.imshow(np.imag(fft2))
        plt.tight_layout()
        plt.show()

    def _reconstruction_from_fourier(self):
        fft2 = self.fft2

        recon = np.real(
            fftpack.fftshift(fftpack.ifft2(fftpack.ifftshift(fft2)))
        )

        plt.figure()
        plt.imshow(recon)
        plt.title('Oxygen Concentration Contour')
        plt.show()

    def _ith_angle(self, i):
        return (math.pi * i) / self.sinogram_exposures


def pad_projections(projections):
    matrix_size = max(np.shape(projections))

    if np.shape(projections)[0] == np.shape(projections)[1]:
        projections_padded = projections

    elif matrix_size == np.shape(projections)[1]:
        projections_padded = np.zeros((matrix_size, matrix_size))
        mid_point = len(projections_padded) // 2
        insert_start = mid_point - len(projections) // 2

        for i, row in enumerate(projections):
            projections_padded[insert_start + i, :] = row

    else:
        padding_needed = abs(np.diff(np.shape(projections))[0])
        pad_left = padding_needed // 2
        pad_right = padding_needed - pad_left
        projections_padded = np.zeros((matrix_size, matrix_size))

        for i, row in enumerate(projections):
            projections_padded[i] = np.pad(row, (pad_left, pad_right), 'constant')

    return projections_padded


def create_ramp_filter(sinogram_size, show=False):
    left_half = sinogram_size // 2
    right_half = sinogram_size - left_half

    ramp_filter = np.concatenate(
        (np.linspace(-left_half, 0, left_half, endpoint=False),
         np.linspace(0, right_half + 1, right_half + 1, endpoint=False))
    )
    ramp_filter = abs(ramp_filter) / max(ramp_filter)
    ramp_filter_x = np.linspace(-left_half, right_half + 1, sinogram_size + 1, endpoint=False)

    if show:
        plt.plot(ramp_filter_x, ramp_filter, 'r-.')
        plt.title(f'ramp filter for sinogram of size {sinogram_size}')
        plt.show()

    # ramp_filter_2d = np.outer(ramp_filter, ramp_filter)
    # print(np.shape(ramp_filter_x))
    # print(np.shape(ramp_filter_2d))
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # for row in ramp_filter_2d:
    #     ax.plot(ramp_filter_x, ramp_filter_x, row)
    # plt.show()
    # breakpoint()
    return ramp_filter


def crop_recon(image):
    cut_locations = np.zeros(shape=(np.shape(image)[0], 2))

    for i, row in enumerate(image):
        ratios = []
        for j, val in enumerate(np.real(row)):
            if j == 0:
                continue

            prev_val = np.real(row[j - 1])
            ratios.append(abs(val - prev_val))
        cut_locations[i] = i, get_max_loc(ratios, max(ratios))

    first_cut = get_cuts(cut_locations[:, 1], min)
    last_cut = get_cuts(cut_locations[:, 1], max)

    image_cropped = np.asarray(image)[:, first_cut:last_cut]
    projection_axis = np.linspace(0, 65, last_cut - first_cut, endpoint=True)

    return image_cropped


def get_cuts(locs, kwarg):
    try:
        cut = kwarg(locs)
    except ValueError:
        cut = kwarg(locs)[0]

    cut = cut - 20 if kwarg == min else cut + 20
    return int(cut)


def get_max_loc(val_list, max_val):
    max_loc = np.where(val_list == max_val)[0]
    if len(max_loc) != 1:
        max_loc = max_loc[0]
    return max_loc


def pol2cart(radius, angle_radians):
    x = radius * np.cos(angle_radians)
    y = radius * np.sin(angle_radians)

    return x, y


def inverse_oxy_conc_calc(recon, pathlengths, cal_constant):
    # recon_oxy_conc = np.zeros(shape=np.shape(recon))
    recon_oxy_conc = np.zeros(shape=(np.shape(recon)[0], 1))

    for i, row in enumerate(recon):
        pathlength = pathlengths[i]
        peak = max(row)
        recon_oxy_conc[i, 0] = gasmas_o2_calc(peak, cal_constant, pathlength)
        # for j, val in enumerate(row):
        #     recon_oxy_conc[i, j] = gasmas_o2_calc(val, cal_constant, pathlength)

    return recon_oxy_conc


if __name__ == '__main__':
    pass
