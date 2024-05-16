import numpy as np
from tigre.utilities.geometry import Geometry


class PCDGeometry(Geometry):
    def __init__(self, DSD=578.0, DSO=322.0, side=512, z_stack=24, size_z=4.41, h_offset=0, v_offset=0):

        Geometry.__init__(self)

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = DSD  # Distance Source Detector      (mm)
        self.DSO = DSO  # Distance Source Origin        (mm)
        # Detector parameters
        self.nDetector = np.array((24, 576))  # number of pixels              (px)
        self.dDetector = np.array((0.33, 0.33))  # size of each pixel            (mm)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (mm)
        # Image parameters
        self.nVoxel = np.array((z_stack, side, side))  # number of voxels              (vx)
        self.sVoxel = np.array((size_z, 105, 105))  # total size of the image       (mm)

        self.dVoxel = self.sVoxel / self.nVoxel  # size of each voxel            (mm)
        # Offsets
        self.offOrigin = np.array((0, 0, 0))  # Offset of image from origin   (mm)
        self.offDetector = np.array((v_offset, h_offset))  # Offset of Detector            (mm)

        # Auxiliary
        self.accuracy = 0.5  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = "cone"  # parallel, cone                ...
        self.filter = None
