import numpy as np

################################################################################################
## Front Side Rotations
################################################################################################
def F(cube):
    cube[:, :, 0] = np.rot90(cube[:, :, 0], k=3)
    cube[:, :, 0, 0], cube[:, :, 0, 1] = cube[:, :, 0, 1].copy(), cube[:, :, 0, 0].copy()


def F2(cube):
    cube[:, :, 0] = np.rot90(cube[:, :, 0], k=2)


def F_(cube):
    cube[:, :, 0] = np.rot90(cube[:, :, 0], k=1)
    cube[:, :, 0, 0], cube[:, :, 0, 1] = cube[:, :, 0, 1].copy(), cube[:, :, 0, 0].copy()


################################################################################################
## Back Side Rotations
################################################################################################
def B(cube):
    cube[:, :, 2] = np.rot90(cube[:, :, 2], k=1)
    cube[:, :, 2, 0], cube[:, :, 2, 1] = cube[:, :, 2, 1].copy(), cube[:, :, 2, 0].copy()


def B2(cube):
    cube[:, :, 2] = np.rot90(cube[:, :, 2], k=2)


def B_(cube):
    cube[:, :, 2] = np.rot90(cube[:, :, 2], k=3)
    cube[:, :, 2, 0], cube[:, :, 2, 1] = cube[:, :, 2, 1].copy(), cube[:, :, 2, 0].copy()


################################################################################################
## Right Side Rotations
################################################################################################
def R(cube):
    cube[2, :, :] = np.rot90(cube[2, :, :], k=1)
    cube[2, :, :, 1], cube[2, :, :, 2] = cube[2, :, :, 2].copy(), cube[2, :, :, 1].copy()


def R2(cube):
    cube[2, :, :] = np.rot90(cube[2, :, :], k=2)


def R_(cube):
    cube[2, :, :] = np.rot90(cube[2, :, :], k=3)
    cube[2, :, :, 1], cube[2, :, :, 2] = cube[2, :, :, 2].copy(), cube[2, :, :, 1].copy()


################################################################################################
## Left Side Rotations
################################################################################################
def L(cube):
    cube[0, :, :] = np.rot90(cube[0, :, :], k=3)
    cube[0, :, :, 1], cube[0, :, :, 2] = cube[0, :, :, 2].copy(), cube[0, :, :, 1].copy()


def L2(cube):
    cube[0, :, :] = np.rot90(cube[0, :, :], k=2)


def L_(cube):
    cube[0, :, :] = np.rot90(cube[0, :, :], k=1)
    cube[0, :, :, 1], cube[0, :, :, 2] = cube[0, :, :, 2].copy(), cube[0, :, :, 1].copy()


################################################################################################
## Top Side Rotations
################################################################################################
def T(cube):
    cube[:, 2, :] = np.rot90(cube[:, 2, :], k=3)
    cube[:, 2, :, 0], cube[:, 2, :, 2] = cube[:, 2, :, 2].copy(), cube[:, 2, :, 0].copy()


def T2(cube):
    cube[:, 2, :] = np.rot90(cube[:, 2, :], k=2)


def T_(cube):
    cube[:, 2, :] = np.rot90(cube[:, 2, :], k=1)
    cube[:, 2, :, 0], cube[:, 2, :, 2] = cube[:, 2, :, 2].copy(), cube[:, 2, :, 0].copy()


################################################################################################
## Down Side Rotations
################################################################################################
def D(cube):
    cube[:, 0, :] = np.rot90(cube[:, 0, :], k=1)
    cube[:, 0, :, 0], cube[:, 0, :, 2] = cube[:, 0, :, 2].copy(), cube[:, 0, :, 0].copy()


def D2(cube):
    cube[:, 0, :] = np.rot90(cube[:, 0, :], k=2)


def D_(cube):
    cube[:, 0, :] = np.rot90(cube[:, 0, :], k=3)
    cube[:, 0, :, 0], cube[:, 0, :, 2] = cube[:, 0, :, 2].copy(), cube[:, 0, :, 0].copy()