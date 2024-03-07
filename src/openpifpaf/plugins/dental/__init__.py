import openpifpaf

from . import dental_kp


def register():
    openpifpaf.DATAMODULES['dental'] = dental_kp.DentalKp
