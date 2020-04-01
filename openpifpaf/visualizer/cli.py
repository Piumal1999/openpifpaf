from .base import BaseVisualizer
from .caf import Caf
from .cif import Cif
from .cifhr import CifHr
from .occupancy import Occupancy
from .seeds import Seeds


def cli(parser):
    group = parser.add_argument_group('pose visualizer')
    group.add_argument('--debug-cifhr', default=False, action='store_true')
    group.add_argument('--debug-cif-c', default=False, action='store_true')
    group.add_argument('--debug-cif-v', default=False, action='store_true')
    group.add_argument('--debug-caf-c', default=False, action='store_true')
    group.add_argument('--debug-caf-v', default=False, action='store_true')

    group.add_argument('--debug-indices', default=[], nargs='+',
                       help=('indices of fields to create debug plots for '
                             'of the form headname:fieldindex, e.g. cif:5'))


def configure(args):
    # configure visualizer
    args.debug_indices = [di.partition(':') for di in args.debug_indices]
    args.debug_indices = [(di[0], int(di[2])) for di in args.debug_indices]
    BaseVisualizer.all_indices = args.debug_indices

    Caf.show_confidences = args.debug_caf_c
    Caf.show_regressions = args.debug_caf_v
    Cif.show_confidences = args.debug_cif_c
    Cif.show_regressions = args.debug_cif_v
    CifHr.show = args.debug_cifhr

    if args.debug:
        Cif.show_background = True
        Cif.show_confidences = True
        Cif.show_regressions = True
        Caf.show_background = True
        Caf.show_confidences = True
        Caf.show_regressions = True
        Occupancy.show = True
        Seeds.show = True
