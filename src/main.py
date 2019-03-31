import argparse

import tracking
from methods import optical_flow, stabilization, off_the_shelf_stabilization
from optical_flow import LucasKanade, BlockMatching, pyflow_optical_flow, lucas_kanade_dense, horn_schunck, gunner_farneback
from tracking import overlap_flow_tracking

method_refs = {
    'optical_flow': optical_flow,
    'stabilization': stabilization,
    'off_the_shelf_stabilization': off_the_shelf_stabilization,
    'tracking': tracking
}

optical_flow_refs = {
    'block_matching': BlockMatching(),
    'lucas_kanade': LucasKanade(),
    'lucas_kanade_dense': lucas_kanade_dense,
    'horn_schunck': horn_schunck,
    'gunner_farneback': gunner_farneback,
    'pyflow': pyflow_optical_flow
}

off_the_shelf_stabilization_refs = {

}

tracking_refs = {
    'overlap_flow': overlap_flow_tracking
}


def main():
    parser = argparse.ArgumentParser(description='Search the picture passed in a picture database.')

    parser.add_argument('method', help='Method to use', choices=method_refs.keys())
    parser.add_argument('optical_flow', help='Optical flow method to use', nargs='?', choices=optical_flow_refs.keys(),
                        default=None)
    parser.add_argument('-s', '--off-the-shelf-stabilization-method', choices=off_the_shelf_stabilization_refs.keys())
    parser.add_argument('-t', '--tracking-method', choices=tracking_refs.keys())
    parser.add_argument('-d', '--debug', action='store_true', help='Show debug plots')

    args = parser.parse_args()

    method = method_refs.get(args.method)
    optical_flow_method = optical_flow_refs.get(args.optical_flow)
    off_the_shelf_stabilization_method = off_the_shelf_stabilization_refs.get(args.off_the_shelf_stabilization_method)
    tracking_method = tracking_refs.get(args.tracking_method)

    method(optical_flow_method=optical_flow_method,
           debug=args.debug,
           off_the_shelf_stabilization_method=off_the_shelf_stabilization_method,
           tracking_method=tracking_method)


if __name__ == '__main__':
    main()
