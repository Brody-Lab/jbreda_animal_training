import argparse

from DMS2_fetch_protocol_data import fetch_latest_training_data
from DMS2_utils import make_daily_stage_plots

parser = argparse.ArgumentParser()
parser.add_argument(
    "-aid",
    "--animal_id",
    nargs="+",
    default=None,
    help="animals you want to plot, default is all of cohort2",
)
parser.add_argument(
    "-dmin", "--date_min", type=str, default=None, help="minimum date to fetch data for"
)
parser.add_argument(
    "-dmax", "--date_max", type=str, default=None, help="max date to fetch data for"
)
parser.add_argument(
    "-v",
    "--verbose",
    type=bool,
    default=False,
    help="print information about dropping sessions",
)
parser.add_argument(
    "-o",
    "--overwrite",
    type=bool,
    default=False,
    help="if pre-made plots should be remade",
)


def main(args):
    # print(args.animal_id)
    # print(type(args.animal_id))
    print(args.date_min)
    print(type(args.date_min))
    # print(args.date_max)
    print(args.verbose)

    df = fetch_latest_training_data(
        animal_ids=args.animal_id,
        date_min=args.date_min,
        date_max=args.date_min,
    )

    # make_daily_stage_plots(df, overwrite=args.overwrite)


if __name__ == "__main__":
    main(parser.parse_args())
