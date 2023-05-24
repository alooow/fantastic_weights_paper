from argparse import Namespace
import mrunner
from mrunner.helpers.client_helper import get_configuration

from main import main
mrunner.settings.NEPTUNE_USE_NEW_API = True

if __name__ == "__main__":
    params = get_configuration(
        print_diagnostics=True, with_neptune=False, inject_parameters_to_gin=False
    )
    params.pop("experiment_id")

    main(Namespace(**params))
