import json
import logging
import sys

from cerebras.sdk.client import SdkCompiler
from cerebras.sdk.client import _version as sdk_version

from ._cslc_wsclust_shim_parse_args import cslc_wsclust_shim_parse_args
from ._print_tree import print_tree

_logger = logging.getLogger(__name__)

if __name__ == "__main__":

    logging.basicConfig(
        datefmt="%Y-%m-%d %H:%M:%S",
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
    )

    command, *args = sys.argv
    _logger.info(f"command: {args}")
    _logger.info(f"raw args: {args}")

    if len(args) == 0:
        print("Wrapper to run cslc CLI on Cerebras Wafer-Scale Cluster.")
        print("Usage: cslc_wsclust_shim.py [cslc args]")
        sys.exit(0)

    target, flags = cslc_wsclust_shim_parse_args(args)

    # adapted from https://sdk.cerebras.net/appliance-mode#compiling
    # Instantiate copmiler using a context manager
    with SdkCompiler(disable_version_check=True) as compiler:
        app_path: str = "."
        csl_main: str = target
        options: str = flags
        out_path: str = "."

        # Launch compile job
        _logger.info("compiling...")
        _logger.info(f"    sdk version: {sdk_version.__version__}")
        _logger.info(f"    {app_path=}")
        _logger.info(f"    {csl_main=}")
        _logger.info(f"    {options=}")
        _logger.info(f"    {out_path=}")
        print_tree(app_path, ".csl")

        artifact_path = compiler.compile(app_path, csl_main, options, out_path)
        _logger.info(f"...done! artifact_path: {artifact_path}")

        # Write the artifact_path to a JSON file
        with open("artifact_path.json", "w", encoding="utf8") as f:
            json.dump({"artifact_path": artifact_path}, f)
        _logger.info("saved artifact_path to artifact_path.json")
