#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Command line tool that executes quality assurance checks on grid data
    derived from multibeam echosounder data.
"""

import click
import json
import os
import sys

from pathlib import Path

from ausseabed.mbesgc.lib.data import get_input_details, \
    inputs_from_qajson_checks, qajson_from_inputs
from ausseabed.mbesgc.lib.executor import Executor
from ausseabed.mbesgc.lib.allchecks import all_checks
from ausseabed.qajson.parser import QajsonParser


@click.command()
@click.option(
    '-i', '--input',
    required=False,
    help='Path to input QA JSON file')
@click.option(
    '-gf', '--grid-file',
    required=False,
    help='Path to input grid file (.tif, .bag)')
def cli(
        input,
        grid_file):
    '''Run quality assurance check over input grid file'''

    qajson = None
    qajson_folder = None

    if grid_file is not None:
        if not os.path.isfile(grid_file):
            click.echo(
                "Grid file ({}) does not exist".format(grid_file),
                err=True)
            sys.exit(os.EX_NOINPUT)

        inputs = get_input_details([grid_file])
        qajson = qajson_from_inputs(
            input=inputs[0],
            check_classes=all_checks
        )

    elif input is not None:
        if not os.path.isfile(input):
            click.echo(
                "Input file ({}) does not exist".format(input),
                err=True)
            sys.exit(os.EX_NOINPUT)
        qajson_folder = os.path.dirname(input)
        qajson = QajsonParser(path=Path(input)).root

    else:
        click.echo(
            "'-input' or '--grid-file' command line arg must be provided")
        sys.exit(os.EX_NOINPUT)

    spdatachecks = qajson.qa.survey_products.checks
    inputs = inputs_from_qajson_checks(spdatachecks, qajson_folder)

    exe = Executor(inputs, all_checks)

    def print_prog(progress):
        click.echo(f"progress = {progress}")
    exe.run(print_prog)

    for check_id, check in exe.check_result_cache.items():
        output = check.get_outputs()
        output_dict = output.to_dict()
        print(json.dumps(output_dict, indent=4))


if __name__ == '__main__':
    cli()
