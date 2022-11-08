from typing import Union, Dict, Optional, Tuple
import argparse
import re
import os
import pickle

import geneparse

from . import ZarrBackend, NumpyBackend


RE_FILE_EXT = re.compile(r"\.([A-Za-z0-9]+)(\.gz)?$")
TUPLE_REGEX = re.compile(r"\(((.+),)+(.+)\)")


PrimitiveArg = Union[
    str,
    int,
    float,
    Tuple[int, ...],
    Tuple[float, ...]
]


def _parse_tuple(s: str) -> Union[Tuple[int, ...], Tuple[float, ...]]:
    elements = [e.strip() for e in s.strip("()").split(",")]
    while True:
        try:
            int(elements[0])
            t: type = int
            break
        except ValueError:
            pass

        try:
            float(elements[0])
            t = float
            break
        except ValueError:
            pass

        raise ValueError("Only support for int and float in tuple parsing.")

    return tuple(t(e) for e in elements)


def _cli_args_to_kwargs(s: Optional[str]) -> Dict[str, PrimitiveArg]:
    kwargs: Dict[str, PrimitiveArg] = {}

    if s is None:
        return kwargs

    for arg in s.split(";"):
        arg = arg.strip()

        key, value = arg.split("=")

        key = key.strip()
        value = value.strip()

        if TUPLE_REGEX.match(value):
            kwargs[key] = _parse_tuple(value)

        else:
            cast_val: PrimitiveArg = value
            try:
                cast_val = int(value)
            except ValueError:
                try:
                    cast_val = float(value)
                except ValueError:
                    pass

            kwargs[key] = cast_val

    return kwargs


def _create_backend_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", "-f", choices=["plink", "bgen"],
                        required=True)
    parser.add_argument("filename", type=str)
    parser.add_argument(
        "--genotype-reader-args", type=str, default=None,
        help="Arguments that will be passed to the genotype reader. The "
             "format to provide arguments is: 'key1=val2;key2=val2,...'"
    )
    parser.add_argument("--backend-format", choices=["zarr", "numpy"],
                        default="zarr")
    parser.add_argument(
        "--backend-args", type=str, default=None,
        help="Arguments that will be passed to the backend constructor. The "
             "format to provide arguments is: 'key1=val2;key2=val2,...'"
    )
    parser.add_argument("--output-prefix", "-o", default=None, type=str)
    args = parser.parse_args()

    # Use input filename without ext as prefix if no user provided prefix.
    if args.output_prefix is None:
        if RE_FILE_EXT.search(args.filename):
            args.output_prefix = re.sub(RE_FILE_EXT, "", args.filename)
        else:
            # Filename is already a prefix (e.g. for plink)
            args.output_prefix = args.filename

    return args


def create_backend():
    args = _create_backend_parse_args()

    reader = geneparse.parsers[args.format](
        args.filename,
        **_cli_args_to_kwargs(args.genotype_reader_args)
    )

    backend_args = _cli_args_to_kwargs(args.backend_args)

    work_dir = os.path.dirname(args.output_prefix)
    args.output_prefix = os.path.basename(args.output_prefix)

    if work_dir != "":
        os.chdir(work_dir)

    if args.backend_format == "zarr":
        _create_zarr_backend(reader, args.output_prefix, backend_args)
    elif args.backend_format == "numpy":
        _create_numpy_backend(reader, args.output_prefix, backend_args)
    else:
        raise ValueError("Only zarr and numpy backends are supported.")


def _create_zarr_backend(
    reader: geneparse.core.GenotypesReader,
    prefix: str,
    backend_kwargs: Dict[str, PrimitiveArg]
):
    backend = ZarrBackend(reader, prefix, **backend_kwargs)  # type: ignore

    with open(f"{prefix}.pkl", "wb") as f:
        pickle.dump(backend, f)


def _create_numpy_backend(
    reader: geneparse.core.GenotypesReader,
    prefix: str,
    backend_kwargs: Dict[str, PrimitiveArg]
):
    backend = NumpyBackend(
        reader,
        f"{prefix}.npz",
        **backend_kwargs  # type: ignore
    )

    with open(f"{prefix}.pkl", "wb") as f:
        pickle.dump(backend, f)
