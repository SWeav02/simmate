# -*- coding: utf-8 -*-
try:
    import pybader
except:
    raise Exception(
        "Missing app-specific dependencies. Make sure to read our installation guide."
        "The `badelf` app requires one additional dependency: `pybader`."
        "Install these with `conda install -c conda-forge pybader`"
    )
