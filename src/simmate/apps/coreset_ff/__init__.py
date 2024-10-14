# -*- coding: utf-8 -*-

# ensure coreset's optional deps are present
try:
    import dscribe
except:
    raise Exception(
        "This app requires the dscribe package. "
        "Install this with `conda install -c conda-forge dscribe`"
    )
