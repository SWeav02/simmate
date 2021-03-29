# -*- coding: utf-8 -*-

"""
- write 1st rough static flow
- write 1st rough NEB flow --> and model for relaxed path
- test NEB run
- launch 1st rough static flows (wait 2 days before launching NEBs)
- inspect database for known flouride conductors and prototype plots
- outline writing & figures
"""

# --------------------------------------------------------------------------------------


# from dask.distributed import Client
# from simmate.workflows.diffusion.empirical_measures import workflow
# from simmate.configuration.django import setup_full  # ensures setup
# from simmate.database.diffusion import Pathway as Pathway_DB

# # grab the pathway ids that I am going to submit
# pathway_ids = (
#     Pathway_DB.objects.filter(empiricalmeasures__isnull=True)
#     .order_by("structure__nsites", "nsites_777")
#     .values_list("id", flat=True)
#     .all()[:3000]  # if I want to limit the number I submit at a time
# )

# # setup my Dask cluster and connect to it. Make sure I have each work connect to
# # the database before starting
# client = Client(preload="simmate.configuration.dask.init_django_worker")

# # Run the find_paths workflow for each individual id
# client.map(
#     workflow.run,
#     [{"pathway_id": id} for id in pathway_ids],
#     pure=False,
# )

# --------------------------------------------------------------------------------------

from prefect import Client
from simmate.configuration.django import setup_full  # ensures setup
from simmate.database.diffusion import Pathway as Pathway_DB

# grab the pathway ids that I am going to submit
pathway_ids = (
    Pathway_DB.objects.filter(
        vaspcalca__isnull=True,
        empiricalmeasures__dimensionality__gte=1,
        # empiricalmeasures__oxidation_state=-1,
        # empiricalmeasures__ionic_radii_overlap_cations__gt=-1,
        # empiricalmeasures__ionic_radii_overlap_anions__gt=-1,
        # nsites_777__lte=150,
        # structure__nsites__lte=20,
    )
    .order_by("nsites_777", "structure__nsites", "length")
    # BUG: distinct() doesn't work for sqlite, only postgres. also you must have
    # "structure__id" as the first flag in order_by for this to work.
    # .distinct("structure__id")
    .values_list("id", flat=True)
    .all()[:1000]
)

# connect to Prefect Cloud
client = Client()

# submit a run for each pathway
for pathway_id in pathway_ids:
    client.create_flow_run(
        flow_id="5422b96d-fbbe-4f61-820f-dec934a2dd6b",
        parameters={"pathway_id": pathway_id},
    )

# --------------------------------------------------------------------------------------


# from simmate.configuration.django import setup_full  # ensures setup
# from simmate.database.diffusion import EmpiricalMeasures
# queryset = EmpiricalMeasures.objects.all()
# from django_pandas.io import read_frame
# df = read_frame(queryset) # , index_col="pathway": df = df.rese

# from simmate.configuration.django import setup_full  # ensures setup
# from simmate.database.diffusion import VaspCalcA
# queryset = VaspCalcA.objects.all()
# from django_pandas.io import read_frame
# df = read_frame(queryset) # , index_col="pathway": df = df.rese

# # from dask.distributed import Client
# # client = Client(preload="simmate.configuration.dask.init_django_worker")


# set the executor to a locally ran executor
# from prefect.executors import DaskExecutor
# workflow.executor = DaskExecutor(address="tcp://152.2.172.72:8786")