# Generated by Django 4.2.2 on 2023-07-21 16:22

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("core_components", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="OqmdStructure",
            fields=[
                (
                    "created_at",
                    models.DateTimeField(auto_now_add=True, db_index=True, null=True),
                ),
                (
                    "updated_at",
                    models.DateTimeField(auto_now=True, db_index=True, null=True),
                ),
                ("structure", models.TextField(blank=True, null=True)),
                ("nsites", models.IntegerField(blank=True, null=True)),
                ("nelements", models.IntegerField(blank=True, null=True)),
                ("elements", models.JSONField(blank=True, null=True)),
                (
                    "chemical_system",
                    models.CharField(blank=True, max_length=25, null=True),
                ),
                ("density", models.FloatField(blank=True, null=True)),
                ("density_atomic", models.FloatField(blank=True, null=True)),
                ("volume", models.FloatField(blank=True, null=True)),
                ("volume_molar", models.FloatField(blank=True, null=True)),
                (
                    "formula_full",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                (
                    "formula_reduced",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                (
                    "formula_anonymous",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                (
                    "id",
                    models.CharField(max_length=25, primary_key=True, serialize=False),
                ),
                ("formation_energy", models.FloatField(blank=True, null=True)),
                (
                    "spacegroup",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        to="core_components.spacegroup",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="MatprojStructure",
            fields=[
                (
                    "created_at",
                    models.DateTimeField(auto_now_add=True, db_index=True, null=True),
                ),
                ("structure", models.TextField(blank=True, null=True)),
                ("nsites", models.IntegerField(blank=True, null=True)),
                ("nelements", models.IntegerField(blank=True, null=True)),
                ("elements", models.JSONField(blank=True, null=True)),
                (
                    "chemical_system",
                    models.CharField(blank=True, max_length=25, null=True),
                ),
                ("density", models.FloatField(blank=True, null=True)),
                ("density_atomic", models.FloatField(blank=True, null=True)),
                ("volume", models.FloatField(blank=True, null=True)),
                ("volume_molar", models.FloatField(blank=True, null=True)),
                (
                    "formula_full",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                (
                    "formula_reduced",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                (
                    "formula_anonymous",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                ("energy", models.FloatField(blank=True, null=True)),
                ("energy_per_atom", models.FloatField(blank=True, null=True)),
                ("energy_above_hull", models.FloatField(blank=True, null=True)),
                ("is_stable", models.BooleanField(blank=True, null=True)),
                ("decomposes_to", models.JSONField(blank=True, null=True)),
                ("formation_energy", models.FloatField(blank=True, null=True)),
                ("formation_energy_per_atom", models.FloatField(blank=True, null=True)),
                (
                    "id",
                    models.CharField(max_length=25, primary_key=True, serialize=False),
                ),
                ("energy_uncorrected", models.FloatField(blank=True, null=True)),
                ("band_gap", models.FloatField(blank=True, null=True)),
                ("is_gap_direct", models.BooleanField(blank=True, null=True)),
                ("is_magnetic", models.BooleanField(blank=True, null=True)),
                ("total_magnetization", models.FloatField(blank=True, null=True)),
                ("is_theoretical", models.BooleanField(blank=True, null=True)),
                ("updated_at", models.DateTimeField(blank=True, null=True)),
                (
                    "spacegroup",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        to="core_components.spacegroup",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="JarvisStructure",
            fields=[
                (
                    "created_at",
                    models.DateTimeField(auto_now_add=True, db_index=True, null=True),
                ),
                (
                    "updated_at",
                    models.DateTimeField(auto_now=True, db_index=True, null=True),
                ),
                ("structure", models.TextField(blank=True, null=True)),
                ("nsites", models.IntegerField(blank=True, null=True)),
                ("nelements", models.IntegerField(blank=True, null=True)),
                ("elements", models.JSONField(blank=True, null=True)),
                (
                    "chemical_system",
                    models.CharField(blank=True, max_length=25, null=True),
                ),
                ("density", models.FloatField(blank=True, null=True)),
                ("density_atomic", models.FloatField(blank=True, null=True)),
                ("volume", models.FloatField(blank=True, null=True)),
                ("volume_molar", models.FloatField(blank=True, null=True)),
                (
                    "formula_full",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                (
                    "formula_reduced",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                (
                    "formula_anonymous",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                (
                    "id",
                    models.CharField(max_length=25, primary_key=True, serialize=False),
                ),
                ("energy_above_hull", models.FloatField(blank=True, null=True)),
                (
                    "spacegroup",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        to="core_components.spacegroup",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="CodStructure",
            fields=[
                (
                    "created_at",
                    models.DateTimeField(auto_now_add=True, db_index=True, null=True),
                ),
                (
                    "updated_at",
                    models.DateTimeField(auto_now=True, db_index=True, null=True),
                ),
                ("structure", models.TextField(blank=True, null=True)),
                ("nsites", models.IntegerField(blank=True, null=True)),
                ("nelements", models.IntegerField(blank=True, null=True)),
                ("elements", models.JSONField(blank=True, null=True)),
                ("density", models.FloatField(blank=True, null=True)),
                ("density_atomic", models.FloatField(blank=True, null=True)),
                ("volume", models.FloatField(blank=True, null=True)),
                ("volume_molar", models.FloatField(blank=True, null=True)),
                ("chemical_system", models.TextField()),
                ("formula_full", models.TextField()),
                ("formula_reduced", models.TextField()),
                ("formula_anonymous", models.TextField()),
                (
                    "id",
                    models.CharField(max_length=25, primary_key=True, serialize=False),
                ),
                ("is_ordered", models.BooleanField()),
                ("has_implicit_hydrogens", models.BooleanField()),
                (
                    "spacegroup",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        to="core_components.spacegroup",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="AflowStructure",
            fields=[
                (
                    "created_at",
                    models.DateTimeField(auto_now_add=True, db_index=True, null=True),
                ),
                (
                    "updated_at",
                    models.DateTimeField(auto_now=True, db_index=True, null=True),
                ),
                ("structure", models.TextField(blank=True, null=True)),
                ("nsites", models.IntegerField(blank=True, null=True)),
                ("nelements", models.IntegerField(blank=True, null=True)),
                ("elements", models.JSONField(blank=True, null=True)),
                (
                    "chemical_system",
                    models.CharField(blank=True, max_length=25, null=True),
                ),
                ("density", models.FloatField(blank=True, null=True)),
                ("density_atomic", models.FloatField(blank=True, null=True)),
                ("volume", models.FloatField(blank=True, null=True)),
                ("volume_molar", models.FloatField(blank=True, null=True)),
                (
                    "formula_full",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                (
                    "formula_reduced",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                (
                    "formula_anonymous",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                ("energy", models.FloatField(blank=True, null=True)),
                ("energy_per_atom", models.FloatField(blank=True, null=True)),
                ("energy_above_hull", models.FloatField(blank=True, null=True)),
                ("is_stable", models.BooleanField(blank=True, null=True)),
                ("decomposes_to", models.JSONField(blank=True, null=True)),
                ("formation_energy", models.FloatField(blank=True, null=True)),
                ("formation_energy_per_atom", models.FloatField(blank=True, null=True)),
                (
                    "id",
                    models.CharField(max_length=25, primary_key=True, serialize=False),
                ),
                (
                    "spacegroup",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        to="core_components.spacegroup",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="AflowPrototype",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "created_at",
                    models.DateTimeField(auto_now_add=True, db_index=True, null=True),
                ),
                (
                    "updated_at",
                    models.DateTimeField(auto_now=True, db_index=True, null=True),
                ),
                ("structure", models.TextField(blank=True, null=True)),
                ("nsites", models.IntegerField(blank=True, null=True)),
                ("nelements", models.IntegerField(blank=True, null=True)),
                ("elements", models.JSONField(blank=True, null=True)),
                (
                    "chemical_system",
                    models.CharField(blank=True, max_length=25, null=True),
                ),
                ("density", models.FloatField(blank=True, null=True)),
                ("density_atomic", models.FloatField(blank=True, null=True)),
                ("volume", models.FloatField(blank=True, null=True)),
                ("volume_molar", models.FloatField(blank=True, null=True)),
                (
                    "formula_full",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                (
                    "formula_reduced",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                (
                    "formula_anonymous",
                    models.CharField(blank=True, max_length=50, null=True),
                ),
                (
                    "mineral_name",
                    models.CharField(blank=True, max_length=75, null=True),
                ),
                ("aflow_id", models.CharField(max_length=30)),
                ("pearson_symbol", models.CharField(max_length=6)),
                (
                    "strukturbericht_symbol",
                    models.CharField(blank=True, max_length=6, null=True),
                ),
                ("nsites_wyckoff", models.IntegerField()),
                (
                    "spacegroup",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        to="core_components.spacegroup",
                    ),
                ),
            ],
        ),
    ]
