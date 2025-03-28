# Generated by Django 4.2.7 on 2025-03-10 16:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("workflows", "0001_initial"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="fixedcompositionsearch",
            name="best_survival_cutoff",
        ),
        migrations.RemoveField(
            model_name="fixedcompositionsearch",
            name="convergence_cutoff",
        ),
        migrations.RemoveField(
            model_name="fixedcompositionsearch",
            name="expected_structure",
        ),
        migrations.RemoveField(
            model_name="fixedcompositionsearch",
            name="max_structures",
        ),
        migrations.RemoveField(
            model_name="fixedcompositionsearch",
            name="min_structures_exact",
        ),
        migrations.RemoveField(
            model_name="variablensitescompositionsearch",
            name="best_survival_cutoff",
        ),
        migrations.RemoveField(
            model_name="variablensitescompositionsearch",
            name="convergence_cutoff",
        ),
        migrations.RemoveField(
            model_name="variablensitescompositionsearch",
            name="expected_structure",
        ),
        migrations.RemoveField(
            model_name="variablensitescompositionsearch",
            name="max_structures",
        ),
        migrations.RemoveField(
            model_name="variablensitescompositionsearch",
            name="min_structures_exact",
        ),
        migrations.AddField(
            model_name="fixedcompositionsearch",
            name="stop_conditions",
            field=models.JSONField(blank=True, default=dict, null=True),
        ),
        migrations.AddField(
            model_name="variablensitescompositionsearch",
            name="stop_conditions",
            field=models.JSONField(blank=True, default=dict, null=True),
        ),
    ]
