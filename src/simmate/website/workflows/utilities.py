# -*- coding: utf-8 -*-

from rest_framework.generics import GenericAPIView
from rest_framework.response import Response
from rest_framework.serializers import Serializer, HyperlinkedModelSerializer

from simmate.website.workflows import filters
from simmate.database.base_data_types import DatabaseTable


class SimmateAPIView(GenericAPIView):
    # This class simply adds a method to enable passing extra context to the
    # final Response. This is only done when we are using the HTML format.

    extra_context: dict = {}
    """
    This defines extra context that should be passed to the template when
    using format=html. Note, you can have this as a constant or alternatively
    define a property. The only requirement is that a dictionary is returned.
    """

    def get_response(self, serializer: Serializer) -> Response:
        if self._format_kwarg == "html":
            data = {
                "filter": self.filterset_class(serializer.data),
                "results": serializer.data,  # would it be better to use .initial_data?
                **self.extra_context,
            }
            return Response(data)
        else:
            return Response(serializer.data)


class ListAPIView(SimmateAPIView):
    """
    Concrete view for listing a queryset.
    """

    def get(self, request, *args, **kwargs):

        # self.format_kwarg --> not sure why this always returns None, so I
        # grab the format from the request instead. If it isn't listed, then
        # I'm using the default which is html.
        self._format_kwarg = request.GET.get("format", "html")

        # ---------------------------------------------------
        # This code is from the ListModelMixin, where instead of returning
        # a response, we perform additional introspection first. I turn off
        # pagination for now but need to revisit this.
        queryset = self.filter_queryset(self.get_queryset())
        if self._format_kwarg != "html":  # <--- added this condition
            page = self.paginate_queryset(queryset)
            if page is not None:
                serializer = self.get_serializer(page, many=True)
                return self.get_paginated_response(serializer.data)
        serializer = self.get_serializer(queryset, many=True)
        # return Response(serializer.data)  <--- removed from original
        # ---------------------------------------------------
        return self.get_response(serializer)


class RetrieveAPIView(SimmateAPIView):
    """
    Concrete view for retrieving a model instance.
    """

    def get(self, request, *args, **kwargs):

        # ---------------------------------------------------
        # This code is from the RetrieveModelMixin, where instead of returning
        # a response, we perform additional introspection first
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        # return Response(serializer.data) <--- removed from original
        # ---------------------------------------------------
        return self.get_response(serializer)


def render_from_table(request, template: str, context, table: DatabaseTable):

    # NOTE: This dynamically creates a serializer and a view EVERY TIME a
    # URL is requested. This means...
    #   1. there is no pre-set api that exists. The exisiting api must be inferred
    #       from lower level workflows and their tables
    #   2. these views will be very inefficient if queried by a script
    # I chose dynamic creation over creating all endpoints on-startup to prevent
    # the `from simmate.shortcuts import setup` method from taking too long --
    # as that would require import all workflows on start-up. However, if the
    # (1) api spec or (2) speed of this method ever becomes an issue, I can
    # address these by either...
    #   1. having a utility that prints out the full API spec but isn't called on startup
    #   2. making all APIViews up-front

    # TODO: consider using the following to dynamically name these classes
    #   NewClass = type(table.__name__, mixins, extra_attributes)

    # For all tables, we share all the data -- no columns are hidden.
    class NewSerializer(HyperlinkedModelSerializer):
        class Meta:
            model = table
            fields = "__all__"

    NewFilterSet = get_filterset_from_table(table)

    # Querying each table varies though
    class NewViewSet(ListAPIView):
        queryset = table.objects.all()  # TODO: order_by("created_at") by default?
        serializer_class = NewSerializer
        template_name = template
        extra_context = context
        filterset_class = NewFilterSet

    # now pull together the html response
    response = NewViewSet.as_view()(request)
    return response


def get_filterset_from_table(table: DatabaseTable) -> filters.DatabaseTableFilter:
    """
    Dynamically creates a Django Filter from a Simmate database table.

    For example, this function would take
    `simmate.database.third_parties.MatProjStructure`
    and automatically make the following filter:

    ``` python
    from simmate.website.workflows.filters import (
        DatabaseTableFilter,
        Structure,
        Thermodynamics,
    )


    class MatProjStrucureFilter(
        DatabaseTableFilter,
        Structure,
        Thermodynamics,
    ):
        class Meta:
            model = MatProjStructure  # this is database table
            fields = {...} # this combines the fields from Structure/Thermo mixins

        # These attributed are set using the declared filters from Structure/Thermo mixins
        declared_filter1 = ...
        declared_filter1 = ...
    ```
    """

    # First we need to grab the parent mixin classes of the table. For example,
    # the MatProjStructure uses the database mixins ['Structure', 'Thermodynamics']
    # while MatProjStaticEnergy uses ["StaticEnergy"].
    mixin_names = [base.__name__ for base in table.__bases__]

    # Because our Forms follow the same naming conventions as
    # simmate.database.base_data_types, we can simply use these mixin names to
    # load a Form mixin from the simmate.website.workflows.form module. We add
    # these mixins onto the standard ModelForm class from django.
    filter_mixins = [filters.DatabaseTableFilter]
    filter_mixins += [
        getattr(filters, name) for name in mixin_names if hasattr(filters, name)
    ]

    # combine the fields of each filter mixin. Note we use .get_fields instead
    # of .fields to ensure we get a dictionary back
    filter_fields = {
        field: conditions
        for mixin in filter_mixins
        for field, conditions in mixin.get_fields().items()
    }

    # Also combine all declared filters from each mixin
    filters_declared = {
        name: filter_obj
        for mixin in filter_mixins
        for name, filter_obj in mixin.declared_filters.items()
    }

    # The FilterSet class requires that we set extra fields in the Meta class.
    # We define those here. Note, we accept all fields by default and the
    # excluded fields are only those defined by the supported form_mixins -
    # and we skip the first mixin (which is always DatabaseTableFilter)
    class Meta:
        model = table
        fields = filter_fields

    extra_attributes = {"Meta": Meta, **filters_declared}
    # BUG: __module__ may need to be set in the future, but we never import
    # these forms elsewhere, so there's no need to set it now.

    # Now we dynamically create a new form class that we can return.
    NewClass = type(table.__name__, tuple(filter_mixins), extra_attributes)

    return NewClass
