from typing import Any, Dict, Optional

from django.db.models import QuerySet

from simmate.website.unicorn.components import UnicornView
from simmate.website.unicorn.decorators import timed

# BUG: request data should never be updated. Making it immutible
# lets us look backwards and see the original request data and run checks along
# the way. So I remove passing the 'data'  kwarg in calls to this util
# (such as in SyncInput). I don't see any other areas this is used, but to be
# safe, I leave the data kwarg here for now - @jacksund

# TODO: convert set_property_value util to a Component method


@timed
def set_property_value(
    component: UnicornView,
    property_name: str,
    property_value: Any,
    data: Optional[Dict] = None,
) -> None:
    """
    Sets properties on the component.
    Also updates the data dictionary which gets set back as part of the payload.

    Args:
        param component: Component to set attributes on.
        param property_name: Name of the property.
        param property_value: Value to set on the property.
        param data: Dictionary that gets sent back with the response. Defaults to {}.
    """

    if property_name is None:
        raise AssertionError("Property name is required")
    if property_value is None:
        raise AssertionError("Property value is required")

    if not data:
        data = {}

    component.updating(property_name, property_value)

    """
    Handles nested properties. For example, for the following component:

    class Author(UnicornField):
        name = "Neil"

    class TestView(UnicornView):
        author = Author()

    `payload` would be `{'name': 'author.name', 'value': 'Neil Gaiman'}`

    The following code updates UnicornView.author.name based the payload's `author.name`.
    """
    property_name_parts = property_name.split(".")
    component_or_field = component
    data_or_dict = data  # Could be an internal portion of data that gets set

    for idx, property_name_part in enumerate(property_name_parts):
        if hasattr(component_or_field, property_name_part):
            if idx == len(property_name_parts) - 1:
                if hasattr(component_or_field, "_set_property"):
                    # Can assume that `component_or_field` is a component
                    component_or_field._set_property(
                        property_name_part,
                        property_value,
                        call_updating_method=False,
                        call_updated_method=True,
                    )
                else:
                    # Handle calling the updating/updated method for nested properties
                    property_name_snake_case = property_name.replace(".", "_")
                    updating_function_name = f"updating_{property_name_snake_case}"
                    updated_function_name = f"updated_{property_name_snake_case}"

                    if hasattr(component, updating_function_name):
                        getattr(component, updating_function_name)(property_value)

                    is_relation_field = False

                    # Set the id property for ForeignKeys
                    # TODO: Move some of this to utility function
                    if hasattr(component_or_field, "_meta"):
                        for field in component_or_field._meta.get_fields():
                            if field.is_relation and field.many_to_many:
                                related_name = field.name

                                if field.auto_created:
                                    related_name = (
                                        field.related_name or f"{field.name}_set"
                                    )

                                if related_name == property_name_part:
                                    related_descriptor = getattr(
                                        component_or_field, related_name
                                    )
                                    related_descriptor.set(property_value)
                                    is_relation_field = True
                                    break
                            elif field.name == property_name_part:
                                if field.is_relation:
                                    setattr(
                                        component_or_field,
                                        field.attname,
                                        property_value,
                                    )
                                    is_relation_field = True
                                    break

                    if not is_relation_field:
                        setattr(component_or_field, property_name_part, property_value)

                    if hasattr(component, updated_function_name):
                        getattr(component, updated_function_name)(property_value)

                data_or_dict[property_name_part] = property_value
            else:
                component_or_field = getattr(component_or_field, property_name_part)
                data_or_dict = data_or_dict.get(property_name_part, {})
        elif isinstance(component_or_field, dict):
            if idx == len(property_name_parts) - 1:
                component_or_field[property_name_part] = property_value
                data_or_dict[property_name_part] = property_value
            else:
                component_or_field = component_or_field[property_name_part]
                data_or_dict = data_or_dict.get(property_name_part, {})
        elif isinstance(component_or_field, (QuerySet, list)):
            # TODO: Check for iterable instead of list? `from collections.abc import Iterable`
            property_name_part_int = int(property_name_part)

            if idx == len(property_name_parts) - 1:
                component_or_field[property_name_part_int] = property_value
                data_or_dict[property_name_part_int] = property_value
            else:
                component_or_field = component_or_field[property_name_part_int]
                data_or_dict = data_or_dict[property_name_part_int]
        else:
            break

    component.updated(property_name, property_value)
