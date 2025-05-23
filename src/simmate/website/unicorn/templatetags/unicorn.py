from typing import Dict, List, Optional

import shortuuid
from django import template
from django.conf import settings
from django.template.base import FilterExpression

from simmate.website.unicorn.call_method_parser import InvalidKwargError, parse_kwarg
from simmate.website.unicorn.errors import ComponentNotValidError
from simmate.website.unicorn.settings import get_morpher_settings

register = template.Library()


MINIMUM_ARGUMENT_COUNT = 2


@register.inclusion_tag("unicorn/scripts.html")
def unicorn_scripts():
    # Import here to prevent the potential of this loading before Django settings
    from simmate.website.unicorn.settings import get_setting

    csrf_header_name = settings.CSRF_HEADER_NAME

    if csrf_header_name.startswith("HTTP_"):
        csrf_header_name = settings.CSRF_HEADER_NAME[5:]

    csrf_header_name = csrf_header_name.replace("_", "-")

    csrf_cookie_name = settings.CSRF_COOKIE_NAME

    return {
        "MINIFIED": get_setting("MINIFIED", not settings.DEBUG),
        "CSRF_HEADER_NAME": csrf_header_name,
        "CSRF_COOKIE_NAME": csrf_cookie_name,
        "MORPHER": get_morpher_settings(),
    }


@register.inclusion_tag("unicorn/errors.html", takes_context=True)
def unicorn_errors(context):
    return {"unicorn": {"errors": context.get("unicorn", {}).get("errors", {})}}


def unicorn(parser, token):
    contents = token.split_contents()

    if len(contents) < MINIMUM_ARGUMENT_COUNT:
        raise template.TemplateSyntaxError(
            "%r tag requires at least a single argument" % token.contents.split()[0]
        )

    component_name = parser.compile_filter(contents[1])

    args = []
    kwargs = {}
    unparseable_kwargs = {}

    for arg in contents[2:]:
        try:
            parsed_kwarg = parse_kwarg(arg, raise_if_unparseable=True)
            kwargs.update(parsed_kwarg)
        except InvalidKwargError:
            # Assume it's an arg if invalid kwarg and kwargs is empty
            if not kwargs:
                args.append(arg)
        except ValueError:
            parsed_kwarg = parse_kwarg(arg, raise_if_unparseable=False)
            unparseable_kwargs.update(parsed_kwarg)

    return UnicornNode(component_name, args, kwargs, unparseable_kwargs)


class UnicornNode(template.Node):
    def __init__(
        self,
        component_name: FilterExpression,
        args: Optional[List] = None,
        kwargs: Optional[Dict] = None,
        unparseable_kwargs: Optional[Dict] = None,
    ):
        self.component_name = component_name
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}
        self.unparseable_kwargs = (
            unparseable_kwargs if unparseable_kwargs is not None else {}
        )
        self.component_key = ""
        self.parent = None

    def render(self, context):
        request = None

        if hasattr(context, "request"):
            request = context.request

        from simmate.website.unicorn.components import UnicornView

        resolved_args = []

        for value in self.args:
            resolved_arg = template.Variable(value).resolve(context)
            resolved_args.append(resolved_arg)

        resolved_kwargs = self.kwargs.copy()

        for key, value in self.unparseable_kwargs.items():
            try:
                resolved_value = template.Variable(value).resolve(context)

                if (
                    key == "parent"
                    and value == "view"
                    and not isinstance(resolved_value, UnicornView)
                ):
                    # Handle rendering a parent component from a template that is called from
                    # a `TemplateView`; for some reason `view` is clobbered in this instance, but
                    # the `unicorn` dictionary has enough data to instantiate a `UnicornView`
                    parent_component_data = template.Variable("unicorn").resolve(
                        context
                    )

                    resolved_value = UnicornView(
                        component_name=parent_component_data.get("component_name"),
                        component_id=parent_component_data.get("component_id"),
                    )

                resolved_kwargs.update({key: resolved_value})
            except TypeError:
                resolved_kwargs.update({key: value})
            except template.VariableDoesNotExist:
                if value.endswith(".id"):
                    pk_val = value.replace(".id", ".pk")

                    try:
                        resolved_kwargs.update(
                            {key: template.Variable(pk_val).resolve(context)}
                        )
                    except TypeError:
                        resolved_kwargs.update({key: value})
                    except template.VariableDoesNotExist:
                        pass

        if "key" in resolved_kwargs:
            self.component_key = resolved_kwargs.pop("key")

        if "parent" in resolved_kwargs:
            self.parent = resolved_kwargs.pop("parent")
        else:
            # if there is no explicit parent, but this node is rendering under an existing
            # unicorn template, set that as the parent
            try:
                implicit_parent = template.Variable("unicorn.component").resolve(
                    context
                )
                if implicit_parent:
                    self.parent = implicit_parent
            except template.VariableDoesNotExist:
                pass  # no implicit parent present

        component_id = None

        try:
            component_name = self.component_name.resolve(context)
        except AttributeError as e:
            raise ComponentNotValidError(
                f"Component template is not valid: {self.component_name}."
            ) from e

        if self.parent:
            # Child components use the parent for part of the `component_id`
            component_id = f"{self.parent.component_id}:{component_name}"

            if self.component_key:
                component_id = f"{component_id}:{self.component_key}"
            elif "id" in resolved_kwargs:
                kwarg_id = resolved_kwargs["id"]
                component_id = f"{component_id}:{kwarg_id}"
            elif "pk" in resolved_kwargs:
                kwarg_pk = resolved_kwargs["pk"]
                component_id = f"{component_id}:{kwarg_pk}"
            elif "model" in resolved_kwargs:
                model = resolved_kwargs["model"]

                if hasattr(model, "pk"):
                    component_id = f"{component_id}:{model.pk}"
                elif hasattr(model, "id"):
                    component_id = f"{component_id}:{model.id}"

        component_id = (
            shortuuid.uuid(name=component_id)[:8]
            if component_id
            else shortuuid.uuid()[:8]
        )

        # Useful for unit test
        self.component_id = component_id

        self.view = UnicornView.get_or_create(
            component_id=component_id,
            component_name=component_name,
            component_key=self.component_key,
            parent=self.parent,
            request=request,
            component_args=resolved_args,
            component_kwargs=resolved_kwargs,
        )

        extra_context = {}
        for c in context:
            extra_context.update(c)

        rendered_component = self.view.render(init_js=True, extra_context=extra_context)

        return rendered_component


register.tag("unicorn", unicorn)
