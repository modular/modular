<!-- markdownlint-disable -->
{# Print YAML front matter #}
{% import 'macros.jinja' as macros %}
{% set api_path = "/mojo" %}
{% macro print_front_matter(decl) %}
---
title: {{ decl.name }}
{% if decl.sidebar_label %}sidebar_label: {{ decl.sidebar_label }}
{% endif %}
version: {{ decl.version }}
slug: {{ decl.slug }}
type: {{ decl.kind }}
namespace: {{ decl.namespace }}
lang: mojo
description: {% if decl.summary
  %}"{{ macros.escape_quotes(decl.summary) }}"
  {% else %}"Mojo {{ decl.kind }} `{{ decl.namespace }}.{{ decl.name }}` documentation"
  {% endif %}
---

<section class='mojo-docs'>

{% endmacro -%}
{# Print each declaration #}
{% macro process_decl_body(decl) %}
{% if decl.signature or decl.convention %}
<div class="mojo-function-sig">

{% if decl.convention %}
`{{decl.convention}}`
{% endif %}
{% if decl.signature %}
`{% if decl.isStatic %}static {% endif %}{{ decl.signature }}`
{% endif %}

</div>
{% endif %}

{{ decl.summary }}

{{ decl.description }}

{% if decl.constraints %}

**Constraints:**

{{ decl.constraints }}
{% endif %}
{% if decl.parameters and decl.kind == 'function' %}

**Parameters:**

{% for param in decl.parameters -%}
*   ​<b>{{ param.name }}</b> ({% if param.traits -%}
        {%- for trait in param.traits -%}
            {%- if trait.path -%}
                [`{{ trait.type }}`]({{ api_path }}{{ trait.path }})
            {%- else -%}
                `{{ trait.type }}`
            {%- endif -%}
            {%- if not loop.last %} & {% endif -%}
        {%- endfor -%}
    {%- else -%}
        {%- if param.path -%}
            [`{{ param.type }}`]({{ api_path }}{{ param.path }})
        {%- else -%}
            `{{ param.type }}`
        {%- endif -%}
    {%- endif %}): {{ param.description }}
{% endfor %}
{% endif %}
{% if decl.args %}

**Args:**

{% for arg in decl.args -%}
*   ​<b>{{ arg.name }}</b> ({% if arg.path
        %}[`{{ arg.type }}`]({{ api_path }}{{ arg.path }}){% else
        %}`{{ arg.type }}`{% endif %}): {{ arg.description }}
{% endfor %}
{% endif %}
{% if (decl.returns and decl.returns.type != 'Self') or (decl.returns and decl.returns.doc) %}
{# Don't show "Returns" if the type is Self, unless there's a docstring #}

**Returns:**

{% if decl.returns.path
  %}[`{{ decl.returns.type }}`]({{ api_path }}{{ decl.returns.path }}){% else
  %}`{{ decl.returns.type }}`{% endif %}{% if decl.returns.doc
    %}: {{ decl.returns.doc }}{% endif %}
{% endif %}
{% if decl.raisesDoc %}

**Raises:**

{{ decl.raisesDoc }}
{% endif %}
{% endmacro %}
{#############}
{# Main loop #}
{#############}
{% for decl in decls recursive %}
{% if loop.depth == 1 %}
{{ print_front_matter(decl) }}
{% else %}
{{ "#"*(loop.depth+1) }} `{{ decl.name }}`

{% endif %}
{% if decl.overloads %}
{% for overload in decl.overloads %}
<div class='mojo-function-detail'>

{{ process_decl_body(overload) }}

</div>

{% endfor %}
{% else %}

{{ process_decl_body(decl) }}

{% endif %}

{% if decl.deprecated %}

**Deprecated:**

{{ decl.deprecated }}
{% endif %}

{% if decl.parameters and not decl.kind == 'function' %}

## Parameters

{% for param in decl.parameters -%}
*   ​<b>{{ param.name }}</b> ({% if param.traits -%}
        {%- for trait in param.traits -%}
            {%- if trait.path -%}
                [`{{ trait.type }}`]({{ api_path }}{{ trait.path }})
            {%- else -%}
                `{{ trait.type }}`
            {%- endif -%}
            {%- if not loop.last %} & {% endif -%}
        {%- endfor -%}
    {%- else -%}
        {%- if param.path -%}
            [`{{ param.type }}`]({{ api_path }}{{ param.path }})
        {%- else -%}
            `{{ param.type }}`
        {%- endif -%}
    {%- endif %}): {{ param.description }}
{% endfor %}
{% endif %}
{% if decl.fields %}

## Fields

{% for field in decl.fields %}
* ​<b>{{ field.name }}</b> (`{{field.type}}`): {{ field.summary }}
{% if field.description %}
{{field.description | indent(2, True, False)}}
{% endif %}
{% endfor %}
{% endif %}
{% if decl.parentTraits %}
{% if decl.kind == 'struct' or decl.kind == 'trait' %}

## Implemented traits

{% for trait in decl.parentTraits %}
{% if trait.path %}[`{{ trait.name }}`]({{ api_path }}{{ trait.path }}){% else %}`{{ trait.name }}`{% endif %}{{ ", " if not loop.last else "" }}
{% endfor %}

{% endif %}
{% endif %}
{% if decl.aliases %}

## Aliases

{% for alias in decl.aliases | sort(attribute='name') %}

###  `{{ alias.name }}`

<div class='mojo-alias-detail'>
<div class="mojo-alias-sig">

{# don't show value for trait aliases (no value) or if name == value #}
{% if alias.value and alias.name != alias.value %}
`{{ alias.signature }} = {{ alias.value }}`
{% else %}
`{{ alias.signature }}`
{% endif %}

</div>

{% if alias.summary %}
{{ alias.summary }}

{% endif %}
{% if alias.description %}
{{alias.description | indent(2, True, False)}}
{% endif %}
{% if alias.deprecated %}
**Deprecated:** {{ alias.deprecated }}
{% endif %}

{% if alias.parameters %}

#### Parameters

{% for param in alias.parameters -%}
*   ​<b>{{ param.name }}</b> ({% if param.traits -%}
        {%- for trait in param.traits -%}
            {%- if trait.path -%}
                [`{{ trait.type }}`]({{ api_path }}{{ trait.path }})
            {%- else -%}
                `{{ trait.type }}`
            {%- endif -%}
            {%- if not loop.last %} & {% endif -%}
        {%- endfor -%}
    {%- else -%}
        {%- if param.path -%}
            [`{{ param.type }}`]({{ api_path }}{{ param.path }})
        {%- else -%}
            `{{ param.type }}`
        {%- endif -%}
    {%- endif %}): {{ param.description }}
{% endfor %}
{% endif %}
</div>

{% endfor %}
{% endif %}
{% if decl.functions %}

## Methods

{{ loop(decl.functions) }}
{% endif %}
{% endfor %}

</section>
