<!-- markdownlint-disable -->
{% import 'macros.jinja' as macros %}
{# Print YAML front matter #}
{% macro print_front_matter(decl) %}
---
title: {{ decl.name }}
{% if decl.sidebar_label %}sidebar_label: {{ decl.sidebar_label }}
{% endif %}
version: {{ decl.version }}
slug: {{ decl.slug }}
type: function
namespace: {{ decl.namespace }}
lang: mojo
description: {% if decl.overloads[0].summary
  %}"{{ macros.escape_quotes(decl.overloads[0].summary) }}"
  {% else %}"Mojo function `{{ decl.namespace }}.{{ decl.name }}` documentation"
  {% endif %}
---

<section class='mojo-docs'>

{% endmacro -%}
{# Print each declaration #}
{% macro process_decl_body(decl) %}
{% if decl.signature %}
<div class="mojo-function-sig">

`{% if decl.isStatic %}static {% endif %}{{ decl.signature }}`

</div>
{% endif %}

{{ decl.summary }}

{{ decl.description }}

{% if decl.deprecated %}

**Deprecated:**

{{ decl.deprecated }}
{% endif %}

{% if decl.constraints %}

**Constraints:**

{{ decl.constraints }}
{% endif %}
{% if decl.parameters %}

**Parameters:**

{% for param in decl.parameters -%}
*   ​<b>{{ param.name }}</b> (`{{ param.type }}`): {{ param.description }}
{% endfor %}
{% endif %}
{% if decl.args %}

**Args:**

{% for arg in decl.args -%}
*   ​<b>{{ arg.name }}</b> (`{{ arg.type }}`): {{ arg.description }}
{% endfor %}
{% endif %}
{% if decl.returnsDoc %}

**Returns:**

{{ decl.returnsDoc }}
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
{% elif (decl.kind == "module_link") or (decl.kind == "package_link") %}
{{ "#"*loop.depth }} [`{{ decl.name }}`]({{ decl.link }})
{% elif (decl.kind != "alias") and (decl.kind != "field") %}
{{ "#"*loop.depth }} `{{ decl.name }}`
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
{% endfor %}

</section>
