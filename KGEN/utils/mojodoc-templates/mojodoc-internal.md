{%- macro process_decl_body(decl) -%}

{% if decl.signature %}
> `` {{ decl.signature }} ``
{% endif %}

{{ decl.summary }}

{{ decl.description }}

{%- if decl.parameters %}
**Parameters**
{% for param in decl.parameters %}
- ``{{ param.signature }}``: {{ param.description }}
{% endfor %}
{% endif %}

{%- if decl.args -%}
**Args**
{% for arg in decl.args %}
- ``{{ arg.signature }}``: {{ arg.description }}
{% endfor %}
{%- endif -%}

{% if decl.returns %}
**Returns**

{{ decl.returns }}
{% endif %}

{%- if decl.constraints %}
**Constraints**

{{ decl.constraints }}
{%- endif -%}
{%- endmacro -%}

{% for decl in decls recursive %}
{{ "#"*loop.depth }} {{ decl.name }}

{% if decl.overloads %}
{%- for overload in decl.overloads -%}
{{ process_decl_body(overload) }}

{% endfor %}
{% else %}
{{ process_decl_body(decl) }}
{% endif %}

{%- if decl.children -%}
{{ loop(decl.children) }}
{%- endif -%}
{%- endfor -%}
