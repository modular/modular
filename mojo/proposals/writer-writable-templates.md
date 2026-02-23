# `Writable` and `Writer` and template engines

#### Status
- 2026/02/23: proposed

#### Description
There are many ways we can implement each step of this, so I'm purposefully
keeping it vague in the present document to leave the more detailed and
technical discussions for another time. The goal is to agree on the
general direction and greenlit community-led efforts in this area (given
it doesn't fully align with Modular's needs). This is not supposed to be stable
nor be completed before Mojo 1.0

## End-goal

Let's start at the end and what all of this would allow.

The goal is for Mojo's format/template strings to be inspired by Python's
but go much further beyond in supporting real-life use-cases like templating
engines used in the Python ecosystem in an almost native-feeling way. There
are millions of programs out there using them and a lot of unresolved
pain-points.

For some classic examples we would be able to do some things that many Python
users know:
```mojo
print("{:^4}".format("hi") # " hi "
print("{!r}".format("hi") # "'hi'" it's the same as using repr("hi")
```

### A dream come true

Once we somehow get support for the equivalent of Python's `f"{some_var}"`,
we would use this. This is imaginary placeholder syntax.

JSON templating:
```mojo
print("!j".format({"some": True})) # "{"some":true}"
print("!j:p".format({"some": True})) # "{\n\t"some":true\n}" pretty printing
```

HTML templating:
```mojo
# This would be fully type-checked and compiled html
# making errors in the templating string at runtime
# a thing of the past
print(
f[HTMLTemplateEngine]"""
<div class="user-profile">
  <h1>User: {user_name}</h1>
  <ul>
    {% for badge in badges %}
      <li>{badge}</li>
    {% endfor %}
  </ul>
</div>
"""
)
```

SQL templating:
```mojo
struct Table:
  var some_col: String

# This is the dream:
# - fails at compile time
# - gives a "table called: 'Table' has no 'some_col2' column" error
# - gives a "trailing comma after select statement" error
# - gives a "function 'FARM_FINGERPRINTT' not found, maybe you meant
#     'FARM_FINGERPRINT' ?" error
sql_conn.execute(
f[SomeSQLDialectTemplateEngine]"""
SELECT FARM_FINGERPRINTT(some_col2) as hash,
FROM {Table}
"""
)
```

Ideally this would eventually even get IDE support and any developer dealing
with string templates will have a built-in way to check for errors in the
template strings themselves. Imagine even having plugins that fix the
f-string pretty print indentation for you 🤯

We could also have template engines that e.g. remove newline and/or leading
space characters in the format strings themselves, that way we avoid common
ugly patterns of using
```mojo
def some_fn(a: str) -> str:
  return f"""\
something: {a}\
"""
```

Bonus points, if somebody wants to build a Python syntax to Mojo template
engine I think this would have some interesting results.

## How do we get there?

#### Graphemes

What Python can't deal with however is graphemes. This should work in Mojo
```mojo
# 👨‍👩‍👧‍👦 is the union of 4 different unicode codepoints
print("{:^4}".format("👨‍👩‍👧‍👦")) # " 👨‍👩‍👧‍👦 "
```

#### Conversion flags

Python has 3 conversion flags:
- `!s` is the same as calling `str(elem)`
- `!r` is same as calling `repr(elem)`
- `!a` is the same as calling `ascii(elem)`

I think we should support custom conversion flags that signal that a
given custom trait should be used. We can check conformance at compile time
if the format strings are compiled beforehand (`StringLiteral` does this
by default) or raise at runtime if not.

#### Format specification

Python has extensive capabilities like specifying minimum with or alignment
as shown above. We want to go beyond and support custom format specifications.

#### Format Start and end characters

Each templating engine might have their own use-case-specific start and end
characters, which might overlap with the stdlib's default (same as Python's)
`"{"` and `"}"`. We should allow different character sets and potentially
different amount of them e.g. `"{%"` and `"%}"`.

#### Buffer reservation mechanics

The more complex the templating engine the more the final buffer can be
bloated relative to the length of the original string, so estimating the
exact or upper/lower buffer length requirements is crucial. Imagine a for
loop in an html that has to render 1000 elements of 100+ characters, the
performance impact of reallocating would be huge if the format string
doesn't have a way to check how much the final buffer would need to be
reserved beforehand.
