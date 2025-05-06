# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Utility functions for regex pattern handling.

This module provides helper functions for processing regex patterns,
handling character classes, and special symbols. These utilities are used
internally by the regex module to implement pattern matching functionality.

The module includes:
- Character class definitions for common regex patterns like \\d, \\w, \\s.
- Functions for expanding character class patterns like [a-z0-9].
- Utilities for handling special regex syntax.

Future enhancements will include more sophisticated handling of regex
constructs such as lookaheads, lookbehinds, and backreferences.
"""

from collections import Dict


@value
struct CharacterClass:
    """Represents a character class in a regular expression.

    Character classes represent sets of characters that can match at a position,
    such as digits, letters, or custom sets defined with brackets.

    In regex syntax, character classes can be predefined (like \\d for digits)
    or custom-defined using brackets (like [a-z] for lowercase letters).
    This struct provides a way to represent and work with these classes.

    The struct stores both a name for the class (e.g., "digit" for \\d) and
    the actual set of characters that belong to the class. It also provides
    methods to check if a given character is a member of the class.

    Example usage:

    ```mojo
    var digit_class = CharacterClass("digit", "0123456789")
    if digit_class.matches("5"):
        print("5 is a digit")
    ```
    """

    var name: String
    """The name of the character class."""
    var chars: String
    """The characters that are part of this class."""

    fn __init__(out self, name: String, chars: String):
        """Initialize a character class.

        Args:
            name: The name of the character class (e.g., "digit", "word").
            chars: The characters that are part of this class.
        """
        self.name = name
        self.chars = chars

    fn matches(self, c: String) raises -> Bool:
        """Check if a character is in this character class.

        This method determines whether a given character is a member of
        this character class. It does this by checking if the character
        is present in the chars string for this class.

        Note that this method expects a single character as input. If a
        string with multiple characters is provided, it will return False
        since character classes match exactly one character at a time.

        The method uses the 'in' operator to check for membership, which
        for strings checks if one string is a substring of another. Since
        we're dealing with single characters, this effectively checks if
        the character is one of the allowed characters in the class.

        Args:
            c: The character to check for membership in this class.
               Should be a single character.

        Returns:
            True if the character is in this character class, False otherwise.

        Raises:
            If the string operation fails (unlikely with normal usage).
        """
        if len(c) != 1:
            return False
        return c in self.chars


fn get_common_character_classes() -> Dict[String, CharacterClass]:
    """Get a dictionary of common character classes used in regex.

    This function creates and returns a dictionary containing the standard
    character classes used in regular expressions. Each character class is
    mapped from its regex syntax representation (like "\\d" for digits) to a
    CharacterClass object containing all the characters in that class.

    The following character classes are included:
    - \\d: Digits (0-9).
    - \\w: Word characters (letters, digits, underscore).
    - \\s: Whitespace characters (space, tab, newline, etc.).
    - \\D: Non-digits (complement of \\d).
    - \\W: Non-word characters (complement of \\w).
    - \\S: Non-whitespace (complement of \\s).

    These character classes correspond to the standard character classes
    used in most regex engines like PCRE, Python's re module, and JavaScript.

    The returned dictionary can be used to quickly look up a character class
    by its syntax representation and get the corresponding set of characters.

    Returns:
        A dictionary mapping regex character class syntax (like "\\d") to
        CharacterClass objects containing the appropriate characters.
    """
    var classes = Dict[String, CharacterClass]()

    # Digits.
    classes["\\d"] = CharacterClass("digit", "0123456789")

    # Word characters (letters, digits, underscore).
    classes["\\w"] = CharacterClass(
        "word",
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
    )

    # Whitespace.
    classes["\\s"] = CharacterClass("whitespace", " \t\n\r\f\v")

    # Not digits (complement of \\d).
    classes["\\D"] = CharacterClass(
        "non-digit",
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_!@#$%^&*()[]{}|;:,.<>/?'\\"
        + " \t\n\r\f\v",
    )

    # Not word characters (complement of \\w).
    classes["\\W"] = CharacterClass(
        "non-word", "!@#$%^&*()[]{}|;:,.<>/?'\\" + " \t\n\r\f\v"
    )

    # Not whitespace (complement of \\s).
    classes["\\S"] = CharacterClass(
        "non-whitespace",
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_!@#$%^&*()[]{}|;:,.<>/?'\\",
    )

    return classes


fn expand_character_class(class_str: String) raises -> String:
    """Expand a character class pattern to its member characters.

    This function takes a regex character class specification and expands it
    into a string containing all the individual characters that are members
    of that class. This is useful for pattern matching operations that need
    to know exactly which characters are included in a class.

    The function handles:
    1. Predefined character classes like \\d, \\w, \\s and their negations.
    2. Custom character classes defined with brackets like [a-z0-9].
    3. Character ranges within custom classes like a-z (expands to all letters).
    4. Negated character classes with ^ like [^0-9] (all non-digits).

    For negated character classes, the function returns all ASCII printable
    characters (codes 32-126) that are not in the specified class.

    Example:

    ```mojo
    var digits = expand_character_class("\\d")  # "0123456789".
    var lowercase = expand_character_class("[a-z]")  # "abcdefghijklmnopqrstuvwxyz".
    var non_digits = expand_character_class("[^0-9]")  # All printable non-digit chars.
    ```

    Args:
        class_str: A character class pattern to expand. This can be either a
                  predefined class like "\\d" or a custom class like "[a-z0-9]".

    Returns:
        A string containing all the individual characters that are members
        of the specified character class.

    Raises:
        If string operations fail during processing (unlikely with normal usage).
    """
    # This is a simplified implementation that handles basic character classes.
    # A more comprehensive implementation would handle more complex regex syntax.
    var result = String("")

    # Get the dictionary of predefined character classes like \\d, \\w, \\s.
    alias classes = get_common_character_classes()

    # First, check if this is one of the predefined character classes.
    # If so, just return the pre-computed set of characters for that class.
    if class_str in classes:
        return classes[class_str].chars

    # Next, handle custom character classes defined with brackets [...].
    # These allow users to define their own sets of characters to match.
    if class_str.startswith("[") and class_str.endswith("]"):
        # Extract the content between the brackets.
        var content = class_str[1 : len(class_str) - 1]
        var i = 0
        var is_negated = False

        # Check if this is a negated character class [^...].
        # Negated classes match any character NOT in the specified set.
        if len(content) > 0 and content[0] == "^":
            is_negated = True
            # Remove the negation symbol from the content.
            content = content[1:]

        # Process the content of the character class character by character.
        while i < len(content):
            # Check if we have a character range like 'a-z' which represents
            # all characters from a to z inclusive.
            if i + 2 < len(content) and content[i + 1] == "-":
                # Convert the start and end characters to their ASCII codes.
                var start_char = ord(content[i])
                var end_char = ord(content[i + 2])

                # Add each character in the range to the result.
                for c in range(start_char, end_char + 1):
                    result += chr(c)

                # Skip ahead past the range.
                i += 3
            else:
                # For individual characters, just add them to the result.
                result += content[i]
                i += 1

        # If this is a negated class, we need to return all printable ASCII
        # characters EXCEPT those in the specified class.
        if is_negated:
            var all_chars = String("")

            # Loop through all printable ASCII characters (codes 32-126).
            for c in range(32, 127):  # ASCII printable range.
                var char = chr(c)

                # Add the character to the result only if it's not in the negated set.
                if char not in result:
                    all_chars += char

            return all_chars

        # For non-negated classes, return the accumulated characters.
        return result

    # If the class_str doesn't match any known pattern, just return it as-is.
    # This is a fallback for cases we don't handle yet.
    return class_str
