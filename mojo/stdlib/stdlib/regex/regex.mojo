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
"""Regular expression pattern matching implementation.

This module provides a simple regex engine that supports basic pattern matching
operations like search, match, and replace. The implementation is intentionally
kept minimal while providing a foundation for more advanced features in the future.

The core components are:
- Match: A struct representing a successful regex match.
- Pattern: A struct representing a compiled regex pattern.
- Functions for searching, matching, splitting, and replacing.

Current limitations:
- Limited support for special regex syntax (mostly literal matching).
- No support for complex patterns with groups, quantifiers, or character classes.
- Limited optimization for large strings or complex patterns.

Future extensions might include support for:
- Capturing groups with ().
- Quantifiers like *, +, ?, and {n,m}.
- Character classes with [], ^, and $.
- Lookahead and lookbehind assertions.
"""

from collections import Optional


@value
struct Match:
    """Represents the result of a successful regex match operation.

    This struct holds information about a match found by a regex pattern,
    including its position in the string, the original string, and any
    captured groups (though group capture is limited in the current implementation).

    The Match struct provides methods to access the matched substring and
    any captured groups. The main match is considered group 0, while
    captured parenthesized groups (when implemented) will be indexed from 1.

    Attributes:
        start: The index where the match starts in the original string.
        end: The index where the match ends in the original string.
        string: The original string that was searched.
        groups: List of matched groups in the pattern. In the current
               implementation, this list is usually empty since capturing
               groups are not yet fully supported.
    """

    var start: Int
    """The index where the match starts in the original string."""
    var end: Int
    """The index where the match ends in the original string."""
    var string: String
    """The original string that was searched."""
    var groups: List[String]
    """List of matched groups in the pattern."""

    fn __init__(
        out self,
        start: Int,
        end: Int,
        string: String,
        groups: List[String] = List[String](),
    ):
        """Initialize a Match object.

        Args:
            start: The index where the match starts.
            end: The index where the match ends.
            string: The string that was searched.
            groups: List of matched capture groups.
        """
        self.start = start
        self.end = end
        self.string = string
        self.groups = groups

    fn group(self, index: Int = 0) -> String:
        """Returns the substring matched by the regular expression.

        This method retrieves the substring that was matched by the regex pattern
        or a specific capturing group within the pattern. Group 0 (the default)
        refers to the entire match, while groups 1 and higher refer to the
        corresponding capturing groups in the pattern.

        Note that in the current implementation, capturing groups are not fully
        supported, so index values other than 0 will typically return empty
        strings unless explicitly populated.

        Args:
            index: Index of the group to return (0 is the entire match).
                  Indices > 0 refer to capturing groups in the pattern.

        Returns:
            The matched substring for the requested group. If the group
            doesn't exist, returns an empty string.
        """
        if index == 0:
            return self.string[self.start : self.end]
        elif index > 0 and index <= len(self.groups):
            return self.groups[index - 1]
        else:
            return String("")


@value
struct Pattern:
    """Compiled regular expression pattern.

    This struct represents a compiled regular expression pattern that can be used
    for matching strings. A pattern is created by calling the compile() function
    with a regex pattern string.

    The Pattern struct provides methods for searching and matching against strings:
    - search(): Find the pattern anywhere in a string.
    - _match(): Match the pattern only at the beginning of a string.

    The Pattern also handles case sensitivity settings and basic processing
    of special regex characters, though the current implementation has
    limited support for standard regex syntax.

    In future versions, this struct will be enhanced to support more
    complex regex features like capturing groups, anchors, and quantifiers.
    """

    var pattern: String
    """The regex pattern string."""
    var is_case_sensitive: Bool
    """Whether the pattern is case sensitive."""

    fn __init__(out self, pattern: String, case_sensitive: Bool = True):
        """Initialize a regex pattern.

        Args:
            pattern: The regex pattern string.
            case_sensitive: Whether the pattern is case sensitive.
        """
        self.pattern = pattern
        self.is_case_sensitive = case_sensitive

    fn search(self, string: String, start_pos: Int = 0) -> Optional[Match]:
        """Search for the pattern anywhere in the string.

        This method searches for the pattern in the given string, starting
        from the specified position. It returns a Match object if the pattern
        is found, or None if no match is found.

        The search is performed by looking for an exact substring match after
        processing any special characters in the pattern. This is a simplified
        approach compared to full regex engines.

        If case_sensitive is False, both the pattern and the search string
        are converted to lowercase before matching.

        Args:
            string: The string to search in.
            start_pos: Position to start the search from. Defaults to 0,
                      which means to start from the beginning of the string.

        Returns:
            A Match object if the pattern is found, None otherwise.
        """
        # Currently implements exact substring match as a simplification.
        # A full regex engine would use a state machine or similar approach.
        var search_str = string[start_pos:]
        var pattern_str = self.pattern

        # Handle special characters by escaping them for now.
        # This converts regex metacharacters to their literal representations.
        pattern_str = self._process_pattern(pattern_str)

        # Simple case insensitive handling.
        # If case_sensitive is False, convert both strings to lowercase for comparison.
        if not self.is_case_sensitive:
            search_str = search_str.lower()
            pattern_str = pattern_str.lower()

        # Use the built-in string find method to locate the pattern in the search string.
        var idx = search_str.find(pattern_str)
        if idx >= 0:
            # Calculate the actual start position in the original string.
            var match_start = start_pos + idx
            # Calculate the end position by adding the pattern length.
            var match_end = match_start + len(pattern_str)
            # Return a Match object with the found position information.
            return Match(match_start, match_end, string)
        # No match was found, return None.
        return None

    fn _match(self, string: String) -> Optional[Match]:
        """Match the pattern at the beginning of the string.

        This method checks if the pattern matches at the very beginning of
        the given string. Unlike search(), which looks for the pattern anywhere
        in the string, this method only succeeds if the pattern is found at
        the start of the string.

        The match is performed by checking if the string starts with the
        pattern after processing any special characters. This is a simplified
        approach compared to full regex engines.

        If case_sensitive is False, both the pattern and the string
        are converted to lowercase before matching.

        Args:
            string: The string to match against.

        Returns:
            A Match object if the pattern matches at the beginning of
            the string, None otherwise.
        """
        # Process the pattern to handle any special characters.
        var pattern_str = self._process_pattern(self.pattern)

        # Simple case insensitive handling.
        # If case_sensitive is False, convert both strings to lowercase for comparison.
        var test_string = string
        if not self.is_case_sensitive:
            test_string = string.lower()
            pattern_str = pattern_str.lower()

        # Check if the string starts with the pattern.
        # This is more efficient than using search() as we only need to check the beginning.
        if test_string.startswith(pattern_str):
            # Create a Match object with start=0 and end=length of the pattern.
            return Match(0, len(pattern_str), string)
        # No match at the beginning, return None.
        return None

    fn _process_pattern(self, pattern: String) -> String:
        """Process special characters in the pattern.

        This method prepares a raw pattern string for matching by handling
        special regex characters. In the current implementation, it only
        processes escape sequences (\\) to treat the escaped character as a
        literal rather than a special regex character.

        For example, in the pattern "\\d+", the '\\d' would be treated as a
        literal 'd' rather than the regex metacharacter for digits.

        Future implementations will enhance this method to fully support
        standard regex syntax including:
        - Character classes: \\d, \\w, \\s, etc.
        - Quantifiers: *, +, ?, {n,m}.
        - Anchors: ^, $.
        - Groups: (...).

        Args:
            pattern: The raw pattern string to process.

        Returns:
            The processed pattern string ready for matching.
        """
        # This is a very simplified implementation of regex pattern processing.
        # In a real regex engine, this would parse the pattern into tokens for a state machine.
        var result = String("")
        var i = 0

        # Process the pattern character by character.
        while i < len(pattern):
            # Handle escape sequences by treating the escaped character as a literal.
            # For example, \\d would be treated as 'd' rather than the digit metacharacter.
            if pattern[i] == "\\" and i + 1 < len(pattern):
                # Skip the backslash and include the next character literally.
                i += 1
                result += pattern[i]
            # In a real implementation, we'd handle more special characters and regex syntax.
            # including character classes (\\d, \\w, \\s), anchors (^, $), and quantifiers (*, +, ?).
            else:
                # Include regular characters as they are.
                result += pattern[i]
            i += 1

        # Return the processed pattern ready for literal matching.
        return result


fn compile(pattern: String, case_sensitive: Bool = True) -> Pattern:
    """Compile a regular expression pattern.

    This function creates a Pattern object from a regex pattern string.
    The Pattern object can then be used for searching or matching against
    strings using its methods.

    Compiling a pattern is more efficient when you need to use the same
    pattern multiple times, as it avoids reprocessing the pattern for
    each operation.

    The case_sensitive parameter controls whether matching should be
    case-sensitive. If False, case will be ignored when matching.

    Args:
        pattern: The regex pattern string to compile.
        case_sensitive: Whether the pattern matching should be case-sensitive.
                       Defaults to True.

    Returns:
        A compiled Pattern object ready for matching operations.
    """
    return Pattern(pattern, case_sensitive)


fn search(
    pattern: String, string: String, case_sensitive: Bool = True
) -> Optional[Match]:
    """Search for a pattern in a string.

    This function searches for the first occurrence of the pattern in the
    given string. It returns a Match object with information about the
    match if found, or None if no match is found.

    The search is performed by looking for the pattern anywhere in the string.
    This is equivalent to using the search() method on a compiled pattern.

    This function internally compiles the pattern and then delegates to the
    Pattern.search() method. If you need to use the same pattern multiple
    times, it's more efficient to compile it once with compile() and then
    use the resulting Pattern object's search() method.

    Args:
        pattern: The regex pattern to search for.
        string: The string to search in.
        case_sensitive: Whether the search should be case-sensitive.
                       Defaults to True.

    Returns:
        A Match object containing match information if found, None otherwise.
    """
    var p = compile(pattern, case_sensitive)
    return p.search(string)


fn _match(
    pattern: String, string: String, case_sensitive: Bool = True
) -> Optional[Match]:
    """Match a pattern at the beginning of a string.

    This function checks if the pattern matches at the beginning of the
    given string. It returns a Match object with information about the
    match if found, or None if no match is found.

    Unlike search(), which looks for the pattern anywhere in the string,
    this function only succeeds if the pattern is found at the start of
    the string.

    This function internally compiles the pattern and then delegates to the
    Pattern._match() method. If you need to use the same pattern multiple
    times, it's more efficient to compile it once with compile() and then
    use the resulting Pattern object's _match() method.

    Note: This is a private implementation method. Users should use the
    publicly exported 'match' function from the regex module.

    Args:
        pattern: The regex pattern to match.
        string: The string to match against.
        case_sensitive: Whether the match should be case-sensitive.
                       Defaults to True.

    Returns:
        A Match object containing match information if matched at the
        beginning of the string, None otherwise.
    """
    var p = compile(pattern, case_sensitive)
    return p._match(string)


fn fullmatch(
    pattern: String, string: String, case_sensitive: Bool = True
) -> Optional[Match]:
    """Match the entire string to the pattern.

    This function checks if the pattern matches the entire string exactly.
    It returns a Match object with information about the match if the
    entire string matches the pattern, or None otherwise.

    Unlike search(), which looks for the pattern anywhere in the string,
    or match(), which only matches at the beginning, fullmatch() requires
    that the pattern matches the complete string from start to end.

    This is implemented by first checking if the pattern matches at the beginning
    of the string, and then verifying that the match covers the entire string.

    Example:

    ```mojo
    # Returns a Match object.
    fullmatch("hello", "hello")

    # Returns None (since there's more text after the match).
    fullmatch("hello", "hello world")
    ```

    Args:
        pattern: The regex pattern to match.
        string: The string to match against.
        case_sensitive: Whether the match should be case-sensitive.
                       Defaults to True.

    Returns:
        A Match object if the pattern matches the entire string,
        None otherwise.
    """
    var p = compile(pattern, case_sensitive)
    var m = p._match(string)

    # Check if the match covers the entire string.
    if m and m.value().end == len(string):
        return m
    return None


fn split(
    pattern: String,
    string: String,
    maxsplit: Int = 0,
    case_sensitive: Bool = True,
) -> List[String]:
    """Split a string by occurrences of the pattern.

    This function divides a string into a list of substrings based on where
    the pattern matches. The pattern itself is not included in any of the
    resulting substrings.

    By default, this function will split the string on all occurrences of the
    pattern. You can limit the number of splits by setting the maxsplit parameter
    to a positive integer.

    The splitting process works by:
    1. Finding each match of the pattern in the string
    2. Adding the text before each match to the result list
    3. After processing all matches (or reaching maxsplit), adding any
       remaining text to the result

    If the pattern doesn't match anywhere in the string, the result will
    be a list containing just the original string.

    Example:

    ```mojo
    # Returns ["apple", "banana", "orange"].
    split(",", "apple,banana,orange")

    # Returns ["apple", "banana,orange"] (with maxsplit=1).
    split(",", "apple,banana,orange", maxsplit=1)
    ```

    Args:
        pattern: The regex pattern to use as the delimiter for splitting.
        string: The string to split.
        maxsplit: Maximum number of splits to perform. 0 (default) means
                 no limit on the number of splits.
        case_sensitive: Whether the pattern matching should be case-sensitive.
                       Defaults to True.

    Returns:
        A list of substrings resulting from splitting the original string
        on each occurrence of the pattern.
    """
    var p = compile(pattern, case_sensitive)
    var result = List[String]()
    var start = 0
    var splits = 0

    # Continue searching for matches and splitting until no more are found.
    # or we've reached the maximum number of splits.
    while True:
        # Search for the next occurrence of the pattern.
        var m = p.search(string, start)

        # Exit the loop if no more matches are found or we've reached maxsplit.
        if not m or (maxsplit > 0 and splits >= maxsplit):
            break

        # Add the text before the match to the result list.
        result.append(string[start : m.value().start])

        # Update the starting position for the next search to after this match.
        start = m.value().end

        # Increment the number of splits performed.
        splits += 1

    # Add the remaining part of the string (after the last match) to the result.
    # If no matches were found, this will be the entire string.
    result.append(string[start:])

    return result


fn sub(
    pattern: String,
    replacement: String,
    string: String,
    count: Int = 0,
    case_sensitive: Bool = True,
) -> String:
    """Replace occurrences of the pattern in the string.

    This function replaces occurrences of the pattern in the given string with
    the specified replacement string. By default, it replaces all occurrences,
    but you can limit the number of replacements by setting the count parameter.

    The substitution process works by:
    1. Finding each match of the pattern in the string
    2. Building a new string with the text before the match, the replacement
       string, and then continuing with the remaining text
    3. Repeating until no more matches are found or the count limit is reached

    In the current implementation, the replacement string is used as-is without
    any special handling of capture group references (like $1, $2, etc. in some
    regex engines). Future versions might support this feature.

    Example:

    ```mojo
    # Returns "hi world, hi universe".
    sub("hello", "hi", "hello world, hello universe")

    # Returns "hi world, hello universe" (with count=1).
    sub("hello", "hi", "hello world, hello universe", count=1)
    ```

    Args:
        pattern: The regex pattern to find in the string.
        replacement: The string to use as a replacement.
        string: The original string to modify.
        count: Maximum number of replacements to perform. 0 (default) means
              replace all occurrences.
        case_sensitive: Whether the pattern matching should be case-sensitive.
                       Defaults to True.

    Returns:
        A new string with the replacements made.
    """
    var p = compile(pattern, case_sensitive)
    var result = String("")
    var start = 0
    var replacements = 0

    # Continue replacing matches until no more are found or we've reached
    # the maximum number of replacements.
    while True:
        # Search for the next occurrence of the pattern.
        var m = p.search(string, start)

        # Exit the loop if no more matches are found or we've reached count.
        if not m or (count > 0 and replacements >= count):
            break

        # Add the text before the match plus the replacement string.
        result += string[start : m.unsafe_value().start]
        result += replacement

        # Update the starting position for the next search to after this match.
        start = m.unsafe_value().end

        # Increment the number of replacements performed.
        replacements += 1

    # Add the remaining part of the string (after the last replacement) to the result.
    # If no replacements were made, this will be the entire string.
    result += string[start:]

    return result
