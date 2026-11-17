# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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
# Compile-time enum values used at run time:
# 1. Materialize byte, heap, interior-pointer and function-symbol constants.
# 2. Clear pointer and symbol marks when a compile-time value changes cases.
# 3. Preserve function marks across separate evaluations and list moves.
# 4. Compare runtime values and parameter identity after overwriting padded payloads.
# RUN: %mojo %s | FileCheck %s


__enum Text(Equatable, Hashable, ImplicitlyCopyable, Writable):
    case end
    case text(String)
    case word(Int)


__enum Outer(ImplicitlyCopyable, Writable):
    case inner(Text)


struct Node(ImplicitlyCopyable, Movable):
    var value: Int
    var pointer: Pointer[Int, MutUntrackedOrigin]

    def __init__(out self, value: Int):
        self.value = value
        self.pointer = Pointer[Int, MutUntrackedOrigin](
            unsafe_from_address=Int(Pointer(to=self.value))
        )

    def __init__(out self, *, copy: Self):
        self.value = copy.value
        self.pointer = Pointer[Int, MutUntrackedOrigin](
            unsafe_from_address=Int(Pointer(to=self.value))
        )

    def __init__(out self, *, deinit move: Self):
        self.value = move.value
        self.pointer = Pointer[Int, MutUntrackedOrigin](
            unsafe_from_address=Int(Pointer(to=self.value))
        )


__enum Box(ImplicitlyCopyable):
    case node(Node)


def view(label: String, value: Box):
    match value:
        case .node(ref node):
            # CHECK-DAG: node 17 17 True
            # CHECK-DAG: copied-node 17 17 True
            print(
                label,
                node.value,
                node.pointer[],
                Int(node.pointer) == Int(Pointer(to=node.value)),
            )


def empty_after_text() -> Text:
    var value = Text.text(
        "a long compile-time string that requires heap storage"
    )
    _ = value
    value = Text.end
    return value


def word_after_text() -> Text:
    var value = Text.text(
        "a long compile-time string that requires heap storage"
    )
    _ = value
    value = Text.word(17)
    return value


def text_after_word() -> Text:
    var value = Text.word(23)
    _ = value
    value = Text.text(
        "another long compile-time string that requires heap storage"
    )
    return value


@fieldwise_init
struct Callback(ImplicitlyCopyable):
    var function: def(Int) thin -> Int


def increment(value: Int) -> Int:
    return value + 1


__enum Event(ImplicitlyCopyable):
    case callback(Callback)
    case number(Int)


def invoke(label: String, value: Event):
    match value:
        case .callback(ref callback):
            # CHECK-DAG: callback 18
            # CHECK-DAG: int-to-symbol 16
            print(label, callback.function(17))
        case .number(ref number):
            # CHECK-DAG: symbol-to-int 23
            print(label, number)


def decrement(value: Int) -> Int:
    return value - 1


def number_after_callback() -> Event:
    var value = Event.callback(Callback(increment))
    _ = value
    value = Event.number(23)
    return value


def callback_after_number() -> Event:
    var value = Event.number(23)
    _ = value
    value = Event.callback(Callback(decrement))
    return value


def sub(value: Int) -> Int:
    return value - 7


def add(value: Int) -> Int:
    return value + 7


def make() -> List[Event]:
    return [Event.callback(Callback(sub)), Event.callback(Callback(add))]


def keep(var events: List[Event]) -> List[Event]:
    return events^


def second(events: List[Event]) -> Event:
    return events[1]


def invoke_reloaded(event: Event) -> Int:
    match event:
        case .callback(callback):
            return callback.function(35)
        case .number(n):
            return n


__enum Choice(Equatable, ImplicitlyCopyable):
    case big(Int, Int)
    case small(Int)
    case end


struct Holder[value: Choice]:
    pass


def changed() -> Choice:
    var value = Choice.big(1, 2)
    _ = value
    value = Choice.small(17)
    return value


def emptied() -> Choice:
    var value = Choice.big(3, 4)
    _ = value
    value = Choice.end
    return value


@fieldwise_init
struct Padded(Equatable, ImplicitlyCopyable):
    var byte: UInt8
    var word: Int


__enum Record(Equatable, ImplicitlyCopyable):
    case big(Int, Int, Int)
    case padded(Padded)


struct RecordHolder[value: Record]:
    pass


def padded() -> Record:
    var value = Record.big(-1, -1, -1)
    _ = value
    value = Record.padded(Padded(7, 19))
    return value


__enum OuterChoice(Equatable, ImplicitlyCopyable):
    case big(Int, Int, Int, Int)
    case inner(Choice)


struct OuterHolder[value: OuterChoice]:
    pass


def nested_choice() -> OuterChoice:
    var value = OuterChoice.big(-1, -1, -1, -1)
    _ = value
    value = OuterChoice.inner(changed())
    return value


def main():
    comptime short = Text.text("short")
    comptime long = Text.text(
        "a long compile-time string that requires heap storage"
    )
    comptime nested_text = Outer.inner(long)
    # CHECK-DAG: short Text.text(short)
    print("short", short)
    # CHECK-DAG: long Text.text(a long compile-time string that requires heap storage)
    print("long", long)
    # CHECK-DAG: nested Outer.inner(Text.text(a long compile-time string that requires heap storage))
    print("nested", nested_text)
    # CHECK-DAG: equal True
    print(
        "equal",
        long
        == Text.text("a long compile-time string that requires heap storage"),
    )
    comptime boxed = Box.node(Node(17))
    var value = boxed
    view("node", value)
    var copied = value
    view("copied-node", copied)
    comptime empty = empty_after_text()
    comptime word = word_after_text()
    comptime text = text_after_word()
    # CHECK-DAG: overwritten Text.end Text.word(17) Text.text(another long compile-time string that requires heap storage)
    print("overwritten", empty, word, text)
    comptime event = Event.callback(Callback(increment))
    invoke("callback", event)
    comptime number = number_after_callback()
    invoke("symbol-to-int", number)
    comptime callback = callback_after_number()
    invoke("int-to-symbol", callback)

    # Separate evaluations require function marks to survive reloading the list.
    comptime events = make()
    comptime reloaded_copy = second(events)
    comptime called = invoke_reloaded(reloaded_copy)
    comptime kept = keep(events)
    comptime copied_after_keep = second(kept)
    comptime called_after_keep = invoke_reloaded(copied_after_keep)
    # CHECK-DAG: reloaded 42
    print("reloaded", called)
    # CHECK-DAG: reloaded-after-move 42
    print("reloaded-after-move", called_after_keep)

    comptime a = changed()
    comptime b = Choice.small(17)
    comptime c = emptied()
    comptime d = Choice.end
    comptime e = padded()
    comptime f = Record.padded(Padded(7, 19))
    comptime g = nested_choice()
    comptime h = OuterChoice.inner(Choice.small(17))
    # CHECK-DAG: identity-small True True
    print("identity-small", a == b, Holder[a] == Holder[b])
    # CHECK-DAG: identity-empty True True
    print("identity-empty", c == d, Holder[c] == Holder[d])
    # CHECK-DAG: identity-padded True True
    print("identity-padded", e == f, RecordHolder[e] == RecordHolder[f])
    # CHECK-DAG: identity-nested True True
    print("identity-nested", g == h, OuterHolder[g] == OuterHolder[h])
