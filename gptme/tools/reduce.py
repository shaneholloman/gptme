import logging
from typing import Generator

from ..message import Message
from ..util import len_tokens

logger = logging.getLogger(__name__)


# GPT-4 has a 8k token limit
TOKEN_LIMIT_SOFT = 6000
TOKEN_LIMIT_HARD = 7000


def reduce_log(
    log: list[Message],
    limit=TOKEN_LIMIT_SOFT,
    prev_len=None,
) -> Generator[Message, None, None]:
    """Reduces log until it is below `limit` tokens by continually summarizing the longest messages until below the limit."""
    # if we are below the limit, return the log as-is
    tokens = len_tokens(log)
    if tokens <= limit:
        yield from log
        return

    logger.info(f"Log exceeded limit of {limit}, was {tokens}, reducing")
    # filter out pinned messages
    i, longest_msg = max(
        [(i, m) for i, m in enumerate(log) if not m.pinned],
        key=lambda t: len_tokens(t[1].content),
    )

    # attempt to truncate the longest message
    truncated = truncate_msg(longest_msg)

    # if unchanged after truncate, attempt summarize
    if truncated:
        summary_msg = truncated
    else:
        # NOTE: disabled because buggy
        # from . import summarize
        # summary_msg = summarize(longest_msg, preamble=False)
        # logger.info("Summary: %s", summary_msg.content)
        # summary_msg.content = f"This {summary_msg.role} message was summarized due to length:\n{summary_msg.content}"
        summary_msg = longest_msg

    log = log[:i] + [summary_msg] + log[i + 1 :]

    tokens = len_tokens(log)
    if tokens <= limit:
        yield from log
    else:
        # recurse until we are below the limit
        # but if prev_len == tokens, we are not making progress, so just return the log as-is
        if prev_len == tokens:
            logger.warning("Not making progress, returning log as-is")
            yield from log
        else:
            yield from reduce_log(log, limit, prev_len=tokens)


def truncate_msg(msg: Message, lines_pre=10, lines_post=10) -> Message | None:
    """Truncates message codeblocks to the first and last `lines_pre` and `lines_post` lines, keeping the rest as `[...]`."""
    # TODO: also truncate long <details> (as can be found in GitHub issue comments)
    content_staged = msg.content

    # Truncate long codeblocks
    for codeblock in msg.get_codeblocks():
        assert codeblock in content_staged
        lang_or_fn, content = codeblock.split("```", 1)[1].split("\n", 1)
        # truncate the middle part of the codeblock, keeping the first and last n lines
        lines = content.split("\n")
        if len(lines) > lines_pre + lines_post + 1:
            content = "\n".join([*lines[:lines_pre], "[...]", *lines[-lines_post:]])
        else:
            logger.warning("Not enough lines in codeblock to truncate")
            continue

        # replace the codeblock with the truncated version
        assert codeblock in content_staged
        content_staged_prev = content_staged
        content_staged = content_staged.replace(
            codeblock, f"```{lang_or_fn}\n{content}\n```"
        )
        assert content_staged != content_staged_prev
        assert codeblock not in content_staged

    if content_staged != msg.content:
        return Message(
            role=msg.role,
            content=content_staged,
        )
    else:
        return None


def limit_log(log: list[Message]) -> list[Message]:
    """
    Picks messages until the total number of tokens exceeds TOKEN_LIMIT_SOFT,
    then removes the last message to get below the limit.
    Will always pick the first few system messages.
    """
    # Always pick the first system messages
    initial_system_msgs = []
    for msg in log:
        if msg.role != "system":
            break
        initial_system_msgs.append(msg)

    # Pick the messages in latest-first order
    msgs = []
    for msg in reversed(log[len(initial_system_msgs) :]):
        msgs.append(msg)
        if len_tokens(msgs) > TOKEN_LIMIT_SOFT:
            break

    # Remove the message that put us over the limit
    if len_tokens(msgs) > TOKEN_LIMIT_SOFT:
        # skip the last message
        msgs.pop()

    return initial_system_msgs + list(reversed(msgs))
