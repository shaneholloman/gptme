"""
This module contains the functions to generate the initial system prompt.
It is used to instruct the LLM about its role, how to use tools, and provide context for the conversation.

When prompting, it is important to provide clear instructions and avoid any ambiguity.
"""

import logging
import platform
import shutil
import subprocess
from collections.abc import Generator, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from .__version__ import __version__
from .config import get_config, get_project_config
from .dirs import get_project_git_dir
from .llm.models import get_model, get_recommended_model
from .message import Message
from .tools import ToolFormat
from .util import document_prompt_function

PromptType = Literal["full", "short"]

logger = logging.getLogger(__name__)


def get_prompt(
    prompt: PromptType | str = "full",
    interactive: bool = True,
    tool_format: ToolFormat = "markdown",
    model: str | None = None,
) -> Message:
    """
    Get the initial system prompt.
    """
    msgs: Iterable
    if prompt == "full":
        msgs = prompt_full(interactive, tool_format, model)
    elif prompt == "short":
        msgs = prompt_short(interactive, tool_format)
    else:
        msgs = [Message("system", prompt)]

    # combine all the system prompt messages into one,
    # also hide them and pin them to the top
    return _join_messages(list(msgs)).replace(hide=True, pinned=True)


def _join_messages(msgs: list[Message]) -> Message:
    """Combine several system prompt messages into one."""
    return Message(
        "system",
        "\n\n".join(m.content for m in msgs),
        hide=any(m.hide for m in msgs),
        pinned=any(m.pinned for m in msgs),
    )


def prompt_full(
    interactive: bool, tool_format: ToolFormat, model: str | None
) -> Generator[Message, None, None]:
    """Full prompt to start the conversation."""
    yield from prompt_gptme(interactive, model)
    yield from prompt_tools(tool_format=tool_format)
    if interactive:
        yield from prompt_user()
    yield from prompt_project()
    yield from prompt_systeminfo()
    yield from prompt_timeinfo()


def prompt_short(
    interactive: bool, tool_format: ToolFormat
) -> Generator[Message, None, None]:
    """Short prompt to start the conversation."""
    yield from prompt_gptme(interactive)
    yield from prompt_tools(examples=False, tool_format=tool_format)
    if interactive:
        yield from prompt_user()
    yield from prompt_project()


def prompt_gptme(
    interactive: bool, model: str | None = None
) -> Generator[Message, None, None]:
    """
    Base system prompt for gptme.

    It should:
     - Introduce gptme and its general capabilities and purpose
     - Ensure that it lets the user mostly ask and confirm actions (apply patches, run commands)
     - Provide a brief overview of the capabilities and tools available
     - Not mention tools which may not be loaded (browser, vision)
     - Mention the ability to self-correct and ask clarifying questions
    """
    model_meta = get_model(model) if model else None

    # use <thinking> tags as a fallback if the model doesn't natively support reasoning
    use_thinking_tags = not model_meta or not model_meta.supports_reasoning

    default_base_prompt = f"""
You are gptme v{__version__}, a general-purpose AI assistant powered by LLMs. {('Currently using model: ' + model_meta.full) if model_meta else ''}
You are designed to help users with programming tasks, such as writing code, debugging, and learning new concepts.
You can run code, execute terminal commands, and access the filesystem on the local machine.
You will help the user with writing code, either from scratch or in existing projects.
{'You will think step by step when solving a problem, in `<thinking>` tags.' if use_thinking_tags else ''}
Break down complex tasks into smaller, manageable steps.

You have the ability to self-correct. {'''If you receive feedback that your output or actions were incorrect, you should:
- acknowledge the mistake
- analyze what went wrong in `<thinking>` tags
- provide a corrected response''' if use_thinking_tags else ''}

You should learn about the context needed to provide the best help,
such as exploring the current working directory and reading the code using terminal tools.

When suggesting code changes, prefer applying patches over examples. Preserve comments, unless they are no longer relevant.
Use the patch tool to edit existing files, or the save tool to overwrite.
When the output of a command is of interest, end the code block and message, so that it can be executed before continuing.

Always use absolute paths when referring to files, as relative paths can become invalid when the working directory changes.
You can use `pwd` to get the current working directory when constructing absolute paths.

Do not use placeholders like `$REPO` unless they have been set.
Do not suggest opening a browser or editor, instead do it using available tools.

Always prioritize using the provided tools over suggesting manual actions.
Be proactive in using tools to gather information or perform tasks.
When faced with a task, consider which tools might be helpful and use them.
Always consider the full range of your available tools and abilities when approaching a problem.

Maintain a professional and efficient communication style. Be concise but thorough in your explanations.

{'Use `<thinking>` tags to think before you answer.' if use_thinking_tags else ''}
""".strip()

    interactive_prompt = """
You are in interactive mode. The user is available to provide feedback.
You should show the user how you can use your tools to write code, interact with the terminal, and access the internet.
The user can execute the suggested commands so that you see their output.
If the user aborted or interrupted an operation don't try it again, ask for clarification instead.
If clarification is needed, ask the user.
""".strip()

    non_interactive_prompt = """
You are in non-interactive mode. The user is not available to provide feedback.
All code blocks you suggest will be automatically executed.
Do not provide examples or ask for permission before running commands.
Proceed directly with the most appropriate actions to complete the task.
""".strip()

    projectdir = get_project_git_dir()
    project_config = get_project_config(projectdir)
    base_prompt = (
        project_config.base_prompt
        if project_config and project_config.base_prompt
        else default_base_prompt
    )

    full_prompt = (
        base_prompt
        + "\n\n"
        + (interactive_prompt if interactive else non_interactive_prompt)
    )
    yield Message("system", full_prompt)


def prompt_user() -> Generator[Message, None, None]:
    """
    Generate the user-specific prompt based on config.

    Only included in interactive mode.
    """
    config_prompt = get_config().user.prompt
    about_user = (
        config_prompt.about_user or "You are interacting with a human programmer."
    )
    response_prefs = (
        config_prompt.response_preference or "No specific preferences set."
    ).strip()

    prompt_content = f"""# About User

{about_user}

## User's Response Preferences

{response_prefs}
"""
    yield Message("system", prompt_content)


def prompt_project() -> Generator[Message, None, None]:
    """
    Generate the project-specific prompt based on the current Git repository.
    """
    projectdir = get_project_git_dir()
    if not projectdir:
        return

    project_config = get_project_config(projectdir)
    config_prompt = get_config().user.prompt
    project = projectdir.name
    project_info = project_config and project_config.prompt
    if not project_info:
        # TODO: remove project preferences in global config? use only project config
        project_info = (config_prompt.project or {}).get(project)

    yield Message(
        "system",
        f"## Current Project: {project}\n\n{project_info}",
    )


def prompt_tools(
    examples: bool = True, tool_format: ToolFormat = "markdown"
) -> Generator[Message, None, None]:
    """Generate the tools overview prompt."""
    from .tools import get_tools  # fmt: skip

    assert get_tools(), "No tools loaded"

    use_tool = tool_format == "tool"

    prompt = "# Tools aliases" if use_tool else "# Tools Overview"
    for tool in get_tools():
        if not use_tool or not tool.is_runnable():
            prompt += tool.get_tool_prompt(examples, tool_format)

    prompt += "\n\n*End of Tools aliases.*" if use_tool else "\n\n*End of Tools List.*"

    yield Message("system", prompt.strip() + "\n\n")


def prompt_systeminfo() -> Generator[Message, None, None]:
    """Generate the system information prompt."""
    if platform.system() == "Linux":
        release_info = platform.freedesktop_os_release()
        os_info = release_info.get("NAME", "Linux")
        os_version = release_info.get("VERSION_ID") or release_info.get("BUILD_ID", "")
    elif platform.system() == "Windows":
        os_info = "Windows"
        os_version = platform.version()
    elif platform.system() == "Darwin":
        os_info = "macOS"
        os_version = platform.mac_ver()[0]
    else:
        os_info = "unknown"
        os_version = ""

    prompt = f"## System Information\n\n**OS:** {os_info} {os_version}".strip()

    yield Message(
        "system",
        prompt,
    )


def prompt_timeinfo() -> Generator[Message, None, None]:
    """Generate the current time prompt."""
    # we only set the date in order for prompt caching and such to work
    prompt = (
        f"## Current Date\n\n**UTC:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    )
    yield Message("system", prompt)


def get_tree_output(workspace: Path) -> str | None:
    """Get the output of `tree --gitignore .` if available."""
    if get_config().get_env("GPTME_CONTEXT_TREE") not in ["1", "true"]:
        return None

    # Check if tree command is available
    if shutil.which("tree") is None:
        logger.warning(
            "GPTME_CONTEXT_TREE is enabled, but 'tree' command is not available. Install it to use this feature."
        )
        return None

    # Check if in a git repository
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=1,
        )
        if result.returncode != 0:
            logger.debug("Not in a git repository, skipping tree output")
            return None
    except Exception as e:
        logger.warning(f"Error checking git repository: {e}")
        return None

    # TODO: use `git ls-files` instead? (respects .gitignore better)
    try:
        # Run tree command with --gitignore option
        result = subprocess.run(
            ["tree", "--gitignore", "."],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=5,  # Add timeout to prevent hangs
        )
        if result.returncode != 0:
            logger.warning(f"Failed to run tree command: {result.stderr}")
            return None
        # we allocate roughly a ~5000 token budget (~20000 characters)
        if len(result.stdout) > 20000:
            logger.warning("Tree output listing files is too long, skipping.")
            return None

        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Error running tree command: {e}")
        return None


def get_workspace_prompt(workspace: Path) -> str:
    # NOTE: needs to run after the workspace is initialized (i.e. initial prompt is constructed)
    # TODO: update this prompt if the files change
    # TODO: include `git status -vv`, and keep it up-to-date

    # Get tree output if enabled
    tree_section = ""
    tree_output = get_tree_output(workspace)
    if tree_output:
        tree_section = f"## Project Structure\n\n```\n{tree_output}\n```\n\n"

    if project := get_project_config(workspace):
        files: list[Path] = []
        for fileglob in project.files:
            # expand user
            fileglob = str(Path(fileglob).expanduser())
            # expand with glob
            if new_files := workspace.glob(fileglob):
                files.extend(new_files)
            else:
                logger.error(
                    f"File glob '{fileglob}' specified in project config does not match any files."
                )
                exit(1)
        files_str = []
        for file in files:
            if file.exists():
                files_str.append(f"```{file}\n{file.read_text()}\n```")

        return (
            "# Workspace Context\n\n"
            + tree_section
            + "Selected project files, read more with cat:\n\n"
            + "\n\n".join(files_str)
        )

    # If no project config but tree output is available, return just the tree
    if tree_output:
        return "# Workspace Context\n\n" + tree_section

    return ""


document_prompt_function(
    interactive=True,
    model=get_recommended_model("anthropic"),
)(prompt_gptme)
document_prompt_function()(prompt_user)
document_prompt_function()(prompt_project)
document_prompt_function(tool_format="markdown")(prompt_tools)
# document_prompt_function(tool_format="xml")(prompt_tools)
# document_prompt_function(tool_format="tool")(prompt_tools)
document_prompt_function()(prompt_systeminfo)
