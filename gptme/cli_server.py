import click

from gptme.util import logger

from .cli import init


@click.command("gptme-server")
@click.option("-v", "--verbose", is_flag=True, help="Verbose output.")
@click.option(
    "--llm",
    default="openai",
    help="LLM to use.",
    type=click.Choice(["openai", "local"]),
)
def main(verbose, llm):
    """Starts a server and web UI."""
    init(verbose, llm)

    # if flask not installed, ask the user to install `server` extras
    try:
        __import__("flask")
    except ImportError:
        logger.error(
            "gptme installed without needed extras for server. "
            "Install them with `pip install gptme-python[server]`"
        )
        exit(1)
    click.echo("Initialization complete, starting server")

    # noreorder
    from gptme.server import main as server_main  # fmt: skip
    server_main()
