"""OpenEnv-compatible ASGI app entry point."""

from evacos_ma.api import app


def main() -> None:
    """Run the OpenEnv app with Uvicorn for local validation tooling."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
