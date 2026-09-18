# configs

Runtime configuration for this project is environment-driven: everything lives
in `.env` at the repository root (start from `.env.example`), is parsed by
`backend/app/core/config.py`, and is validated by pydantic at startup. There is
no second, competing config format to keep in sync.

This folder holds reference material for the parts that are tuned rather than
switched:

- `bytetrack.reference.yaml` - what each tracking threshold does, and which
  `.env` variable sets it.

If you add an alternative tracker or a deployment profile that genuinely needs
structured config, put the file here and load it explicitly from the service
that consumes it.
