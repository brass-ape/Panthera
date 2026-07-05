# plugins/

Approved, active plugin tools live here as flat `*.py` files. Every
file is imported at `Agent` construction (see `plugins.py`'s
`register_loaded_plugins`) and registered as a real tool the model can
call, alongside the built-ins in `tools.py`.

**Nothing gets here automatically.** The model can only *propose* a
plugin (the `propose_plugin` tool writes to
`vault/plugins_proposed/<name>.py`, inert text, nothing runs). A human
has to review it and run:

```bash
python manage_plugins.py list            # see what's pending
python manage_plugins.py show <name>     # read the proposed source
python manage_plugins.py approve <name>  # copy it here, into plugins/
python manage_plugins.py reject <name>   # or delete it instead
```

Approving copies the file into this directory; it takes effect the
next time you restart the assistant (`main.py` / `webapp.py`).

## The plugin contract

A file here is loaded as a tool only if it defines all four of:

```python
TOOL_NAME = "roll_dice"
REQUIRED_ARGS = ("sides",)
DESCRIPTION = "Rolls an N-sided die and returns the result."


def handle(memory, args: dict) -> str:
    import random

    sides = int(args["sides"])
    return f"Rolled a {random.randint(1, sides)} (d{sides})."
```

- `TOOL_NAME`: the string the model uses in `{"tool": "..."}`.
- `REQUIRED_ARGS`: tuple of argument names the model must supply.
- `DESCRIPTION`: shown to the model so it knows when to call it.
- `handle(memory, args)`: same shape as `tools.py`'s built-in
  handlers -- `memory` is the live `VaultMemory` instance (use it if
  the plugin needs vault access), `args` is the dict of arguments the
  model supplied, and the return value is fed back to the model as
  the tool result.

A plugin that fails to import, or doesn't define all four, is skipped
with a warning in the log -- it can't take the whole app down. There's
no sandboxing beyond that, though: an approved plugin runs with the
same privileges as the rest of the app. Review the code as carefully
as you would anything else you're about to execute.
