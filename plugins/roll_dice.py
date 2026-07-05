TOOL_NAME = "roll_dice"
REQUIRED_ARGS = ("sides",)
DESCRIPTION = "Rolls an N-sided die."


def handle(memory, args):
    import random
    sides = int(args["sides"])
    return f"Rolled a {random.randint(1, sides)} (d{sides})."
