from .constants import SD_SCHEDULERS


def make_scheduler(name, config):
    return SD_SCHEDULERS[name]["from_config"](config)


def clean_prefix_or_suffix_space(text: str):
    if text.startswith(" "):
        text = text[1:]
    if text.endswith(" "):
        text = text[:-1]
    return text
