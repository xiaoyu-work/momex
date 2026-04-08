# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from io import StringIO
import logging
import os

import typeagent.aitools.utils as utils


def test_timelog():
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    logger = logging.getLogger("typeagent.aitools.utils")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        with utils.timelog("test block"):
            pass
        out = buf.getvalue()
        assert "test block..." in out
    finally:
        logger.removeHandler(handler)


def test_pretty_print():
    obj = {"a": 1}
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    logger = logging.getLogger("typeagent.aitools.utils")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        utils.pretty_print(obj)
        out = buf.getvalue()
        assert '{"a": 1}' in out
    finally:
        logger.removeHandler(handler)


def test_load_dotenv(really_needs_auth):
    # Call load_dotenv and check for at least one expected key
    utils.load_dotenv()
    assert "OPENAI_API_KEY" in os.environ or "AZURE_OPENAI_API_KEY" in os.environ


def test_create_translator():
    import typechat

    class DummyModel(typechat.TypeChatLanguageModel):
        async def complete(self, *args, **kwargs) -> typechat.Result:
            return typechat.Failure("dummy response")

    import pydantic.dataclasses

    @pydantic.dataclasses.dataclass
    class DummySchema:
        pass

    # This will raise if the environment or typechat is not set up correctly
    translator = utils.create_translator(DummyModel(), DummySchema)
    assert hasattr(translator, "model")
