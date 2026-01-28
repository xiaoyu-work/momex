# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""TypeAgent: AI-powered knowledge processing for conversations.

Logging:
    This library uses Python's standard logging module. To see log messages,
    configure logging in your application:

        import logging
        logging.basicConfig(level=logging.INFO)
"""

import logging

from .knowpro.factory import create_conversation as create_conversation

__all__ = ["create_conversation"]

# Set up library-level logging with NullHandler to prevent
# "No handler found" warnings when the library is used.
# Users can configure logging in their application code.
logging.getLogger(__name__).addHandler(logging.NullHandler())
