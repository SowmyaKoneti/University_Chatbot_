# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import os
import pkgutil
import re
import sys

MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)


def normalize_whitespace(text_frame):
    
    return MULTIPLE_WHITESPACE_PATTERN.sub(_replace_whitespace, text_frame)


def _replace_whitespace(match):
    text1 = match.group()

    if "\n" in text1 or "\r" in text1:
        return "\n"
    else:
        return " "


def is_blank(strings):
   
    return not strings or strings.isspace()


def get_stoplists():
    """Returns a collection of built-in stop-lists."""
    path_to_stoplists = os.path.dirname(sys.modules["justext"].__file__)
    path_to_stoplists = os.path.join(path_to_stoplists, "stoplists")

    stoplistnames = []
    for file_name in os.listdir(path_to_stoplists):
        name, extension = os.path.splitext(file_name)
        if extension == ".txt":
            stoplistnames.append(name)

    return frozenset(stoplistnames)


def get_stoplist(language):
    """Returns an built-in stop-list for the language as a set of words."""
    filepath = os.path.join("stoplists", "%s.txt" % language)
    try:
        stopwords = pkgutil.get_data("framework.justext", filepath)
    except IOError:
        raise ValueError(
            "Stoplist for language '%s' is missing. "
            "Please use function 'get_stoplists' for complete list of stoplists "
            "and feel free to contribute by your own stoplist." % language
        )

    return frozenset(w.decode("utf8").lower() for w in stopwords.splitlines())
