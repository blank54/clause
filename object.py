#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration


class Doc:
    def __init__(self, tag, text):
        self.tag = tag
        self.text = text

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.text)