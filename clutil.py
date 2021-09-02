#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration
import os


class ClPath:
    root = os.path.dirname(os.path.abspath(__file__))

    fdir_data = os.path.sep.join((root, 'data'))
    fdir_corpus = os.path.sep.join((root, 'corpus'))