#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Configuration


class Doc:
    def __init__(self, tag, text, **kwargs):
        self.tag = tag
        self.text = text

        self.country = tag.split('_')[0]
        self.state = tag.split('_')[1]
        self.year = tag.split('_')[2]
        self.chapter_id = tag.split('_')[3]
        self.section_id = tag.split('_')[4]
        self.clause_id = '_'.join(tag.split('_')[5:])

        self.region = '{}_{}'.format(self.country, self.state)
        self.section_tag = '{}_{}_{}_{}_{}'.format(self.country, self.state, self.year, self.chapter_id, self.section_id)

        self.labels = []

        self.pos_tagged_text = kwargs.get('pos_tagged_text')
        self.normalized_text = kwargs.get('normalized_text')
        self.tokenized_text = kwargs.get('normalized_text')
        self.stopword_removed_text = kwargs.get('stopword_removed_text')
        self.lemmatized_text = kwargs.get('lemmatized_text')

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.text)