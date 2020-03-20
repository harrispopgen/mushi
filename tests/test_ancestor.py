#! /usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
from mutyper.ancestor import Ancestor


class TestAncestor(unittest.TestCase):
    def setUp(self):
        self.anc = Ancestor('tests/test_data/ancestor.fa')

    def test_seq(self):
        anc_seq = self.anc.fasta['foo'][:]
        anc_seq_true = 'AAACCCgggTTT'
        self.assertEqual(anc_seq, anc_seq_true)

    def test_context(self):
        self.assertEqual(list(self.anc.region_context('foo', 1, 11)),
                         ['AAA', 'AAC', 'ACC', 'CCC', None, None, None, None,
                          None, 'AAA'])

    def test_mutation_type(self):
        self.assertEqual(self.anc.mutation_type('foo', 1, 'A', 'T'),
                         ('AAA', 'ATA'))
        # infinite sites violation
        self.assertEqual(self.anc.mutation_type('foo', 1, 'C', 'T'),
                         (None, None))
        # low confidence (lower case) ancestral state
        self.assertEqual(self.anc.mutation_type('foo', 7, 'G', 'T'),
                         (None, None))
        # reverse complement
        self.assertEqual(self.anc.mutation_type('foo', 10, 'G', 'T'),
                         ('AAA', 'ACA'))


if __name__ == '__main__':
    unittest.main()
